// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tensorflow_frontend/frontend.hpp>
#include <tensorflow_frontend/model.hpp>

//#include <ngraph/pass/transpose_sinking.h>
#include <ngraph/pass/constant_folding.hpp>

#include "op_table.hpp"
#include "tf_framework_node.hpp"

using namespace google;

using namespace ngraph::frontend;
using namespace ngraph::frontend::tf;
using namespace ::tensorflow::ngraph_bridge;

using ::tensorflow::GraphDef;

namespace {
void TranslateGraph(const std::shared_ptr<ngraph::frontend::InputModelTF>& model,
                    const std::string model_name,
                    bool fail_fast,
                    bool no_conversion,
                    std::shared_ptr<ngraph::Function>& ng_function) {
    using OpMap = std::unordered_map<std::string, std::vector<ngraph::Output<ngraph::Node>>>;
    // a map from operation names to generated nGraph Output<TFNodeDecoder>
    OpMap ng_op_map;

    ngraph::ParameterVector params;
    ngraph::ResultVector results;
    const auto& operation_places = model->get_op_places();
    const auto& model_inputs = model->get_inputs();
    const auto& model_outputs = model->get_outputs();
    const auto& model_frozen_inputs = model->get_tensor_values();

    std::map<const std::string, const std::function<ngraph::OutputVector(const NodeContext&)>> translate_map;

    const auto TRANSLATE_OP_MAP = ::tensorflow::ngraph_bridge::get_supported_ops();
    if (no_conversion) {
        const std::set<std::string> required_types{"Placeholder", "_Retval", "NoOp"};
        for (auto& name : required_types) {
            translate_map.emplace(name, TRANSLATE_OP_MAP.at(name));
        }
    } else {
        translate_map = TRANSLATE_OP_MAP;
    }

    // fill ng_op_map with Constant outputs for frozen inputs
    for (const auto& frozen_input : model_frozen_inputs) {
        const auto& frozen_input_name = frozen_input.first;
        const auto& frozen_input_value = frozen_input.second;
        ng_op_map[frozen_input_name] = {frozen_input_value};
    }

    // create parameter nodes for all tensor places corresponding to inputs
    for (const auto& input_place : model_inputs) {
        FRONT_END_GENERAL_CHECK(input_place->get_names().size() == 1, "Input place must have one name.");
        auto input_name = input_place->get_names()[0];
        if (ng_op_map.count(input_name)) {
            // probably this input is frozen
            continue;
        }
        const auto& input_tensor_place = std::dynamic_pointer_cast<TensorPlaceTF>(input_place);
        auto input_shape = input_tensor_place->get_partial_shape();
        auto input_type = input_tensor_place->get_element_type();

        auto input_ng_output = ConstructNgNode<opset::Parameter>(input_name, input_type, input_shape);
        auto input_ng_node = std::dynamic_pointer_cast<opset::Parameter>(input_ng_output.get_node_shared_ptr());
        params.push_back(input_ng_node);
        ng_op_map[input_name] = {input_ng_output};
    }

    // create the nGraph ops from TensorFlow ops
    for (auto& operation_place : operation_places) {
        auto operation_decoder = operation_place->get_decoder();
        auto operation_name = operation_place->get_names()[0];

        // output for parameter nodes has been already generated
        if (ng_op_map.count(operation_name)) {
            continue;
        }

        // prepare a list of nGraph node inputs for each node
        ngraph::OutputVector ng_inputs;
        ::ngraph::frontend::tf::NamedInputs named_inputs;
        for (size_t input_port_idx = 0; input_port_idx < operation_decoder->get_input_size(); ++input_port_idx) {
            std::string producer_name;
            size_t producer_port_idx;
            try {
                operation_decoder->get_input_node(input_port_idx, producer_name, producer_port_idx);
            } catch (const std::exception& e) {
                FRONT_END_THROW("[ ERROR ] Exception happened when preparing input " + std::to_string(input_port_idx) +
                                " for op '" + operation_decoder->get_op_name() + "', expected input name: '" +
                                producer_name + "', expected input port index: " + std::to_string(producer_port_idx) +
                                '\n');
            }
            // TODO: re-implement the logic below once Place graph structure is implemented
            // Using Place graph structure (OpPlace, In/OutPortPlace places and their connections) can give
            // names of ports and operations that can be used for further check about existence in ng_op_map

            // check if output vector for places have been already defined and the order of this check is important
            // it moves from places corresponding to input port of the current operation node to output port of original
            // producers
            if (ng_op_map.count(std::to_string(input_port_idx) + ":" + operation_name)) {
                const auto& input_outputs_vector = ng_op_map.at(std::to_string(input_port_idx) + ":" + operation_name);
                FRONT_END_GENERAL_CHECK(input_outputs_vector.size() == 1,
                                        "Input created with pruning must have one output");
                ng_inputs.push_back(input_outputs_vector.at(0));
                named_inputs[input_port_idx] = {input_outputs_vector.at(0)};
            } else if (ng_op_map.count(producer_name + ":" + std::to_string(producer_port_idx))) {
                const auto& input_outputs_vector =
                    ng_op_map.at(producer_name + ":" + std::to_string(producer_port_idx));
                FRONT_END_GENERAL_CHECK(input_outputs_vector.size() == 1,
                                        "Input created with pruning must have one output");
                ng_inputs.push_back(input_outputs_vector.at(0));
                named_inputs[input_port_idx] = {input_outputs_vector.at(0)};
            } else if (ng_op_map.count(producer_name)) {
                const auto& input_outputs_vector = ng_op_map.at(producer_name);
                FRONT_END_GENERAL_CHECK(input_outputs_vector.size() > producer_port_idx,
                                        "Input created with pruning must have one output");
                ng_inputs.push_back(input_outputs_vector.at(producer_port_idx));
                named_inputs[input_port_idx] = {input_outputs_vector.at(producer_port_idx)};
            } else {
                FRONT_END_GENERAL_CHECK(false,
                                        "No input is found for node \"" + operation_name + "\" by port" +
                                            std::to_string(producer_port_idx));
            }
        }

        // generate nGraph node output vector for the current operation node
        ngraph::OutputVector ng_outputs;
        try {
            /*
            if (operation_decoder->IsControlFlow()) {
                FRONT_END_THROW("Encountered a control flow op in the nGraph bridge: " +
                                operation_decoder->DebugString());
            }
            */

            FRONT_END_OP_CONVERSION_CHECK(translate_map.count(operation_decoder->get_op_type()),
                                          "No translator found for " + operation_decoder->get_op_type() + " node.");
            auto op_fun = &(translate_map[operation_decoder->get_op_type()]);
            // NodeContext node_context(ng_inputs, operation_decoder, model_inputs);
            // TODO: Check why NodeContextNew doesn't have ngOutputVector ng_inputs input in constructor
            ::ngraph::frontend::tf::NodeContext node_context(*operation_decoder.get(), named_inputs);
            // generate nGraph node output vector using translator for given operation type
            ng_outputs = (*op_fun)(node_context);
        } catch (...) {
            if (fail_fast) {
                // re-throw any exception
                throw;
            } else {
                auto ng_node =
                    std::make_shared<ngraph::frontend::TFFrameworkNode>(operation_decoder,
                                                                        ng_inputs,
                                                                        operation_place->get_output_ports().size());
                SetTracingInfo(operation_name, ng_node);
                ng_outputs = ng_node->outputs();
            }
        }

        // register nGraph node outputs in the map for new operation node
        for (auto output : ng_outputs) {
            if (auto result = std::dynamic_pointer_cast<opset::Result>(output.get_node_shared_ptr())) {
                // do not add RetVal type operation to ng_op_map
                results.push_back(result);
            } else {
                if (auto param = std::dynamic_pointer_cast<opset::Parameter>(output.get_node_shared_ptr())) {
                    params.push_back(param);
                }
                ng_op_map[operation_name].push_back(output);
            }
        }
    }

    // create Result nodes for all model outputs
    for (const auto& model_output : model_outputs) {
        auto model_output_tensor_place = std::dynamic_pointer_cast<TensorPlaceTF>(model_output);
        auto model_output_name = model_output_tensor_place->get_names()[0];
        std::string operation_name;
        std::string port_type;
        size_t port_index;
        ngraph::frontend::tf::extract_operation_name_and_port(model_output_name, operation_name, port_index, port_type);

        if (port_type == "none") {
            for (const auto& node_output : ng_op_map[operation_name]) {
                results.push_back(std::make_shared<default_opset::Result>(node_output));
            }
        } else if (port_type == "out") {
            const auto& node_outputs = ng_op_map[operation_name];
            FRONT_END_GENERAL_CHECK(node_outputs.size() > port_index,
                                    "Output port with index " + std::to_string(port_index) + " of " + operation_name +
                                        "node specified as custom output does not exist");
            results.push_back(std::make_shared<default_opset::Result>(node_outputs[port_index]));
        } else if (port_type == "in") {
            // TODO: avoid this traversing by having a map for OpPlace objects, for example
            std::shared_ptr<OpPlaceTF> operation_place = nullptr;
            for (const auto& op_place : operation_places) {
                if (op_place->get_names()[0].compare(operation_name) == 0) {
                    operation_place = op_place;
                }
            }
            FRONT_END_GENERAL_CHECK(operation_place, "There is no operation place with a name: " + operation_name);
            auto operation_decoder = operation_place->get_decoder();

            // get to know a producer node and by which its output port data is generated
            std::string producer_name;
            size_t producer_port_idx;
            try {
                operation_decoder->get_input_node(port_index, producer_name, producer_port_idx);
            } catch (const std::exception& e) {
                FRONT_END_THROW("[ ERROR ] Exception happened when preparing input " + std::to_string(port_index) +
                                " for op '" + operation_decoder->get_op_name() + "', expected input name: '" +
                                producer_name + "', expected input port index: " + std::to_string(producer_port_idx) +
                                '\n');
            }

            // add Result node for this producer output port
            const auto& node_outputs = ng_op_map[producer_name];
            FRONT_END_GENERAL_CHECK(node_outputs.size() > producer_port_idx,
                                    "Output port with index " + std::to_string(producer_port_idx) + " of " +
                                        producer_name + "node specified as custom output does not exist");
            results.push_back(std::make_shared<default_opset::Result>(node_outputs[producer_port_idx]));
        }
    }

    // find all terminal nodes in ngraph graph to complete list of results
    if (results.empty()) {
        for (const auto& node_output_vector : ng_op_map) {
            for (auto output : node_output_vector.second) {
                if (output.get_target_inputs().empty() &&
                    !std::dynamic_pointer_cast<opset::Result>(output.get_node_shared_ptr())) {
                    results.push_back(std::make_shared<default_opset::Result>(output));
                }
            }
        }
    }

    // TODO: reorder results and params according to indices given in RT info (if any)

    // create the nGraph function
    ng_function = std::make_shared<ngraph::Function>(results, params, model_name);

    // TODO: request row-major layout on results.
    // why do we need this?
    // for (auto result : ng_function->get_results()) {
    //  result->set_needs_default_layout(true);
    // }
    NGRAPH_VLOG(5) << "Done with translations";
}

void TranslateFWNode(const std::shared_ptr<TFFrameworkNode>& node) {
    auto type = node->get_op_type();

    const auto TRANSLATE_OP_MAP = get_supported_ops();
    auto translator_it = TRANSLATE_OP_MAP.find(type);
    FRONT_END_OP_CONVERSION_CHECK(translator_it != TRANSLATE_OP_MAP.end(), "No translator found for ", type, " node.");

    ngraph::OutputVector ng_inputs;
    NamedInputs named_inputs;
    size_t input_port_idx = 0;
    for (auto& input : node->inputs()) {
        ng_inputs.push_back(input.get_source_output());
        named_inputs[input_port_idx++] = {input.get_source_output()};
    }

    NodeContext node_ctx(*node->get_decoder(), named_inputs);
    auto new_node_outputs = translator_it->second(node_ctx);
    SetTracingInfo(node_ctx.get_name(), new_node_outputs.front());

    auto new_output = new_node_outputs.begin();
    auto old_outputs = node->outputs();
    auto old_output = old_outputs.begin();

    for (; new_output != new_node_outputs.end() && old_output != old_outputs.end(); ++old_output, ++new_output) {
        old_output->replace(*new_output);
    }
}
}  // namespace

/// \brief Check if FrontEndTensorflow can recognize model from given parts
bool FrontEndTF::supported_impl(const std::vector<std::shared_ptr<Variant>>& variants) const {
    // TODO: Support TensorFlow 2 SavedModel format
    if (variants.empty() || variants.size() > 2)
        return false;

    // Validating first path, it must contain a model
    if (ov::is_type<VariantWrapper<std::string>>(variants[0])) {
        std::string suffix = ".pb";
        std::string model_path = ov::as_type_ptr<VariantWrapper<std::string>>(variants[0])->get();
        if (tf::endsWith(model_path, suffix)) {
            return true;
        }
    }
    return false;
}

InputModel::Ptr FrontEndTF::load_impl(const std::vector<std::shared_ptr<Variant>>& variants) const {
    // TODO: convert any format variant to GraphIterator and pass it to the single constuctor
    // InputModelTF with GraphIterator
    if (variants.size() == 1) {
        // a case when protobuf format or SavedModel dir format is provided
        if (ov::is_type<VariantWrapper<std::string>>(variants[0])) {
            std::string m_path = ov::as_type_ptr<VariantWrapper<std::string>>(variants[0])->get();
            return std::make_shared<InputModelTF>(m_path);
        }
    }
    return nullptr;
}

std::shared_ptr<ngraph::Function> FrontEndTF::convert(InputModel::Ptr model) const {
    auto model_tf = std::dynamic_pointer_cast<ngraph::frontend::InputModelTF>(model);
    std::cout << "[ INFO ] FrontEndTensorflow::convert invoked\n";

    std::shared_ptr<ngraph::Function> f;
    TranslateGraph(model_tf, "here_should_be_a_graph_name", true, false, f);
    std::cout << "[ STATUS ] TranslateGraph was called successfuly.\n";
    std::cout << "[ INFO ] Resulting nGraph function contains " << f->get_ops().size() << " nodes." << std::endl;

    normalize(f);

    // TODO: check that nGraph function does not contain operations which are not in the opset

    return f;
}

std::shared_ptr<ngraph::Function> FrontEndTF::convert_partially(InputModel::Ptr model) const {
    auto model_tf = std::dynamic_pointer_cast<ngraph::frontend::InputModelTF>(model);
    std::cout << "[ INFO ] FrontEndTensorflow::convert_partially invoked\n";

    std::shared_ptr<ngraph::Function> f;
    TranslateGraph(model_tf, "here_should_be_a_graph_name", false, false, f);
    std::cout << "[ STATUS ] TranslateGraph was called successfuly.\n";
    std::cout << "[ INFO ] Resulting nGraph function contains " << f->get_ops().size() << " nodes." << std::endl;

    normalize(f);
    return f;
}

std::shared_ptr<ngraph::Function> FrontEndTF::decode(InputModel::Ptr model) const {
    auto model_tf = std::dynamic_pointer_cast<ngraph::frontend::InputModelTF>(model);
    std::cout << "[ INFO ] FrontEndTensorflow::decode invoked\n";

    std::shared_ptr<ngraph::Function> f;
    TranslateGraph(model_tf, "here_should_be_a_graph_name", false, true, f);
    std::cout << "[ STATUS ] TranslateGraphFWNode was called successfuly.\n";
    std::cout << "[ INFO ] Resulting nGraph function contains " << f->get_ops().size() << " nodes." << std::endl;
    return f;
}

void FrontEndTF::convert(std::shared_ptr<ngraph::Function> partiallyConverted) const {
    for (const auto& node : partiallyConverted->get_ordered_ops()) {
        if (ov::is_type<TFFrameworkNode>(node)) {
            TranslateFWNode(std::dynamic_pointer_cast<TFFrameworkNode>(node));
        }
    }
    for (auto result : partiallyConverted->get_results()) {
        result->validate_and_infer_types();
    }

    normalize(partiallyConverted);
}

void FrontEndTF::normalize(std::shared_ptr<ngraph::Function> function) const {
    std::cout << "[ STATUS ] Running Transpose Sinking transformation\n";

    ngraph::pass::Manager manager;
    // manager.register_pass<ngraph::pass::TransposeSinking>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.run_passes(function);

    std::cout << "[ INFO ] Resulting nGraph function contains " << function->get_ops().size() << " nodes." << std::endl;
}
