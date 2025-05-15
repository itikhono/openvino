#include "transformations/utils/block_collection.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/pack_mha.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/core/graph_util.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::pass;
using namespace ov::pass::pattern;

namespace {

std::tuple<ov::NodeVector, Node*> get_sdpa_order(const std::unordered_set<Node*>& post_sdpa_proj) {
    ov::NodeVector post_sdpa_ordered;

    // find the first Add in the chain:
    // AddN(Add2(Add1(post_sdpa_proj_1, post_sdpa_proj_2), post_sdpa_proj_3), post_sdpa_proj_N + 1)
    ov::op::v1::Add* current_add = nullptr;
    for (const auto& proj_node : post_sdpa_proj) {
        std::cout << "proj_node: " << proj_node << "\n";
        const auto& output = proj_node->output(0);
        const auto& targets = output.get_target_inputs();

        if (targets.size() != 1)
            continue;  // Only proceed if there's a single consumer

        const auto& input = *targets.begin();
        auto add_node = ov::as_type<ov::op::v1::Add>(input.get_node());
        if (!add_node)
            continue;

        auto lhs = add_node->input_value(0).get_node();
        auto rhs = add_node->input_value(1).get_node();

        if (post_sdpa_proj.count(lhs) && post_sdpa_proj.count(rhs)) {
            current_add = add_node;
            post_sdpa_ordered.push_back(lhs->shared_from_this());
            post_sdpa_ordered.push_back(rhs->shared_from_this());
            break;
        }
    }

    if (!current_add) {
        std::cout << "Can't find the first Add in the chain\n";
        return {};
    }

    // find the remaining post_sdpa_proj order, starting from the first Add
    // AddN(Add2(Add1(post_sdpa_proj_1, post_sdpa_proj_2), post_sdpa_proj_3), post_sdpa_proj_N + 1)
    while (true) {
        const auto& targets = current_add->output(0).get_target_inputs();
        if (targets.size() != 1) {
            std::cout << "Unexpected number of targets for Add: " << targets.size() << "\n";
            return {};
        }

        auto next_node_input = targets.begin();
        current_add = ov::as_type<ov::op::v1::Add>(next_node_input->get_node());
        if (!current_add)
            break;

        auto another_idx = 1 - next_node_input->get_index();
        auto input_node = current_add->input_value(another_idx).get_node();
        if (post_sdpa_proj.count(input_node)) {
            post_sdpa_ordered.push_back(input_node->shared_from_this());
        } else {
            std::cout << "Unexpected node in Add chain: " << current_add << "\n";
            break;
        }
    }
    return {post_sdpa_ordered, current_add};
}

Output<Node> fuse_weights_and_replace(const ov::NodeVector& ordered_nodes, int64_t input_node_idx, int64_t concat_dim_idx) {
    using namespace ov;
    using namespace ov::op;

    Output<Node> in = ordered_nodes[0]->input_value(0);
    OutputVector all_weights;

    for (const auto& node : ordered_nodes) {
        all_weights.push_back(node->input_value(input_node_idx));
    }

    auto fused_weights = std::make_shared<v0::Concat>(all_weights, concat_dim_idx);
    // Clone and replace the first node with new fused weights
    auto fused_op = ordered_nodes[0]->clone_with_new_inputs({ordered_nodes[0]->input_value(1 - input_node_idx), fused_weights});
    
    ov::replace_node(ordered_nodes[0], fused_op);
    return fused_op->output(0);
}

}

PackMHA::PackMHA() : MultiMatcher("PackMHA") {
    // input -> Normalization Pattern:
    auto norm_input = any_input();
    auto norm_block = blocks::l2_norm_block(norm_input);

    // Attention mask
    //auto attention_mask = blocks::attention_mask();

    // Projection -> SDPA preprocessing (RoPE?) Pattern:
    //auto proj_input = any_input();
    auto constant = wrap_type<v0::Constant>();
    auto convert = optional<v0::Convert>(constant);

    auto dq = blocks::dq_constant_block();
    auto mm = wrap_type<v0::MatMul>({norm_block, dq | convert});
    auto bias = wrap_type<v1::Add>({mm, any_input()});
    //auto pre_sdpa = blocks::sdpa_preprocessing_block(proj_block);

    // SDPA + linear projection Pattern:
    auto q_input = any_input();
    auto k_input = any_input();
    auto v_input = any_input();

    auto q = blocks::sdpa_preprocessing_block(q_input);
    auto k = blocks::sdpa_preprocessing_block(k_input);

    auto reshape_v = wrap_type<v1::Reshape>({v_input, any_input()});
    auto vT = optional<v1::Transpose>({reshape_v, any_input()});

    auto sdpa = blocks::sdpa_block(q, k, vT);
    auto t2 = wrap_type<v1::Transpose>({sdpa, any_input()});
    auto reshaped = wrap_type<v1::Reshape>({t2, any_input()});
    auto proj = wrap_type<v0::MatMul>({reshaped, any_input()});

    auto callback = [=](const std::unordered_map<std::shared_ptr<Node>, std::vector<PatternValueMap>>& matches) {
        using namespace ov::pass::pattern::op;

        // DEBUG
        std::cout << "matches size  " << matches.size() << std::endl;
        for (const auto& [root, patterns] : matches) {
            std::cout << "pattern size  " << patterns.size() << std::endl;
            for (const auto& m : patterns) {
                std::cout << "pattern = " << root->get_friendly_name() << std::endl;
                std::cout << m.at(root->shared_from_this()).get_node()->get_friendly_name() << std::endl;
            }
        }
        // DEBUG END

        std::unordered_set<Node*> post_sdpa_proj;
        std::unordered_map<Node*, const PatternValueMap*> node_to_proj_pm;
        for (const auto& pm : matches.at(proj)) {
            auto root_node = pm.at(proj).get_node();
            post_sdpa_proj.insert(root_node);
            node_to_proj_pm[root_node] = &pm;
        }

        std::unordered_map<Node*, const PatternValueMap*> node_to_bias_pm;
        for (const auto& pm : matches.at(bias)) {
            auto root_node = pm.at(bias).get_node();
            node_to_bias_pm[root_node] = &pm;
        }

        auto [post_sdpa_proj_ordered, node_after_mha] = get_sdpa_order(post_sdpa_proj);
        if (post_sdpa_proj_ordered.empty()) {
            return;
        }

        ov::NodeVector q_matmuls, q_biases,
                       k_matmuls, k_biases,
                       v_matmuls, v_biases,
                       ordered_proj;

        for (const auto& node : post_sdpa_proj_ordered) {
            const auto* pm = node_to_proj_pm.at(node.get());

            // Collect post SDPA projections
            auto matmul = pm->at(proj);
            ordered_proj.push_back(matmul.get_node_shared_ptr());

            // Collect pre-SDPA biases and projections for Q,K,V
            auto q_pm = node_to_bias_pm.at(pm->at(q_input).get_node());
            auto k_pm = node_to_bias_pm.at(pm->at(k_input).get_node());
            auto v_pm = node_to_bias_pm.at(pm->at(v_input).get_node());

            q_matmuls.push_back(q_pm->at(mm).get_node_shared_ptr());
            k_matmuls.push_back(k_pm->at(mm).get_node_shared_ptr());
            v_matmuls.push_back(v_pm->at(mm).get_node_shared_ptr());

            q_biases.push_back(q_pm->at(bias).get_node_shared_ptr());
            k_biases.push_back(k_pm->at(bias).get_node_shared_ptr());
            v_biases.push_back(v_pm->at(bias).get_node_shared_ptr());
        }

        fuse_weights_and_replace(q_matmuls, 1, 1);
        fuse_weights_and_replace(q_biases, 0, 2);

        fuse_weights_and_replace(k_matmuls, 1, 1);
        fuse_weights_and_replace(k_biases, 0, 2);

        fuse_weights_and_replace(v_matmuls, 1, 1);
        fuse_weights_and_replace(v_biases, 0, 2);

        //auto fused = fuse_weights_and_replace(ordered_proj, 1, 1);
        //std::cout << "XXXXXXX fused " << fused << std::endl;
        //int64_t num_heads = post_sdpa_proj_ordered.size();

        const auto* proj_pm = node_to_proj_pm.at(post_sdpa_proj_ordered[0].get());
        auto proj_transpose = proj_pm->at(t2).get_node_shared_ptr();

        auto axis_0 = v0::Constant::create(element::i64, Shape{1}, {2});
        auto reduce_0 = std::make_shared<v1::ReduceSum>(proj_transpose, axis_0, false);
       
        auto proj_reshape = proj_pm->at(reshaped).get_node_shared_ptr();
        proj_reshape->input(0).replace_source_output(reduce_0->output(0));

        // ADD FUSING
        // Originally, we have a chain: Add(...Add(Add(MatMul1, MatMul2), MatMul3)..., MatMulN)
        // Replaced with single ReduceSum + Reshape
        //

        // todo: check norm_input , this is input(x) to the model, detect idx
        auto proj_matmul = proj_pm->at(proj).get_node_shared_ptr();
        node_after_mha->input(0).replace_source_output(proj_matmul);
    };

    register_patterns({bias, proj},
                      callback,
                      true);
}
