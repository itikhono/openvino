// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_strided_slice.hpp"

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

namespace {

int64_t get_unmodified_order_from(const std::shared_ptr<ov::op::v0::Constant>& transpose_order) {
    auto order_val = transpose_order->cast_vector<int64_t>();
    for (int64_t i = static_cast<int64_t>(order_val.size()) - 1; i >= 0; --i) {
        if (i != order_val[i]) {
            return i + 1;
        }
    }
    return 0;
}

std::vector<size_t> convert_mask_to_axis_vec(const std::vector<int64_t>& mask) {
    std::vector<size_t> axes;
    for (size_t i = 0; i < static_cast<size_t>(mask.size()); ++i) {
        if (mask[i] == 1)
            axes.push_back(i);
    }
    return axes;
};

}

TSStridedSliceForward::TSStridedSliceForward() {
    MATCHER_SCOPE(TSStridedSliceForward);
    create_pattern<ov::op::v1::StridedSlice>(true, {0});

    auto sinking_transformation = [=](const std::shared_ptr<Node>& main_node,
                                      const TransposeInputsInfo& transpose_info) -> bool {
        auto strided_slice = ov::as_type_ptr<ov::op::v1::StridedSlice>(main_node);
        if (!strided_slice) {
            return false;
        }

        auto data_partial_shape = strided_slice->input_value(0).get_partial_shape();
        auto data_rank = data_partial_shape.rank();
        if (data_rank.is_dynamic()) {
            return false;
        }

        // the ordinal number of first element from which Transpose does not change the order of the elements
        // e.g. transpose_order = (1, 0, 2, 3, 4)
        // unmodified_order_from = 2, (2, 3, 4) values are arranged in order
        auto unmodified_order_from = get_unmodified_order_from(transpose_info.transpose_const);
        auto data_rank_val = data_rank.get_length();

        // handle begin, end, stride inputs
        std::vector<std::shared_ptr<ov::op::v0::Constant>> new_inputs(3);
        for (size_t input_idx = 1; input_idx <= 3; ++input_idx) {
            auto input = strided_slice->input_value(input_idx);
            auto input_rank = input.get_partial_shape().rank();

            if (input_rank.is_dynamic()) {
                return false;
            }

            // if input_rank less than the unmodified_order_from,
            // then we need to extend the input rank to unmodified_order_from number
            // e.g. transpose_order = (1, 2, 0, 4, 5)
            // unmodified_order_from = 3
            // `begin` input rank = 2
            // we need to add at least one value = 0 to `begin` input to successfully execute Transpose operation
            if (input_rank.get_length() < unmodified_order_from) {
                auto input_const = ov::as_type_ptr<ov::op::v0::Constant>(input.get_node_shared_ptr());
                if (!input_const) {
                    return false;
                }

                auto input_const_val = input_const->cast_vector<int64_t>();
                if (input_idx == 1) {
                    // `begin` input have to be initialized with 0
                    input_const_val.resize(unmodified_order_from, 0);
                } else if (input_idx == 2) {
                    // 'end' input have to be initialized with the corresponding `data` input dim value
                    input_const_val.resize(unmodified_order_from);
                    for (size_t i = (unmodified_order_from - input_rank.get_length()); i < unmodified_order_from; ++i) {
                        // todo: we can use int max value and delete this check
                        if (data_partial_shape[i].is_dynamic()) {
                            return false;
                        }
                        input_const_val[i] = data_partial_shape[i].get_length();
                    }
                } else {
                    // `stride` input have to be initialized with 1
                    input_const_val.resize(unmodified_order_from, 1);
                }
                new_inputs[input_idx-1] = ov::op::v0::Constant::create(input_const->get_element_type(), {input_const_val.size()}, input_const_val);
            }
        }

        for (size_t i = 0; i < new_inputs.size(); ++i) {
            if (new_inputs[i]) {
                strided_slice->input(i).replace_source_output(new_inputs[i]);
            }
        }

        auto transpose_order_values = transpose_info.transpose_const->cast_vector<size_t>();
        auto update_mask = [&](std::vector<int64_t> old_mask) {
            old_mask.resize(data_rank_val, 0);
            std::vector<int64_t> new_mask(data_rank_val);
            for (size_t i = 0; i < old_mask.size(); ++i) {
                new_mask[i] = old_mask[transpose_order_values[i]];
            }
            return new_mask;
        };

        strided_slice->set_begin_mask(update_mask(strided_slice->get_begin_mask()));
        strided_slice->set_end_mask(update_mask(strided_slice->get_end_mask()));
        strided_slice->set_shrink_axis_mask(update_mask(strided_slice->get_shrink_axis_mask()));
        strided_slice->set_new_axis_mask(update_mask(strided_slice->get_new_axis_mask()));

        auto shrink_axes = convert_mask_to_axis_vec(strided_slice->get_shrink_axis_mask());
        auto new_axes = convert_mask_to_axis_vec(strided_slice->get_new_axis_mask());


        default_inputs_update(main_node, transpose_info);
        transpose_order_values = GetOrderAfterReduction(shrink_axes, transpose_order_values);
        transpose_order_values = GetOrderBeforeReduction(new_axes, transpose_order_values);

        auto new_transpose_order = std::make_shared<ov::op::v0::Constant>(transpose_info.transpose_const->get_element_type(),
                                                           Shape{transpose_order_values.size()},
                                                           transpose_order_values);

        TransposeInputsInfo transpose_input_info = {transpose_info.transpose, new_transpose_order, 0};
        default_outputs_update(main_node, transpose_input_info);
        return true;
    };

    transpose_sinking(matcher_name, sinking_transformation);
}

TSStridedSliceBackward::TSStridedSliceBackward() {
    MATCHER_SCOPE(TSStridedSliceBackward);

    auto main_node_label = wrap_type<ov::op::v1::StridedSlice>([](const Output<Node> &output) -> bool {
        return has_static_rank()(output) && CheckTransposeConsumers(output);
    });

    auto transpose_const_label = wrap_type<ov::op::v0::Constant>();

    auto transpose_label = wrap_type<ov::op::v1::Transpose>({main_node_label, transpose_const_label},
                                                            [](const Output<Node> &output) -> bool {
                                                                return has_static_rank()(output);
                                                            });

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();
        auto transpose_order = as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(transpose_const_label));
        auto transpose = pattern_to_output.at(transpose_label);
        auto main_node = pattern_to_output.at(main_node_label);
        if (transformation_callback(main_node)) {
            return false;
        }

        auto strided_slice = ov::as_type_ptr<ov::op::v1::StridedSlice>(main_node);
        if (!strided_slice) {
            return false;
        }

        auto data_partial_shape = strided_slice->input_value(0).get_partial_shape();
        auto data_rank = data_partial_shape.rank();
        if (data_rank.is_dynamic()) {
            return false;
        }

        // the ordinal number of first element from which Transpose does not change the order of the elements
        // e.g. transpose_order = (1, 0, 2, 3, 4)
        // unmodified_order_from = 2, (2, 3, 4) values are arranged in order
        auto unmodified_order_from = get_unmodified_order_from(transpose_order);
        auto data_rank_val = data_rank.get_length();

        // handle begin, end, stride inputs
        std::vector<std::shared_ptr<ov::op::v0::Constant>> new_inputs(3);
        for (size_t input_idx = 1; input_idx <= 3; ++input_idx) {
            auto input = strided_slice->input_value(input_idx);
            auto input_rank = input.get_partial_shape().rank();

            if (input_rank.is_dynamic()) {
                return false;
            }

            // if input_rank less than the unmodified_order_from,
            // then we need to extend the input rank to unmodified_order_from number
            // e.g. transpose_order = (1, 2, 0, 4, 5)
            // unmodified_order_from = 3
            // `begin` input rank = 2
            // we need to add at least one value = 0 to `begin` input to successfully execute Transpose operation
            if (input_rank.get_length() < unmodified_order_from) {
                auto input_const = ov::as_type_ptr<ov::op::v0::Constant>(input.get_node_shared_ptr());
                if (!input_const) {
                    return false;
                }

                auto input_const_val = input_const->cast_vector<int64_t>();
                if (input_idx == 1) {
                    // `begin` input have to be initialized with 0
                    input_const_val.resize(unmodified_order_from, 0);
                } else if (input_idx == 2) {
                    // 'end' input have to be initialized with the corresponding `data` input dim value
                    input_const_val.resize(unmodified_order_from);
                    for (size_t i = (unmodified_order_from - input_rank.get_length()); i < unmodified_order_from; ++i) {
                        // todo: we can use int max value and delete this check
                        if (data_partial_shape[i].is_dynamic()) {
                            return false;
                        }
                        input_const_val[i] = data_partial_shape[i].get_length();
                    }
                } else {
                    // `stride` input have to be initialized with 1
                    input_const_val.resize(unmodified_order_from, 1);
                }
                new_inputs[input_idx-1] = ov::op::v0::Constant::create(input_const->get_element_type(), {input_const_val.size()}, input_const_val);
            }
        }

        for (size_t i = 0; i < new_inputs.size(); ++i) {
            if (new_inputs[i]) {
                strided_slice->input(i).replace_source_output(new_inputs[i]);
            }
        }

        auto transpose_order_values = transpose_order->cast_vector<size_t>();
        auto update_mask = [&](std::vector<int64_t> old_mask) {
            old_mask.resize(data_rank_val, 0);
            std::vector<int64_t> new_mask(data_rank_val);
            for (size_t i = 0; i < old_mask.size(); ++i) {
                new_mask[i] = old_mask[transpose_order_values[i]];
            }
            return new_mask;
        };

        strided_slice->set_begin_mask(update_mask(strided_slice->get_begin_mask()));
        strided_slice->set_end_mask(update_mask(strided_slice->get_end_mask()));
        strided_slice->set_shrink_axis_mask(update_mask(strided_slice->get_shrink_axis_mask()));
        strided_slice->set_new_axis_mask(update_mask(strided_slice->get_new_axis_mask()));

        auto shrink_axes = convert_mask_to_axis_vec(strided_slice->get_shrink_axis_mask());
        auto new_axes = convert_mask_to_axis_vec(strided_slice->get_new_axis_mask());

        transpose_order_values = GetOrderBeforeReduction(shrink_axes, transpose_order_values);
        transpose_order_values = GetOrderAfterReduction(new_axes, transpose_order_values);
        auto new_transpose_order = std::make_shared<ov::op::v0::Constant>(transpose_order->get_element_type(),
                                                                          Shape{transpose_order_values.size()},
                                                                          transpose_order_values);

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(main_node, new_transpose_order)) {
            register_new_node(new_node);
        }
        main_node->validate_and_infer_types();
        RemoveTransposeConsumers(main_node);
        return true;
    };

    auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}