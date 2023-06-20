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

int64_t get_unmodified_order_from(const TransposeInputsInfo& transpose_info) {
    const auto& order = transpose_info.transpose_const;
    auto order_val = order->cast_vector<int64_t>();
    for (int64_t i = static_cast<int64_t>(order_val.size()) - 1; i >= 0; --i) {
        if (i != order_val[i]) {
            return i + 1;
        }
    }
    return 0;
}

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
        auto unmodified_order_from = get_unmodified_order_from(transpose_info);
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

        // remove Transpose on 1st input:
        auto transpose_parent = transpose_info.transpose->input_value(0);
        main_node->input(0).replace_source_output(transpose_parent);

        const auto transpose_axis_order = transpose_info.transpose_const->get_axis_vector_val();
        auto update_mask = [&](std::vector<int64_t> old_mask) {
            old_mask.resize(data_rank_val, 0);
            std::vector<int64_t> new_mask(data_rank_val);
            for (size_t i = 0; i < old_mask.size(); ++i) {
                new_mask[i] = old_mask[transpose_axis_order[i]];
            }
            return new_mask;
        };

        strided_slice->set_begin_mask(update_mask(strided_slice->get_begin_mask()));
        strided_slice->set_end_mask(update_mask(strided_slice->get_end_mask()));
        strided_slice->set_shrink_axis_mask(update_mask(strided_slice->get_shrink_axis_mask()));
        strided_slice->set_new_axis_mask(update_mask(strided_slice->get_new_axis_mask()));

        auto data = std::make_shared<ov::op::v0::Constant>(element::i32,
                                                           Shape{transpose_axis_order.size()},
                                                           transpose_axis_order);


        default_outputs_update(main_node, transpose_info);
        return true;
    };

    transpose_sinking(matcher_name, sinking_transformation);
}

TSStridedSliceBackward::TSStridedSliceBackward() {
    MATCHER_SCOPE(TSStridedSliceBackward);

    auto main_node_label = wrap_type<ov::op::v8::Slice>([](const Output<Node>& output) -> bool {
        return has_static_rank()(output) && CheckTransposeConsumers(output);
    });

    auto transpose_const_label = wrap_type<ov::op::v0::Constant>();

    auto transpose_label = wrap_type<ov::op::v1::Transpose>({main_node_label, transpose_const_label},
                                                            [](const Output<Node>& output) -> bool {
                                                                return has_static_rank()(output);
                                                            });

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose_const =
                as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(transpose_const_label).get_node_shared_ptr());
        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto main_node = pattern_to_output.at(main_node_label).get_node_shared_ptr();
        if (transformation_callback(main_node)) {
            return false;
        }

        if (main_node->get_input_size() < 5) {
            return false;
        }

        for (auto& new_node : sink_backward::InsertTransposeBeforeNode(main_node,
                                                                       transpose_const,
                */
/* input_indexes= */ /*
  {0})) {
             register_new_node(new_node);
         }

         RemoveTransposeConsumers(main_node);
         const auto transpose_axis_order = transpose_const->get_axis_vector_val();
         const auto reversed_transpose_order = ReverseTransposeOrder(transpose_axis_order);
         auto axis = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, std::vector<int32_t>{0});
         auto data = std::make_shared<ov::op::v0::Constant>(element::i32,
                                                            Shape{reversed_transpose_order.size()},
                                                            reversed_transpose_order);
         const auto& indices = main_node->input_value(4);
         auto new_axis = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
         main_node->input(4).replace_source_output(new_axis);

         main_node->validate_and_infer_types();
         return true;
     };

     auto m = std::make_shared<Matcher>(transpose_label, matcher_name);
     register_matcher(m, matcher_pass_callback);
 }
