// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_strided_slice.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "ts_test_case.hpp"
#include "ts_test_utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::element;
using namespace ov::opset12;
using namespace ov::pass::transpose_sinking;
using namespace transpose_sinking::testing::utils;

namespace transpose_sinking {
namespace testing {
namespace strided_slice {

struct StridedSliceMasks {
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> new_axis;
    std::vector<int64_t> shrink_axis;
    std::vector<int64_t> ellipsis;
};

class StridedSliceFactory : public IFactory {
public:
    explicit StridedSliceFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        if (parent_nodes.size() == 4) {
            return make_shared<StridedSlice>(parent_nodes[0],
                                             parent_nodes[1],
                                             parent_nodes[2],
                                             parent_nodes[3],
                                             m_masks.begin,
                                             m_masks.end,
                                             m_masks.new_axis,
                                             m_masks.shrink_axis,
                                             m_masks.ellipsis);
        }
        OPENVINO_ASSERT(false, "Unexpected number of inputs to StridedSlice operation.");
    }

    void set_masks(const StridedSliceMasks& masks) {
        m_masks = masks;
    }
private:
    StridedSliceMasks m_masks;
};

shared_ptr<StridedSliceFactory> CreateStridedSliceFactory(const std::string& type_name) {
    return std::make_shared<StridedSliceFactory>(type_name);
}
// ----------------------------------------------------------------------------

#undef CREATE_STRIDED_SLICE_FACTORY
#define CREATE_STRIDED_SLICE_FACTORY(type_name) CreateStridedSliceFactory(#type_name)
// ----------------------------------------------------------------------------

shared_ptr<ov::Model> create_model(size_t main_node_idx,
                                   const ModelDescription& model_desc,
                                   size_t num_ops,
                                   const OutputVector& inputs_to_main) {
    auto new_inputs = model_desc.preprocess_inputs_to_main.apply(inputs_to_main);
    auto main_node = create_main_node(new_inputs, num_ops, model_desc.main_op[main_node_idx]);
    auto outputs = model_desc.preprocess_outputs_of_main.apply(main_node->outputs());
    return make_shared<ov::Model>(outputs, filter_parameters(inputs_to_main));
}

auto wrapper = [](const TestCase& test_case) {
    OPENVINO_ASSERT(test_case.model.main_op.size() == test_case.model_ref.main_op.size(),
                    "The number of main op (testing op) creator have to be the same for the testing model and for"
                    "the reference model.");
    return ::testing::Combine(::testing::Range<size_t>(0, test_case.num_main_ops.size()),
                              ::testing::Range<size_t>(0, test_case.model.main_op.size()),
                              ::testing::Values(test_case));
};

struct StridedSliceForwardArguments {
    OutputVector inputs_to_main;
    StridedSliceMasks masks;
    vector<int64_t> reference_transpose_order;
};

auto test_forward_strided_slice = [](const StridedSliceForwardArguments& test_arguments) {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSStridedSliceForward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = test_arguments.inputs_to_main;

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    auto strided_slice_factory = CREATE_STRIDED_SLICE_FACTORY(StridedSlice);
    strided_slice_factory->set_masks(test_arguments.masks);
    test_case.model.main_op = {strided_slice_factory};
    test_case.model.model_template = create_model;

    // Reference model description:
    auto change_mask_order = [](std::vector<int64_t> mask) {
        std::reverse(mask.begin(), mask.end());
        return mask;
    };

    StridedSliceMasks updated_masks;
    updated_masks.begin = change_mask_order(test_arguments.masks.begin);
    updated_masks.end = change_mask_order(test_arguments.masks.end);
    updated_masks.new_axis = test_arguments.masks.new_axis; //change_mask_order(
    updated_masks.shrink_axis = change_mask_order(test_arguments.masks.shrink_axis);
    updated_masks.ellipsis = change_mask_order(test_arguments.masks.ellipsis);

    const auto& ref_transpose_order = test_arguments.reference_transpose_order;
    auto new_transpose = [ref_transpose_order](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        auto order = make_shared<Constant>(element::i32,
                                           Shape{ref_transpose_order.size()},
                                           ref_transpose_order);
        new_out_vec[0] = make_shared<Transpose>(out_vec[0], order);
        return new_out_vec;
    };

    auto update_gather_inputs = [](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec = out_vec;
        auto axis = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);
        for (unsigned long idx : idxs) {
            std::vector<int64_t> order(out_vec[idx].get_partial_shape()[0].get_length());
            std::iota(order.rbegin(), order.rend(), 0);
            auto indices = std::make_shared<ov::op::v0::Constant>(element::i32,
                                                                  Shape{order.size()},
                                                                  order);
            new_out_vec[idx] = std::make_shared<ov::op::v8::Gather>(out_vec[idx], indices, axis);
        }
        return new_out_vec;
    };

    test_case.model_ref.preprocess_inputs_to_main = {{update_gather_inputs}, {{1, 2, 3}}};
    auto strided_slice_factory_ref = CREATE_STRIDED_SLICE_FACTORY(StridedSlice);
    strided_slice_factory_ref->set_masks(updated_masks);
    test_case.model_ref.main_op = {strided_slice_factory_ref};
    test_case.model_ref.preprocess_outputs_of_main = {{new_transpose}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

auto fw_test_1 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 10}), // data
            constant<int>(i32, {4}, {1, 2, 2, 1}), // begin
            constant<int>(i32, {4}, {5, 4, 4, 4}), // end
            constant<int>(i32, {4}, {1, 2, 1, 2})  // stride
    };
    // empty masks
    args.masks.begin = {0, 0, 0, 0};
    args.masks.end = {0, 0, 0, 0};
    args.masks.new_axis = {0, 0, 0, 0};
    args.masks.shrink_axis = {0, 0, 0, 0};
    args.masks.ellipsis = {0, 0, 0, 0};

    // reference:
    args.reference_transpose_order = {3, 2, 1, 0};
    return args;
};

auto fw_test_2 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 10}), // data
            constant<int>(i32, {4}, {1, 2, 2, 1}), // begin
            constant<int>(i32, {4}, {5, 4, 4, 4}), // end
            constant<int>(i32, {4}, {1, 2, 1, 2})  // stride
    };
    // begin and end masks
    args.masks.begin = {1, 0, 0, 1};
    args.masks.end = {0, 1, 0, 1};
    args.masks.new_axis = {0, 0, 0, 0};
    args.masks.shrink_axis = {0, 0, 0, 0};
    args.masks.ellipsis = {0, 0, 0, 0};

    // reference:
    args.reference_transpose_order = {3, 2, 1, 0};
    return args;
};

auto fw_test_3 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 10}), // data
            constant<int>(i32, {8}, {1, 2, 2, 1, 0, 0, 0, 0}), // begin
            constant<int>(i32, {8}, {5, 4, 4, 4, 5, 10, 10, 10}), // end
            constant<int>(i32, {8}, {1, 2, 1, 2, 1, 1, 1, 1})  // stride
    };
    // new axis mask
    args.masks.begin = {0, 0, 0, 0};
    args.masks.end = {0, 0, 0, 0};
    args.masks.new_axis = {1, 1, 1, 1};
    args.masks.shrink_axis = {0, 0, 0, 0};
    args.masks.ellipsis = {0, 0, 0, 0};

    // reference:
    args.reference_transpose_order = {0, 1, 2, 3, 7, 6, 5, 4};
    return args;
};

auto fw_test_4 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 10}), // data
            constant<int>(i32, {4}, {1, 2, 2, 1}), // begin
            constant<int>(i32, {4}, {5, 4, 4, 4}), // end
            constant<int>(i32, {4}, {1, 2, 1, 2})  // stride
    };
    // shrink mask
    args.masks.begin = {0, 0, 0, 0};
    args.masks.end = {0, 0, 0, 0};
    args.masks.new_axis = {0, 0, 0, 0};
    args.masks.shrink_axis = {0, 1, 0, 1};
    args.masks.ellipsis = {0, 0, 0, 0};

    // reference:
    args.reference_transpose_order = {1, 0};
    return args;
};

auto fw_test_5 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 10}), // data
            constant<int>(i32, {4}, {1, 2, 2, 1}), // begin
            constant<int>(i32, {4}, {5, 4, 4, 4}), // end
            constant<int>(i32, {4}, {1, 2, 1, 2})  // stride
    };
    // ellipsis mask
    args.masks.begin = {0, 0, 0, 0};
    args.masks.end = {0, 0, 0, 0};
    args.masks.new_axis = {0, 0, 0, 0};
    args.masks.shrink_axis = {0, 0, 0, 0};
    args.masks.ellipsis = {0, 0, 1, 0};
    return args;
};

auto fw_test_6 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 10}), // data
            constant<int>(i32, {4}, {1, 2, 2, 1}), // begin
            constant<int>(i32, {4}, {5, 4, 4, 4}), // end
            constant<int>(i32, {4}, {1, 2, 1, 2})  // stride
    };
    // mixed masks: begin, end, new_axis
    args.masks.begin = {0, 1, 0, 0};
    args.masks.end = {0, 0, 0, 1};
    args.masks.new_axis = {0, 1, 0, 1};
    args.masks.shrink_axis = {0, 0, 0, 0};
    args.masks.ellipsis = {0, 0, 0, 0};
    return args;
};

auto fw_test_7 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 10}), // data
            constant<int>(i32, {4}, {1, 2, 2, 1}), // begin
            constant<int>(i32, {4}, {5, 4, 4, 4}), // end
            constant<int>(i32, {4}, {1, 2, 1, 2})  // stride
    };
    // mixed masks: begin, end, shrink
    args.masks.begin = {0, 1, 0, 0};
    args.masks.end = {1, 0, 0, 1};
    args.masks.new_axis = {0, 0, 0, 0};
    args.masks.shrink_axis = {0, 1, 0, 1};
    args.masks.ellipsis = {0, 0, 0, 0};
    return args;
};

auto fw_test_8 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 10}), // data
            constant<int>(i32, {4}, {1, 2, 2, 1}), // begin
            constant<int>(i32, {4}, {5, 4, 4, 4}), // end
            constant<int>(i32, {4}, {1, 2, 1, 2})  // stride
    };
    // mixed masks: shrink and new_axis
    args.masks.begin = {0, 0, 0, 0};
    args.masks.end = {0, 0, 0, 0};
    args.masks.new_axis = {1, 1, 0, 1};
    args.masks.shrink_axis = {0, 1, 0, 1};
    args.masks.ellipsis = {0, 0, 0, 0};
    return args;
};

auto fw_test_9 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 10}), // data
            constant<int>(i32, {4}, {1, 2, 2, 1}), // begin
            constant<int>(i32, {4}, {5, 4, 4, 4}), // end
            constant<int>(i32, {4}, {1, 2, 1, 2})  // stride
    };
    // mixed masks: shrink, new_axis, ellipsis
    args.masks.begin = {0, 0, 0, 0};
    args.masks.end = {0, 0, 0, 0};
    args.masks.new_axis = {1, 0, 0, 1};
    args.masks.shrink_axis = {0, 1, 0, 1};
    args.masks.ellipsis = {0, 0, 1, 0};
    return args;
};

auto fw_test_10 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 10}), // data
            constant<int>(i32, {4}, {1, 2, 2, 1}), // begin
            constant<int>(i32, {4}, {5, 4, 4, 4}), // end
            constant<int>(i32, {4}, {1, 2, 1, 2})  // stride
    };
    // mixed masks: shrink, new_axis, begin, end
    args.masks.begin = {0, 1, 0, 0};
    args.masks.end = {1, 0, 0, 1};
    args.masks.new_axis = {1, 0, 0, 1};
    args.masks.shrink_axis = {0, 1, 0, 1};
    args.masks.ellipsis = {0, 0, 0, 0};
    return args;
};

INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_1, TSTestFixture, test_forward_strided_slice(fw_test_1()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_2, TSTestFixture, test_forward_strided_slice(fw_test_2()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_3, TSTestFixture, test_forward_strided_slice(fw_test_3()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_4, TSTestFixture, test_forward_strided_slice(fw_test_4()));
/*INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_5, TSTestFixture, test_forward_strided_slice(fw_test_5()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_6, TSTestFixture, test_forward_strided_slice(fw_test_6()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_7, TSTestFixture, test_forward_strided_slice(fw_test_7()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_8, TSTestFixture, test_forward_strided_slice(fw_test_8()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_9, TSTestFixture, test_forward_strided_slice(fw_test_9()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_10, TSTestFixture, test_forward_strided_slice(fw_test_10()));*/

}  // namespace gather
}  // namespace testing
}  // namespace transpose_sinking
