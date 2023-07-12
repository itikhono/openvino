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
    updated_masks.new_axis = change_mask_order(test_arguments.masks.new_axis);
    updated_masks.shrink_axis = change_mask_order(test_arguments.masks.shrink_axis);
    updated_masks.ellipsis = change_mask_order(test_arguments.masks.ellipsis);

    test_case.model_ref.preprocess_inputs_to_main = {{set_gather_for}, {{1, 2, 3}}};
    auto strided_slice_factory_ref = CREATE_STRIDED_SLICE_FACTORY(StridedSlice);
    strided_slice_factory_ref->set_masks(updated_masks);
    test_case.model_ref.main_op = {strided_slice_factory_ref};
    test_case.model_ref.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

auto fw_test_1 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 6}), // data
            constant<int>(i32, {4}, {0, 0, 0, 0}), // begin
            constant<int>(i32, {4}, {2, 3, 2, 4}), // end
            constant<int>(i32, {4}, {1, 2, 1, 2})  // stride
    };
    args.masks.begin = {0, 0, 0, 0};
    args.masks.end = {0, 0, 0, 0};
    args.masks.new_axis = {0, 0, 0, 0};
    args.masks.shrink_axis = {0, 0, 0, 0};
    args.masks.ellipsis = {0, 0, 0, 0};
    return args;
};

auto fw_test_2 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 6}), // data
            constant<int>(i32, {4}, {0, 0, 0, 0}), // begin
            constant<int>(i32, {4}, {2, 3, 2, 4}), // end
            constant<int>(i32, {4}, {1, 2, 1, 2})  // stride
    };
    args.masks.begin = {1, 0, 1, 0};
    args.masks.end = {0, 1, 0, 1};
    args.masks.new_axis = {0, 0, 0, 0};
    args.masks.shrink_axis = {0, 0, 0, 0};
    args.masks.ellipsis = {0, 0, 0, 0};
    return args;
};

auto fw_test_3 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 6}), // data
            constant<int>(i32, {4}, {0, 0, 0, 0}), // begin
            constant<int>(i32, {4}, {2, 3, 2, 4}), // end
            constant<int>(i32, {4}, {1, 2, 1, 2})  // stride
    };
    args.masks.begin = {0, 0, 0, 0};
    args.masks.end = {0, 0, 0, 0};
    args.masks.new_axis = {0, 1, 0, 0};
    args.masks.shrink_axis = {0, 0, 0, 0};
    args.masks.ellipsis = {0, 0, 0, 0};
    return args;
};

auto fw_test_4 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 6}), // data
            constant<int>(i32, {4}, {0, 0, 0, 0}), // begin
            constant<int>(i32, {4}, {2, 3, 2, 4}), // end
            constant<int>(i32, {4}, {1, 2, 1, 2})  // stride
    };
    args.masks.begin = {0, 0, 0, 0};
    args.masks.end = {0, 0, 0, 0};
    args.masks.new_axis = {0, 1, 0, 0};
    args.masks.shrink_axis = {0, 0, 1, 0};
    args.masks.ellipsis = {0, 0, 0, 0};
    return args;
};

INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_1, TSTestFixture, test_forward_strided_slice(fw_test_1()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_2, TSTestFixture, test_forward_strided_slice(fw_test_2()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_3, TSTestFixture, test_forward_strided_slice(fw_test_3()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_4, TSTestFixture, test_forward_strided_slice(fw_test_4()));

}  // namespace gather
}  // namespace testing
}  // namespace transpose_sinking
