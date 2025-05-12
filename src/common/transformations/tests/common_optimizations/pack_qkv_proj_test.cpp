// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/pack_mha.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/pass/serialize.hpp"

using namespace ov;
using namespace ov::opset8;

namespace {

std::shared_ptr<Node> build_l2_norm(const std::shared_ptr<Node>& input) {
    auto pow = std::make_shared<Power>(input, Constant::create(input->get_element_type(), Shape{1}, {2}));
    auto reduce = std::make_shared<ReduceSum>(pow, Constant::create(element::i64, Shape{1}, {1}), true);
    auto sqrt = std::make_shared<Sqrt>(reduce);
    auto div = std::make_shared<Divide>(input, sqrt);
    auto scale = Constant::create(input->get_element_type(), Shape{1}, {1.0f});
    return std::make_shared<Multiply>(div, scale);
}

ov::Output<ov::Node> create_quant_weight(float val, const std::string& name) {
    using namespace ov;
    using namespace ov::opset8;

    auto w_i8 = Constant::create(element::i8, Shape{768, 768}, {static_cast<int8_t>(val)});
    auto zero = Constant::create(element::i8, Shape{1}, {10});
    auto scale = Constant::create(element::f32, Shape{1}, {0.1f});

    auto w_fp32 = std::make_shared<Convert>(w_i8, element::f32);
    auto zp_fp32 = std::make_shared<Convert>(zero, element::f32);
    auto sub = std::make_shared<Subtract>(w_fp32, zp_fp32);
    auto mul = std::make_shared<Multiply>(sub, scale);

    mul->set_friendly_name(name);
    return mul;
}

}  // namespace

TEST_F(TransformationTestsF, FuseThreeMatMulsWithSharedL2Input) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 768});
        auto norm = build_l2_norm(input);

        auto w1 = Constant::create(element::f32, Shape{768, 768}, {0.1f});
        auto w2 = Constant::create(element::f32, Shape{768, 768}, {0.2f});
        auto w3 = Constant::create(element::f32, Shape{768, 768}, {0.3f});

        auto mm1 = std::make_shared<MatMul>(norm, w1); mm1->set_friendly_name("q_proj.0");
        auto mm2 = std::make_shared<MatMul>(norm, w2); mm2->set_friendly_name("q_proj.1");
        auto mm3 = std::make_shared<MatMul>(norm, w3); mm3->set_friendly_name("q_proj.2");

        auto relu1 = std::make_shared<Relu>(mm1);
        auto relu2 = std::make_shared<Relu>(mm2);
        auto relu3 = std::make_shared<Relu>(mm3);

        auto concat = std::make_shared<Concat>(OutputVector {relu1, relu2, relu3}, 1);
        model = std::make_shared<Model>(OutputVector{concat}, ParameterVector{input});
        manager.register_pass<ov::pass::Serialize>(std::string("cbefore.xml"), "cbefore.bin");
        manager.register_pass<ov::pass::PackQKVProj>();
        manager.register_pass<ov::pass::Serialize>(std::string("cafter.xml"), "cafter.bin");
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 768});
        auto norm = build_l2_norm(input);

        auto w1 = Constant::create(element::f32, Shape{768, 768}, {0.1f});
        auto w2 = Constant::create(element::f32, Shape{768, 768}, {0.2f});
        auto w3 = Constant::create(element::f32, Shape{768, 768}, {0.3f});

        auto packed_weights = ov::op::util::make_try_fold<Concat>(OutputVector{w1, w2, w3}, 1);
        auto fused_mm = std::make_shared<MatMul>(norm, packed_weights);
        fused_mm->set_friendly_name("q_proj_fused_mm");

        auto relu = std::make_shared<Relu>(fused_mm);
        auto concat = std::make_shared<Concat>(OutputVector {relu}, 1);
        model_ref = std::make_shared<Model>(OutputVector{concat}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, FuseMatMulsWithSharedL2InputAndQuantWeights) {
    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 768});
        auto norm = build_l2_norm(input);

        auto mm1 = std::make_shared<MatMul>(norm, create_quant_weight(1, "q_proj.0")); mm1->set_friendly_name("q_proj.0");
        auto mm2 = std::make_shared<MatMul>(norm, create_quant_weight(2, "q_proj.1")); mm2->set_friendly_name("q_proj.1");
        auto mm3 = std::make_shared<MatMul>(norm, create_quant_weight(3, "q_proj.2")); mm3->set_friendly_name("q_proj.2");

        auto relu1 = std::make_shared<Relu>(mm1);
        auto relu2 = std::make_shared<Relu>(mm2);
        auto relu3 = std::make_shared<Relu>(mm3);

        auto concat = std::make_shared<Concat>(OutputVector {relu1, relu2, relu3}, 1);
        model = std::make_shared<Model>(OutputVector{concat}, ParameterVector{input});
        manager.register_pass<ov::pass::Serialize>(std::string("before.xml"), "before.bin");
        manager.register_pass<ov::pass::PackQKVProj>();
        manager.register_pass<ov::pass::Serialize>(std::string("after.xml"), "after.bin");
    }

    {
        auto input = std::make_shared<Parameter>(element::f32, Shape{1, 768});
        auto norm = build_l2_norm(input);

        auto w1 = create_quant_weight(1,"q_proj.0");
        auto w2 = create_quant_weight(2,"q_proj.1");
        auto w3 = create_quant_weight(3,"q_proj.2");

        auto packed = ov::op::util::make_try_fold<Concat>(OutputVector{w1, w2, w3}, 1);
        auto fused_mm = std::make_shared<MatMul>(norm, packed);
        fused_mm->set_friendly_name("q_proj_fused_mm");

        auto relu = std::make_shared<Relu>(fused_mm);
        auto concat = std::make_shared<Concat>(OutputVector {relu}, 1);
        model_ref = std::make_shared<Model>(OutputVector{concat}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
