// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/pack_mha.hpp"

#include <gtest/gtest.h>

#include "openvino/opsets/opset10.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include <memory>
#include <openvino/runtime/core.hpp>
#include <transformations/utils/print_model.hpp>
#include <openvino/pass/serialize.hpp>

using namespace ov;
using namespace ov::opset10;

// === Model 1: L2Norm pattern only ===
std::shared_ptr<ov::Model> build_model_norm_block_only() {
    auto input = std::make_shared<Parameter>(element::f32, Shape{1, 128, 64});
    auto pow = std::make_shared<Power>(input, Constant::create(element::f32, Shape{}, {2.0}));
    auto var = std::make_shared<ReduceMean>(pow, Constant::create(element::i64, Shape{1}, {2}), true);
    auto sqrt = std::make_shared<Sqrt>(var);
    auto div = std::make_shared<Divide>(input, sqrt);
    auto scale = std::make_shared<Multiply>(div, Constant::create(element::f32, Shape{1, 1, 64}, {1.0}));
    auto shift = std::make_shared<Add>(scale, Constant::create(element::f32, Shape{1, 1, 64}, {0.0}));
    return std::make_shared<ov::Model>(NodeVector{shift}, ParameterVector{input});
}

// === Model 2: QKV Projection + Preprocessing pattern only ===
std::shared_ptr<ov::Model> build_model_proj_pre_sdpa_only() {
    using namespace ov::opset10;

    auto input = std::make_shared<Parameter>(element::f32, Shape{1, 128, 64});
    auto constant = Constant::create(element::f32, Shape{64, 64}, {0.1f});
    auto convert = std::make_shared<Convert>(constant, element::f32);
    auto mm = std::make_shared<MatMul>(input, convert);
    auto bias = std::make_shared<Add>(mm, Constant::create(element::f32, Shape{1, 1, 64}, {0.01f}));

    auto reshape = std::make_shared<Reshape>(bias, Constant::create(element::i64, Shape{3}, {1, 128, 64}), false);
    auto trans = std::make_shared<Transpose>(reshape, Constant::create(element::i64, Shape{3}, {0, 2, 1}));
    auto split = std::make_shared<VariadicSplit>(
            trans,
            Constant::create(element::i64, Shape{}, {1}),
            Constant::create(element::i64, Shape{2}, {32, 32})
    );

    auto scale = Constant::create(element::f32, Shape{1, 32}, {1.0f});
    auto reshaped_scale = std::make_shared<Reshape>(scale, Constant::create(element::i64, Shape{3}, {1, 32, 1}), false);
    auto mul = std::make_shared<Multiply>(split->output(0), reshaped_scale);
    auto concat = std::make_shared<Concat>(OutputVector{mul, split->output(1)}, 1);
    auto mul2 = std::make_shared<Multiply>(concat, Constant::create(element::f32, Shape{1, 64, 1}, {1.0}));

    // ✅ Add second output branch to match block output
    auto mul3 = std::make_shared<Multiply>(reshape, Constant::create(element::f32, Shape{1, 128, 64}, {1.0}));
    auto trans2 = std::make_shared<Transpose>(mul3, Constant::create(element::i64, Shape{3}, {0, 2, 1}));
    auto add = std::make_shared<Add>(trans2, Constant::create(element::f32, Shape{1, 64, 128}, {0.0}));

    return std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{input});
}

// === Model 3: SDPA + post projection only ===
std::shared_ptr<ov::Model> build_model_sdpa_post_only() {
    using namespace ov::opset10;

    auto q = std::make_shared<Parameter>(element::f32, Shape{1, 64, 128});
    auto k = std::make_shared<Parameter>(element::f32, Shape{1, 64, 128});
    auto v = std::make_shared<Parameter>(element::f32, Shape{1, 128, 64});

    auto kT = std::make_shared<Transpose>(k, Constant::create(element::i64, Shape{3}, {0, 2, 1}));
    auto scaled_k = std::make_shared<Multiply>(kT, Constant::create(element::f32, Shape{1, 128, 1}, {0.125f}));
    auto qk = std::make_shared<MatMul>(q, scaled_k);

    auto bias = std::make_shared<Add>(qk, Constant::create(element::f32, Shape{1, 1, 64}, {0.0f}));
    auto softmax = std::make_shared<Softmax>(bias, 2);

    // ✅ Match the pattern structure: Reshape → Transpose
    auto reshape_v = std::make_shared<Reshape>(
            v, Constant::create(element::i64, Shape{3}, {1, 128, 64}), false);
    auto vT = std::make_shared<Transpose>(reshape_v, Constant::create(element::i64, Shape{3}, {0, 2, 1}));

    auto attn_out = std::make_shared<MatMul>(softmax, vT);

    auto t2 = std::make_shared<Transpose>(attn_out, Constant::create(element::i64, Shape{3}, {0, 2, 1}));
    auto reshaped = std::make_shared<Reshape>(t2, Constant::create(element::i64, Shape{3}, {1, 128, 64}), false);
    auto proj = std::make_shared<MatMul>(reshaped, Constant::create(element::f32, Shape{64, 64}, {1.0}));
    auto out = std::make_shared<Add>(proj, Constant::create(element::f32, Shape{1, 1, 64}, {0.0}));

    return std::make_shared<ov::Model>(NodeVector{out}, ParameterVector{q, k, v});
}

std::shared_ptr<ov::Model> build_model_full_pack_mha() {
    using namespace ov::opset10;

    // === L2Norm block ===
    auto input = std::make_shared<Parameter>(element::f32, Shape{1, 128, 64});
    auto pow = std::make_shared<Power>(input, Constant::create(element::f32, Shape{}, {2.0}));
    auto var = std::make_shared<ReduceMean>(pow, Constant::create(element::i64, Shape{1}, {2}), true);
    auto sqrt = std::make_shared<Sqrt>(var);
    auto div = std::make_shared<Divide>(input, sqrt);
    auto norm_scale = std::make_shared<Multiply>(div, Constant::create(element::f32, Shape{1, 1, 64}, {1.0}));
    auto norm_out = std::make_shared<Add>(norm_scale, Constant::create(element::f32, Shape{1, 1, 64}, {0.0}));

    // === QKV projection + preprocessing ===
    auto constant = Constant::create(element::f32, Shape{64, 64}, {0.1f});
    auto convert = std::make_shared<Convert>(constant, element::f32);
    auto mm = std::make_shared<MatMul>(norm_out, convert);
    auto bias = std::make_shared<Add>(mm, Constant::create(element::f32, Shape{1, 1, 64}, {0.01f}));

    auto reshape = std::make_shared<Reshape>(bias, Constant::create(element::i64, Shape{3}, {1, 128, 64}), false);
    auto trans = std::make_shared<Transpose>(reshape, Constant::create(element::i64, Shape{3}, {0, 2, 1}));
    auto split = std::make_shared<VariadicSplit>(
            trans,
            Constant::create(element::i64, Shape{}, {1}),
            Constant::create(element::i64, Shape{2}, {32, 32})
    );

    auto scale = Constant::create(element::f32, Shape{1, 32}, {1.0f});
    auto reshaped_scale = std::make_shared<Reshape>(scale, Constant::create(element::i64, Shape{3}, {1, 32, 1}), false);
    auto mul = std::make_shared<Multiply>(split->output(0), reshaped_scale);
    auto concat = std::make_shared<Concat>(OutputVector{mul, split->output(1)}, 1);
    auto mul2 = std::make_shared<Multiply>(concat, Constant::create(element::f32, Shape{1, 64, 1}, {1.0}));

    auto mul3 = std::make_shared<Multiply>(reshape, Constant::create(element::f32, Shape{1, 128, 64}, {1.0}));
    auto trans2 = std::make_shared<Transpose>(mul3, Constant::create(element::i64, Shape{3}, {0, 2, 1}));
    auto add = std::make_shared<Add>(trans2, Constant::create(element::f32, Shape{1, 64, 128}, {0.0}));

    // === SDPA + Post projection ===
    auto kT = std::make_shared<Transpose>(add, Constant::create(element::i64, Shape{3}, {0, 2, 1}));
    auto scaled_k = std::make_shared<Multiply>(kT, Constant::create(element::f32, Shape{1, 128, 1}, {0.125f}));
    auto qk = std::make_shared<MatMul>(add, scaled_k);

    auto bias_sdpa = std::make_shared<Add>(qk, Constant::create(element::f32, Shape{1, 1, 64}, {0.0f}));
    auto softmax = std::make_shared<Softmax>(bias_sdpa, 2);

    auto reshape_v = std::make_shared<Reshape>(add, Constant::create(element::i64, Shape{3}, {1, 128, 64}), false);
    auto vT = std::make_shared<Transpose>(reshape_v, Constant::create(element::i64, Shape{3}, {0, 2, 1}));

    auto attn_out = std::make_shared<MatMul>(softmax, vT);

    auto t2 = std::make_shared<Transpose>(attn_out, Constant::create(element::i64, Shape{3}, {0, 2, 1}));
    auto reshaped = std::make_shared<Reshape>(t2, Constant::create(element::i64, Shape{3}, {1, 128, 64}), false);
    auto proj = std::make_shared<MatMul>(reshaped, Constant::create(element::f32, Shape{64, 64}, {1.0}));
    auto out = std::make_shared<Add>(proj, Constant::create(element::f32, Shape{1, 1, 64}, {0.0}));

    return std::make_shared<ov::Model>(NodeVector{out}, ParameterVector{input});
}


// === Unit Tests ===

TEST(PackMHATests, MatchesNormBlock) {
    auto model = build_model_norm_block_only();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PackMHA>();
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();
}

TEST(PackMHATests, MatchesProjPreSDPA) {
    auto model = build_model_proj_pre_sdpa_only();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PackMHA>();
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();
}

TEST(PackMHATests, MatchesSDPAPostBlock) {
    auto model = build_model_sdpa_post_only();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PackMHA>();
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();
}

TEST(PackMHATests, MatchesFullCombinedBlock) {
    auto model = build_model_full_pack_mha();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PackMHA>();
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();
}

TEST(PackMHATests, RealModel) {
    ov::Core core;
    auto model = core.read_model("/workspace/buffer/mha_pattern/model.xml");

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PackMHA>();
    manager.register_pass<ov::pass::Serialize>(std::string("/workspace/buffer/pack_mha.xml"), "/workspace/buffer/pack_mha.bin");
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();
}