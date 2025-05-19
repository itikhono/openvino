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

constexpr size_t batch = 1;
constexpr size_t seq_len = 128;
constexpr size_t num_heads = 1;
constexpr size_t head_size = 64;
constexpr size_t hidden_size = num_heads * head_size;

std::shared_ptr<ov::Node> build_l2_norm(const std::shared_ptr<ov::Node>& input) {
    using namespace ov::opset10;
    auto pow = std::make_shared<Power>(input, Constant::create(element::f32, Shape{}, {2.0}));
    auto var = std::make_shared<ReduceMean>(pow, Constant::create(element::i64, Shape{1}, {2}), true);
    auto sqrt = std::make_shared<Sqrt>(var);
    auto div = std::make_shared<Divide>(input, sqrt);
    auto scale = std::make_shared<Multiply>(div, Constant::create(element::f32, Shape{batch, 1, head_size}, {1.0f}));
    auto shift = std::make_shared<Add>(scale, Constant::create(element::f32, Shape{batch, 1, head_size}, {0.0f}));
    return shift;
}

std::shared_ptr<ov::Node> build_qkv_projection(const std::shared_ptr<ov::Node>& norm_out) {
    using namespace ov::opset10;
    auto weights = Constant::create(element::f32, Shape{head_size, hidden_size}, {0.1f});
    auto zp = Constant::create(element::f32, Shape{}, {0.0f});
    auto scale = Constant::create(element::f32, Shape{}, {0.01f});

    auto weights_f32 = std::make_shared<Convert>(weights, element::f32);
    auto zp_f32 = std::make_shared<Convert>(zp, element::f32);
    auto scale_f32 = std::make_shared<Convert>(scale, element::f32);

    auto weights_sub = std::make_shared<Subtract>(weights_f32, zp_f32);
    auto dq_weights = std::make_shared<Multiply>(weights_sub, scale_f32);

    auto matmul = std::make_shared<MatMul>(norm_out, dq_weights);
    auto bias = std::make_shared<Add>(matmul, Constant::create(element::f32, Shape{batch, 1, hidden_size}, {0.01f}));

    return bias;
}

std::shared_ptr<ov::Node> build_sdpa_preprocessing(const std::shared_ptr<ov::Node>& proj_bias) {
    using namespace ov::opset10;
    auto reshape = std::make_shared<Reshape>(proj_bias, Constant::create(element::i64, Shape{4}, {batch, size_t(-1), seq_len, head_size}), false);
    //auto transpose = std::make_shared<Transpose>(reshape, Constant::create(element::i64, Shape{4}, {0, 1, 3, 2}));
    return reshape;
}

std::shared_ptr<ov::Node> build_ROPE(const std::shared_ptr<ov::Node>& proj_bias) {
    using namespace ov::opset10;

    // Step 1: Reshape to [B, S, H, D]
    auto reshape = std::make_shared<Reshape>(
        proj_bias,
        Constant::create(element::i64, Shape{4}, {batch, seq_len, size_t(-1), head_size}),
        false);

    // Step 2: Transpose to [B, H, S, D]
    auto transpose = std::make_shared<Transpose>(
        reshape,
        Constant::create(element::i64, Shape{4}, {0, 2, 1, 3}));

    // Step 3: Split S dimension (axis=2) into two parts [B, H, half, D], [B, H, half, D]
    size_t half = seq_len / 2;
    auto axis = Constant::create(element::i64, Shape{}, {2});
    auto split_lengths = Constant::create(element::i64, Shape{2}, {half, half});
    auto split = std::make_shared<VariadicSplit>(transpose, axis, split_lengths);

    // Step 4: Dummy "rotation" step
    auto mul_1 = std::make_shared<Multiply>(
        split->output(0),
        Constant::create(element::f32, Shape{batch, num_heads, half, head_size}, {1.0f}));

    // Step 5: Concat [B, H, S, D] along axis=2
    auto concat = std::make_shared<Concat>(OutputVector{mul_1, split->output(1)}, 2);

    // Step 6: Second dummy rotation multiply (keep compatible shapes)
    auto mul_2 = std::make_shared<Multiply>(
        concat,
        Constant::create(element::f32, Shape{batch, num_heads, seq_len, head_size}, {1.0f}));

    // Step 7: Match shape of original tensor
    auto back_mul = std::make_shared<Multiply>(
        reshape,
        Constant::create(element::f32, Shape{batch, seq_len, num_heads, head_size}, {1.0f}));

            // Step 8: Transpose to [B, H, S, D]
    auto transpose_2 = std::make_shared<Transpose>(
        back_mul,
        Constant::create(element::i64, Shape{4}, {0, 2, 1, 3}));
    // Step 9: Final Add → shape: [B, H, S, D]
    auto rotated = std::make_shared<Add>(transpose_2, mul_2);

    return rotated;
}


std::shared_ptr<ov::Node> build_sdpa(const std::shared_ptr<ov::Node>& q,
                                     const std::shared_ptr<ov::Node>& k,
                                     const std::shared_ptr<ov::Node>& v) {
    using namespace ov::opset10;

 
    auto kT = std::make_shared<Transpose>(k, Constant::create(element::i64, Shape{4}, {0, 1, 3, 2}));
    auto scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    auto scaled_k = std::make_shared<Multiply>(kT, Constant::create(element::f32, Shape{1}, {scale}));

    auto qk = std::make_shared<MatMul>(q, scaled_k);  
    std::cout << "qk shape: " << qk->get_shape() << std::endl;             
    auto bias = Constant::create(element::f32, Shape{1, 1, 1, seq_len}, {0.0f});
    auto add = std::make_shared<Add>(qk, bias);
    std::cout << "add shape: " << add->get_shape() << std::endl;
    auto softmax = std::make_shared<Softmax>(add, -1);
    std::cout << "softmax shape: " << softmax->get_shape() << std::endl;             
    auto attn = std::make_shared<MatMul>(softmax, v);    
    std::cout << "v shape: " << v->get_shape() << std::endl;      
    std::cout << "attn shape: " << attn->get_shape() << std::endl;
    return attn;
}


std::shared_ptr<ov::Node> build_post_sdpa(const std::shared_ptr<ov::Node>& attn_out) {
    using namespace ov::opset10;

    // Input: [B, H, S, D]
    auto transpose = std::make_shared<Transpose>(
        attn_out,
        Constant::create(element::i64, Shape{4}, {0, 2, 1, 3}));  // [B, S, H, D]

    auto reshape = std::make_shared<Reshape>(
        transpose,
        Constant::create(element::i64, Shape{3}, {batch, seq_len, hidden_size}),
        false);  // [B, S, H * D]

    auto weights = Constant::create(element::f32, Shape{hidden_size, hidden_size}, {1.0f});
    auto proj = std::make_shared<MatMul>(reshape, weights);       // [B, S, H * D]

    return proj;
}


std::shared_ptr<ov::Model> build_model_gqa_pack_mha(size_t num_heads, size_t num_groups) {
    using namespace ov::opset10;

    OPENVINO_ASSERT(num_heads % num_groups == 0, "num_heads must be divisible by num_groups");

    const size_t heads_per_group = num_heads / num_groups;

    auto input = std::make_shared<Parameter>(element::f32, Shape{1, 128, 64});
    auto norm = build_l2_norm(input);  // L2 normalization

    std::vector<std::shared_ptr<Node>> all_head_outputs;

    for (size_t g = 0; g < num_groups; ++g) {
        // Shared K/V for this group
        auto k_proj = build_qkv_projection(norm);
        auto v_proj = build_qkv_projection(norm);

        auto k = build_ROPE(k_proj);
        auto v = build_sdpa_preprocessing(v_proj);

        for (size_t h = 0; h < heads_per_group; ++h) {
            auto q_proj = build_qkv_projection(norm);
            auto q = build_ROPE(q_proj);

            auto attn_out = build_sdpa(q, k, v);
            auto projected = build_post_sdpa(attn_out);

            all_head_outputs.push_back(projected);
        }
    }

    // Sum all head outputs using chained Add
    std::shared_ptr<Node> combined = all_head_outputs.front();
    for (size_t i = 1; i < all_head_outputs.size(); ++i) {
        combined = std::make_shared<Add>(combined, all_head_outputs[i]);
    }

    // Residual connection
    auto residual = std::make_shared<Add>(combined, input);

    return std::make_shared<ov::Model>(NodeVector{residual}, ParameterVector{input});
}


TEST(PackMHATests, MatchesFullCombinedBlock) {
    auto model = build_model_gqa_pack_mha(6, 3);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Serialize>(std::string("/workspace/buffer/test_model_gqa_before.xml"), "/workspace/buffer/test_model_gqa_before.bin");
    manager.register_pass<ov::pass::PackMHA>();
    manager.register_pass<ov::pass::Serialize>(std::string("/workspace/buffer/test_model_gqa_after.xml"), "/workspace/buffer/test_model_gqa_after.bin");
    
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();
}

TEST(PackMHATests, RealModel) {
    ov::Core core;
    auto model = core.read_model("/workspace/buffer/mha_pattern/model.xml");

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PackMHA>();
    manager.register_pass<ov::pass::Serialize>(std::string("/workspace/buffer/pack_mha_new.xml"), "/workspace/buffer/pack_mha_new.bin");
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();
}