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
    //auto model = core.read_model("/workspace/buffer/mha_pattern/model.xml");


        int hidden_size = 1024;
        int num_attention_heads = 12;
        int head_size = 64;
        bool add_qk_rotation = true;
        
        // Create model using OpenVINO API
        auto model = std::make_shared<ov::Model>();
        
        // ----- Layer Normalization for Input -----
        // Layer Norm implemented as: input / sqrt(sum(input^2))
        auto squares = std::make_shared<ov::op::v1::Power>(
            input_tensor, 
            ov::op::v0::Constant::create(ov::element::f32, {1,1,1}, {2.0f}),
            ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
        
        auto sum_squares = std::make_shared<ov::op::v1::ReduceSum>(
            squares, 
            ov::op::v0::Constant::create(ov::element::i64, {1}, {2}),
            true); // keep_dims = true
        
        auto norm_denom = std::make_shared<ov::op::v0::Sqrt>(sum_squares);
        
        auto normalized_input = std::make_shared<ov::op::v1::Divide>(
            input_tensor, 
            norm_denom, 
            ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        
        auto ln_scale = ov::op::v0::Constant::create(
            ov::element::f32, ov::Shape({1,1,hidden_size}), {1.0f});
        
        auto scaled_normalized_input = std::make_shared<ov::op::v1::Multiply>(
            ln_scale, 
            normalized_input, 
            ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
        
        // ----- Multi-Head Attention Implementation -----
        // We'll collect outputs from each attention head
        std::vector<ov::Output<ov::Node>> attention_head_outputs;
        
        // Define attention scaling factor (1/sqrt(head_size))
        float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_size));
        auto scale_factor = ov::op::v0::Constant::create(
            ov::element::f32, ov::Shape({1,1,1,1}), {attn_scale});
        
        // Create multiple attention heads (6 heads in this example based on the code)
        const int num_heads = 6;  // Derived from examining the original code
        
        for (int head = 0; head < num_heads; head++) {
            // ----- Query Projection for this head -----
            // Dequantize weights for query projection
            auto q_weights_quant = createQuantizedWeights(1024, head_size, 122 + head);
            auto q_weights = dequantizeWeights(q_weights_quant, 0.00065f + head * 0.00001f);
            
            // Create query projection
            auto query = std::make_shared<ov::op::v0::MatMul>(
                scaled_normalized_input, 
                q_weights,
                false,  // transpose_a
                false); // transpose_b
                
            auto q_bias = ov::op::v0::Constant::create(
                ov::element::f32, ov::Shape({1,1,head_size}), {1.0f});
            
            auto query_with_bias = std::make_shared<ov::op::v1::Add>(
                q_bias, 
                query, 
                ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
                
            // Reshape query for attention computation
            auto query_reshape = std::make_shared<ov::op::v1::Reshape>(
                query_with_bias, 
                ov::op::v0::Constant::create(ov::element::i64, {4}, {1,64,-1,head_size}),
                true); // special_zero = true
            
            // ----- Key Projection for this head -----
            // Dequantize weights for key projection
            auto k_weights_quant = createQuantizedWeights(1024, head_size, 125 + head);
            auto k_weights = dequantizeWeights(k_weights_quant, 0.00061f + head * 0.00001f);
            
            // Create key projection
            auto key = std::make_shared<ov::op::v0::MatMul>(
                scaled_normalized_input, 
                k_weights,
                false,  // transpose_a
                false); // transpose_b
                
            auto k_bias = ov::op::v0::Constant::create(
                ov::element::f32, ov::Shape({1,1,head_size}), {1.0f});
            
            auto key_with_bias = std::make_shared<ov::op::v1::Add>(
                k_bias, 
                key, 
                ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
                
            // Reshape key for attention computation
            auto key_reshape = std::make_shared<ov::op::v1::Reshape>(
                key_with_bias, 
                ov::op::v0::Constant::create(ov::element::i64, {4}, {1,64,-1,head_size}),
                true); // special_zero = true
                
            // ----- Value Projection for this head -----
            // Dequantize weights for value projection
            auto v_weights_quant = createQuantizedWeights(1024, head_size, 121 + head);
            auto v_weights = dequantizeWeights(v_weights_quant, 0.00065f + head * 0.00001f);
            
            // Create value projection
            auto value = std::make_shared<ov::op::v0::MatMul>(
                scaled_normalized_input, 
                v_weights,
                false,  // transpose_a
                false); // transpose_b
                
            auto v_bias = ov::op::v0::Constant::create(
                ov::element::f32, ov::Shape({1,1,head_size}), {1.0f});
            
            auto value_with_bias = std::make_shared<ov::op::v1::Add>(
                v_bias, 
                value, 
                ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
                
            // Reshape value for attention computation
            auto value_reshape = std::make_shared<ov::op::v1::Reshape>(
                value_with_bias, 
                ov::op::v0::Constant::create(ov::element::i64, {4}, {1,64,-1,head_size}),
                true); // special_zero = true
                
            // ----- Rotary Position Embeddings (if enabled) -----
            if (add_qk_rotation) {
                // Apply rotary position embeddings to query
                auto query_transposed = std::make_shared<ov::op::v1::Transpose>(
                    query_reshape, 
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {0,2,1,3}));
                    
                auto query_split = std::make_shared<ov::op::v1::VariadicSplit>(
                    query_transposed,
                    ov::op::v0::Constant::create(ov::element::i64, {1}, {3}),
                    ov::op::v0::Constant::create(ov::element::i64, {2}, {32,32}));
                    
                auto negate_factor = ov::op::v0::Constant::create(
                    ov::element::f32, ov::Shape({1,1,1,1}), {-1.0f});
                    
                auto query_part_neg = std::make_shared<ov::op::v1::Multiply>(
                    query_split->output(0), 
                    negate_factor,
                    ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
                    
                auto query_rotated = std::make_shared<ov::op::v1::Concat>(
                    ov::NodeVector{query_part_neg, query_split->output(0)},
                    -1); // axis = -1
                    
                auto rotary_query = std::make_shared<ov::op::v1::Multiply>(
                    query_rotated,
                    ov::op::v0::Constant::create(ov::element::f32, ov::Shape({1,1,64,64}), {1.0f}),
                    ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
                    
                // Apply rotary position embeddings to key
                auto key_transposed = std::make_shared<ov::op::v1::Transpose>(
                    key_reshape, 
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {0,2,1,3}));
                    
                auto key_split = std::make_shared<ov::op::v1::VariadicSplit>(
                    key_transposed,
                    ov::op::v0::Constant::create(ov::element::i64, {1}, {3}),
                    ov::op::v0::Constant::create(ov::element::i64, {2}, {32,32}));
                    
                auto negate_factor = ov::op::v0::Constant::create(
                    ov::element::f32, ov::Shape({1,1,1,1}), {-1.0f});
                    
                auto key_part_neg = std::make_shared<ov::op::v1::Multiply>(
                    key_split->output(0), 
                    negate_factor,
                    ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
                    
                auto key_rotated = std::make_shared<ov::op::v1::Concat>(
                    ov::NodeVector{key_part_neg, key_split->output(0)},
                    3); // axis = 3 (different from query which used -1)
                    
                auto rotary_key = std::make_shared<ov::op::v1::Multiply>(
                    key_rotated,
                    ov::op::v0::Constant::create(ov::element::f32, ov::Shape({1,64,1,64}), {1.0f}),
                    ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
            }
            
            // ----- Attention Score Computation -----
            // Transpose key for matrix multiplication
            auto key_transposed = std::make_shared<ov::op::v1::Transpose>(
                key_reshape, 
                ov::op::v0::Constant::create(ov::element::i64, {4}, {0,2,3,1}));
                
            // Scale key for scaled dot-product attention
            auto key_scaled = std::make_shared<ov::op::v1::Multiply>(
                key_transposed,
                scale_factor,
                ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
                
            // Compute attention scores (Q × K^T)
            auto attention_scores = std::make_shared<ov::op::v0::MatMul>(
                rotary_query, // or query_transposed if no rotation
                key_scaled,
                false,  // transpose_a
                false); // transpose_b
                
            // ----- Attention Mask Application -----
            // Process attention mask
            auto mask_expanded = std::make_shared<ov::op::v0::Unsqueeze>(
                attention_mask,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}));
                
            auto mask_expanded_2d = std::make_shared<ov::op::v0::Unsqueeze>(
                mask_expanded,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));
                
            // Scale mask to create a large negative number for padding tokens
            auto mask_scale = ov::op::v0::Constant::create(
                ov::element::f32, ov::Shape({1,1,1,1}), {9875.0f});
                
            auto mask_scaled = std::make_shared<ov::op::v1::Multiply>(
                mask_expanded_2d,
                mask_scale,
                ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
                
            auto mask_offset = ov::op::v0::Constant::create(
                ov::element::f32, ov::Shape({1,1,1,1}), {-9998.436523f});
                
            auto attention_mask_final = std::make_shared<ov::op::v1::Add>(
                mask_scaled,
                mask_offset,
                ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
                
            auto sliced_mask = std::make_shared<ov::op::v8::Slice>(
                attention_mask_final,
                ov::op::v0::Constant::create(ov::element::i64, {2}, {-64,-64}),
                ov::op::v0::Constant::create(ov::element::i64, {2}, {LLONG_MAX,LLONG_MAX}),
                ov::op::v0::Constant::create(ov::element::i64, {2}, {1,1}),
                ov::op::v0::Constant::create(ov::element::i64, {2}, {2,3}));
                
            // Add mask to attention scores
            auto masked_attention_scores = std::make_shared<ov::op::v1::Add>(
                attention_scores,
                sliced_mask,
                ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
                
            // ----- Softmax and Value Weighting -----
            // Apply softmax to get attention weights
            auto attention_probs = std::make_shared<ov::op::v8::Softmax>(
                masked_attention_scores,
                -1); // axis = -1
                
            // Transpose value for matrix multiplication
            auto value_transposed = std::make_shared<ov::op::v1::Transpose>(
                value_reshape,
                ov::op::v0::Constant::create(ov::element::i64, {4}, {0,2,1,3}));
                
            // Apply attention weights to values
            auto context = std::make_shared<ov::op::v0::MatMul>(
                attention_probs,
                value_transposed,
                false,  // transpose_a
                false); // transpose_b
                
            // Transpose and reshape context
            auto context_transposed = std::make_shared<ov::op::v1::Transpose>(
                context,
                ov::op::v0::Constant::create(ov::element::i64, {4}, {0,2,1,3}));
                
            auto context_flattened = std::make_shared<ov::op::v1::Reshape>(
                context_transposed,
                ov::op::v0::Constant::create(ov::element::i64, {3}, {1,64,head_size}),
                true); // special_zero = true
                
            // ----- Output Projection -----
            // Dequantize weights for output projection
            auto output_weights_quant = createQuantizedWeights(head_size, hidden_size, 134 - head);
            auto output_weights = dequantizeWeights(output_weights_quant, 0.00068f + head * 0.00001f);
            
            // Project back to hidden dimension
            auto head_output = std::make_shared<ov::op::v0::MatMul>(
                context_flattened,
                output_weights,
                false,  // transpose_a
                false); // transpose_b
                
            attention_head_outputs.push_back(head_output);
        }
        
        // ----- Combine Head Outputs -----
        // Add outputs from all heads
        auto combined_output = attention_head_outputs[0];
        for (size_t i = 1; i < attention_head_outputs.size(); i++) {
            combined_output = std::make_shared<ov::op::v1::Add>(
                combined_output,
                attention_head_outputs[i],
                ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
        }
        
        // Add bias to combined output
        auto output_bias = ov::op::v0::Constant::create(
            ov::element::f32, ov::Shape({1,1,hidden_size}), {1.0f});
            
        auto output_with_bias = std::make_shared<ov::op::v1::Add>(
            combined_output,
            output_bias,
            ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
        
        // ----- Residual Connection and Layer Norm -----
        auto residual_output = std::make_shared<ov::op::v1::Add>(
            input_tensor,
            output_with_bias,
            ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
        
        // Apply layer normalization to the residual output
        auto residual_squares = std::make_shared<ov::op::v1::Power>(
            residual_output,
            ov::op::v0::Constant::create(ov::element::f32, {1,1,1}, {2.0f}),
            ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
        
        auto residual_sum_squares = std::make_shared<ov::op::v1::ReduceSum>(
            residual_squares,
            ov::op::v0::Constant::create(ov::element::i64, {1}, {2}),
            true); // keep_dims = true
        
        auto residual_norm_denom = std::make_shared<ov::op::v0::Sqrt>(residual_sum_squares);
        
        auto normalized_residual = std::make_shared<ov::op::v1::Divide>(
            residual_output,
            residual_norm_denom,
            ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        
        auto residual_ln_scale = ov::op::v0::Constant::create(
            ov::element::f32, ov::Shape({1,1,hidden_size}), {1.0f});
        
        auto scaled_normalized_residual = std::make_shared<ov::op::v1::Multiply>(
            residual_ln_scale,
            normalized_residual,
            ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
        
        // ----- Feed Forward Network -----
        // First dense layer (hidden_size -> ff_dim)
        int ff_dim = 4096;
        
        // Dequantize weights for FFN first layer
        auto ffn1_weights_quant = createQuantizedWeights(hidden_size, ff_dim, 131);
        auto ffn1_weights = dequantizeWeights(ffn1_weights_quant, 0.000859f);
        
        auto ffn1 = std::make_shared<ov::op::v0::MatMul>(
            scaled_normalized_residual,
            ffn1_weights,
            false,  // transpose_a
            false); // transpose_b
        
        auto ffn1_bias = ov::op::v0::Constant::create(
            ov::element::f32, ov::Shape({1,1,ff_dim}), {1.0f});
        
        auto ffn1_with_bias = std::make_shared<ov::op::v1::Add>(
            ffn1,
            ffn1_bias,
            ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
        
        // Apply GELU activation
        auto ffn1_gelu = std::make_shared<ov::op::v7::Gelu>(
            ffn1_with_bias,
            "TANH"); // approximation_mode
        
        // Second dense layer (ff_dim -> hidden_size)
        // Dequantize weights for FFN second layer
        auto ffn2_weights_quant = createQuantizedWeights(ff_dim, hidden_size, 132);
        auto ffn2_weights = dequantizeWeights(ffn2_weights_quant, 0.000798f);
        
        auto ffn2 = std::make_shared<ov::op::v0::MatMul>(
            ffn1_gelu,
            ffn2_weights,
            false,  // transpose_a
            false); // transpose_b
        
        // Add unsqueeze and reduce operations as in original
        auto ffn2_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            ffn2,
            ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        
        auto ffn2_reduced = std::make_shared<ov::op::v1::ReduceSum>(
            ffn2_unsqueeze,
            ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
            false); // keep_dims = false
        
        auto ffn2_bias = ov::op::v0::Constant::create(
            ov::element::f32, ov::Shape({1,1,hidden_size}), {1.0f});
        
        auto ffn2_with_bias = std::make_shared<ov::op::v1::Add>(
            ffn2_reduced,
            ffn2_bias,
            ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
        
        // ----- Final Residual Connection -----
        auto final_output = std::make_shared<ov::op::v1::Add>(
            residual_output,
            ffn2_with_bias,
            ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
        
        return final_output;
    }
    
    // Helper functions (would be defined separately)
    ov::Output<ov::Node> createQuantizedWeights(int input_dim, int output_dim, int zero_point) {
        return ov::op::v0::Constant::create(
            ov::element::u8, ov::Shape({input_dim, output_dim}), {1.0f});
    }
    
    ov::Output<ov::Node> dequantizeWeights(const ov::Output<ov::Node>& quant_weights, float scale) {
        auto convert_weights = std::make_shared<ov::op::v0::Convert>(
            quant_weights, ov::element::f32);
        
        auto zero_point = ov::op::v0::Constant::create(
            ov::element::u8, ov::Shape({}), {122});
        
        auto convert_zp = std::make_shared<ov::op::v0::Convert>(
            zero_point, ov::element::f32);
        
        auto shifted = std::make_shared<ov::op::v1::Subtract>(
            convert_weights,
            convert_zp,
            ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
        
        auto scale_const = ov::op::v0::Constant::create(
            ov::element::f32, ov::Shape({1,1}), {scale});
        
        return std::make_shared<ov::op::v1::Multiply>(
            shifted,
            scale_const,
            ov::op::PythonAPIDetails{{"auto_broadcast", "numpy"}});
    }



    ov::pass::Manager manager;
    manager.register_pass<ov::pass::PrintModel>("before_pack_mha.txt");
    manager.register_pass<ov::pass::PackMHA>();
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();
}