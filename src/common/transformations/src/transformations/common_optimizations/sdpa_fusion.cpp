// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/sdpa_fusion.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

namespace ov {
namespace pass {

bool SDPAFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SDPAFusion);
    ov::pass::SymbolicOptimizations symbolic_optimizations(false, get_pass_config());

    auto symbolic_ctx_manager = symbolic_optimizations.get_manager();
    symbolic_ctx_manager->register_pass<ov::pass::SDPAFusionMatcher>();
    return symbolic_optimizations.run_on_model(model);
}

SDPAFusionMatcher::SDPAFusionMatcher() {
    MATCHER_SCOPE(SDPAFusionMatcher);
    using namespace ov::op;
    using namespace ov::pass::pattern;

    auto q = any_input(shape_matches("Batches..., S_q, D"));
    auto k = any_input(shape_matches("Batches..., D, S_kv"));
    auto v = any_input(shape_matches("Batches..., S_kv, D"));

    auto attn_scale = any_input();

    // No transpose check here, there are scenarios where k is not transposed and that uses equation (A*B)^T = B^T * A^T
    auto qk = wrap_type<v0::MatMul>({q, k}, shape_matches("Batches..., S_q, S_kv"));

    // Optional unsqueeze that is converted to Reshape
    auto unsqueeze_axis = wrap_type<v0::Constant>();
    auto qk_opt_unsqueeze = optional<v1::Reshape>({qk, unsqueeze_axis});

    auto qk_scaled = wrap_type<v1::Multiply>({qk_opt_unsqueeze, attn_scale});
    auto qk_opt_scaled = qk_scaled | qk_opt_unsqueeze;

    // optional mask add, there are patterns where before or/and after mask add buffer is reshaped
    auto mask = any_input();
    // Optional reshape befor adding mask
    auto qk_opt_scaled_pre_mask_shape = any_input();
    auto qk_opt_scaled_pre_mask_opt_reshaped = optional<v1::Reshape>({qk_opt_scaled, qk_opt_scaled_pre_mask_shape});
    
    // Optional mask add
    auto qk_opt_scaled_opt_mask_added = optional<v1::Add>({qk_opt_scaled_pre_mask_opt_reshaped, mask});
    // Optional reshape after adding mask
    auto qk_post_mask_shape = any_input();
    auto qk_post_mask_opt_reshaped = optional<v1::Reshape>({qk_opt_scaled_opt_mask_added, qk_post_mask_shape});

    auto softmax = wrap_type<v8::Softmax>({qk_post_mask_opt_reshaped}, shape_matches("Batches..., S_q, S_kv"));
    auto softmax_shape = any_input();
    auto softmax_opt_reshaped = optional<v1::Reshape>({softmax, softmax_shape});

    auto qkv_base = wrap_type<v0::MatMul>({softmax_opt_reshaped, v}, shape_matches("Batches..., S_q, D"),
                                                 {{"transpose_a", false}, {"transpose_b", false}});
    auto qkv_shape = any_input();
    auto qkv = optional<v1::Reshape>({qkv_base, qkv_shape});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        std::cout << "XXXXXXXXX SDPA matched" << std::endl;
        const auto& pattern_map = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto q_node = pattern_map.at(q).get_node_shared_ptr();
        auto k_node = pattern_map.at(k).get_node_shared_ptr();
        auto v_node = pattern_map.at(v).get_node_shared_ptr();

        if (pattern_map.at(qk).get_target_inputs().size() > 1 ||
            pattern_map.at(softmax).get_target_inputs().size() > 1) {
            return false;
        }
        if (pattern_map.count(qk_opt_scaled_opt_mask_added) &&
            (pattern_map.at(qk_opt_scaled_opt_mask_added).get_target_inputs().size() > 1 ||
             pattern_map.at(mask).get_partial_shape().size() > 4)) {
            return false;
        }

        auto T = q_node->output(0).get_element_type();
        ov::Output<ov::Node> scale_node;
        if (pattern_map.count(attn_scale)) {
            scale_node = pattern_map.at(attn_scale);
            auto attn_scale_out_ps = scale_node.get_partial_shape();
            // we need to be able to cast attn_scale layer to Constant layer
            // in order to read actual scale value
            float attn_scale_val = 0;
            if (!ov::op::util::get_constant_value<float>(scale_node.get_node_shared_ptr(), attn_scale_val))
                return false;

            scale_node = v0::Constant::create(T, ov::Shape{}, {attn_scale_val});
        } else {
            scale_node = v0::Constant::create(T, ov::Shape{}, {1.0});
        }

        Output<ov::Node> mask_input;
        if (pattern_map.count(mask) && pattern_map.count(qk_opt_scaled_opt_mask_added)) {
            ov::Output<ov::Node> qk_out = pattern_map.at(qk_opt_scaled_opt_mask_added);
            // Get shape of the first input
            auto qk_out_ps = qk_out.get_target_inputs().begin()->get_partial_shape();

            mask_input = pattern_map.at(mask);
            auto mask_input_ps = mask_input.get_partial_shape();

            if (!qk_out_ps.rank().is_static() || !mask_input_ps.rank().is_static())
                return false;
            if (qk_out_ps.size() > 4)
                return false;

            std::shared_ptr<v0::Unsqueeze> mask_unsqueeze;
            // mask should be broadcastable to qk shape
            if (!ov::PartialShape::broadcast_merge_into(qk_out_ps, mask_input_ps, AutoBroadcastType::NUMPY))
                return false;

            if (mask_input_ps.size() < qk_out_ps.size()) {
                size_t rank_diff = qk_out_ps.size() - mask_input_ps.size();
                std::vector<int64_t> axes(rank_diff);
                std::iota(axes.begin(), axes.end(), 0);
                mask_unsqueeze = std::make_shared<v0::Unsqueeze>(
                    mask_input,
                    v0::Constant::create(ov::element::i64, ov::Shape{rank_diff}, axes));
                mask_unsqueeze->set_friendly_name(mask->get_friendly_name());
                ov::copy_runtime_info(m.get_matched_nodes(), mask_unsqueeze);
                mask_input = mask_unsqueeze;
            }
        } else {
            mask_input = v0::Constant::create(T, ov::Shape{}, {0});
        }


        ov::OutputVector vec = {q_node, k_node, v_node};
        for (size_t i = 0; i < vec.size(); ++i) {
            if (vec[i].get_partial_shape().rank().is_dynamic()) {
                return false;
            }

            int diff  = 4 - vec[i].get_partial_shape().rank().get_length();
            if (diff > 0) {
                std::vector<size_t> axes(diff);
                std::iota(axes.begin(), axes.end(), 0);
                auto axes_node = v0::Constant::create(ov::element::i64, ov::Shape{static_cast<size_t>(diff)}, axes);
                auto unsqueeze = std::make_shared<v0::Unsqueeze>(vec[i], axes_node);
                vec[i] = unsqueeze;
            }

            if (i == 1) {
                // Transpose k
                auto transpose = std::make_shared<v1::Transpose>(vec[i], v0::Constant::create(ov::element::i64, ov::Shape{4},
                                                                                             {0, 1, 3, 2}));
                vec[i] = transpose;
            }
        }

        std::shared_ptr<ov::Node> sdpa = std::make_shared<v13::ScaledDotProductAttention>(vec[0],
                                                                                          vec[1],
                                                                                          vec[2],
                                                                                          mask_input,
                                                                                          scale_node,
                                                                                          false);

        sdpa->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);
        ov::replace_node(m.get_match_root(), sdpa);
        return true;
    };

    auto m = std::make_shared<Matcher>(qkv, "SDPAFusion");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace ov
