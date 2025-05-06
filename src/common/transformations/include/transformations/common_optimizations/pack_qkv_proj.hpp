// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief PackQKVProj transformation detects and fuses multiple MatMul (and optional Add) operations
 * used for unrolled Q, K, and V projections into a single packed MatMul (+ Add) per group.
 *
 * In this case each projection looks like:
 *     normalized_input → MatMul (per-head weight) → [optional Add (bias)]
 *
 * This transformation:
 * - Groups nodes by prefix (q_proj, k_proj, v_proj) using name pattern
 * - Sorts them by numerical suffix
 * - Concatenates weights and biases
 * - Replaces them with a single MatMul (+ optional Add) per group
 *
 * ----------------------------------------------------
 * Example: Before (unrolled projections per head)
 * ----------------------------------------------------
 *
 *             input (L2 normalized)
 *                   |
 *          ┌────────┼────────┐
 *          |        |        |
 *       MatMul   MatMul   MatMul       (q_proj.0, q_proj.1, q_proj.2)
 *          |        |        |
 *        [Add]    [Add]    [Add]       (optional bias)
 *          |        |        |
 *        out0     out1     out2
 *
 *
 * ----------------------------------------------------
 * Example: After (fused projection per group)
 * ----------------------------------------------------
 *
 *             input (L2 normalized)
 *                   |
 *                MatMul           (fused q_proj weights)
 *                   |
 *                [Add]            (fused q_proj biases if present)
 *                   |
 *            [q_proj output]
 *
 * Same logic applies to k_proj and v_proj groups.
 *
 * ----------------------------------------------------
 * Requirements:
 * ----------------------------------------------------
 * - Node names must match: q_proj.N, k_proj.N, v_proj.N
 * - All MatMuls in a group must share the same input
 * - Bias (Add) is optional but must exist for all heads to be fused
 * - Weight can be Constant or quantized: Convert → Subtract → Multiply
 */
class TRANSFORMATIONS_API PackQKVProj;

}  // namespace pass
}  // namespace ov

class ov::pass::PackQKVProj : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PackQKVProj");
    PackQKVProj();
};
