// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/pass/pattern/multi_matcher.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
TBA
 */
class TRANSFORMATIONS_API PackMHA;

}  // namespace ov

namespace ov::pass {

/**
* @brief Detects and fuses unrolled MultiHeadAttention structures:
* - Multiple Q/K/V projections (MatMul+Add)
* - Per-head SDPA paths (MatMul, Softmax, etc)
* - Final attention merge via Add or Concat
*/
class PackMHA : public ov::pass::MultiMatcher {
public:
    OPENVINO_RTTI("PackMHA");

    PackMHA();
};

}