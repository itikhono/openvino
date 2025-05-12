// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "transformations/utils/block_collection.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/pass/pattern/op/block.hpp"

namespace ov::pass::pattern::blocks {

using namespace ov::op;

std::shared_ptr<Node> l2_norm_block(const Output<Node>& input) {
    auto mean_sub = wrap_type<v1::Subtract>({input, any_input()});
    auto pow = wrap_type<v1::Power>({mean_sub, any_input()});
    auto var = wrap_type<v1::ReduceMean>({pow});
    auto sqrt = wrap_type<v0::Sqrt>({var});
    auto div = wrap_type<v1::Divide>({mean_sub, sqrt});
    auto scale = wrap_type<v1::Multiply>({div, any_input()});
    auto shift = wrap_type<v1::Add>({scale, any_input()});

    return std::make_shared<pattern::op::Block>(OutputVector{input}, OutputVector{shift});
}

std::shared_ptr<Node> qkv_projection_block(const Output<Node>& input, const Output<Node>& weight) {
    auto mm = wrap_type<v0::MatMul>({input, weight});
    auto bias = wrap_type<v1::Add>({mm, any_input()});

    return std::make_shared<pattern::op::Block>(OutputVector{input, weight}, OutputVector{bias});
}

std::shared_ptr<Node> sdpa_preprocessing_block(const Output<Node>& input) {
    auto reshape = wrap_type<v1::Reshape>({input, any_input()});
    auto trans1 = wrap_type<v1::Transpose>({reshape, any_input()});
    auto split = wrap_type<v1::VariadicSplit>({trans1, any_input(), any_input()});

    auto mul0 = wrap_type<v1::Multiply>({split, any_input()});
    auto mul1 = wrap_type<v1::Multiply>({split, any_input()});
    auto concat = wrap_type<v0::Concat>({mul0, mul1});

    auto post = wrap_type<v1::Multiply>({concat, any_input()});
    auto trans2 = wrap_type<v1::Transpose>({post, any_input()});
    auto out = wrap_type<v1::Add>({trans2, any_input()});

    return std::make_shared<pattern::op::Block>(OutputVector{input}, OutputVector{out});
}

std::shared_ptr<Node> sdpa_block(const Output<Node>& q,
                                 const Output<Node>& k,
                                 const Output<Node>& v) {
    auto kT = wrap_type<v1::Transpose>({k, any_input()});
    auto qk = wrap_type<v0::MatMul>({q, kT});
    auto bias_add = wrap_type<v1::Add>({qk, any_input()});
    auto softmax = wrap_type<v1::Softmax>({bias_add});
    auto qkv = wrap_type<v0::MatMul>({softmax, v});

    return std::make_shared<pattern::op::Block>(OutputVector{q, k, v}, OutputVector{qkv});
}

std::shared_ptr<Node> post_sdpa_proj(const Output<Node>& qkv) {
    auto t2 = wrap_type<v1::Transpose>({qkv, any_input()});
    auto reshaped = wrap_type<v1::Reshape>({t2, any_input()});
    auto proj = wrap_type<v0::MatMul>({reshaped, any_input()});
    auto out = wrap_type<v1::Add>({proj, any_input()});

    return std::make_shared<pattern::op::Block>(OutputVector{qkv}, OutputVector{out});
}

}