// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/pack_qkv_proj.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/power.hpp"
#include "openvino/core/graph_util.hpp"

#include <regex>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <optional>

using namespace ov;
using namespace ov::op;
using namespace ov::pass;

namespace {

struct ProjectionHead {
    std::shared_ptr<v0::MatMul> matmul;
    Output<Node> weight;
    Output<Node> bias;
    int64_t index;
};

using GroupedEntries = std::map<std::string, std::vector<ProjectionHead>>;

std::optional<Output<Node>> extract_weight(const Output<Node>& node) {
    if (auto c = ov::as_type_ptr<v0::Constant>(node.get_node_shared_ptr())) {
        return c;
    }

    auto mul = ov::as_type_ptr<v1::Multiply>(node.get_node_shared_ptr());
    if (!mul)
        return std::nullopt;

    auto sub = ov::as_type_ptr<v1::Subtract>(mul->input_value(0).get_node_shared_ptr());
    auto scale = ov::as_type_ptr<v0::Constant>(mul->input_value(1).get_node_shared_ptr());
    if (!sub || !scale)
        return std::nullopt;

    auto conv = ov::as_type_ptr<v0::Convert>(sub->input_value(0).get_node_shared_ptr());
    auto zp_conv = ov::as_type_ptr<v0::Convert>(sub->input_value(1).get_node_shared_ptr());
    if (!conv || !zp_conv)
        return std::nullopt;

    auto w_const = ov::as_type_ptr<v0::Constant>(conv->input_value(0).get_node_shared_ptr());
    if (!w_const)
        return std::nullopt;

    return mul;
}

std::tuple<std::string, int64_t> group_searching_criteria(const std::shared_ptr<ov::Node>& matmul) {
    static const std::regex pattern(R"((q_proj|k_proj|v_proj)\.(\d+))");
    static std::smatch match;
    const std::string& name = matmul->get_friendly_name();
    if (!std::regex_match(name, match, pattern))
        return {"", -1};
    const std::string prefix = match[1];
    const int64_t index = std::stoll(match[2]);
    return {prefix, index};
}

GroupedEntries collect_projection_groups(const Output<Node>& root_input) {
    GroupedEntries qkv_grouped;
    for (const auto& input : root_input.get_target_inputs()) {
        auto mm = ov::as_type_ptr<v0::MatMul>(input.get_node()->shared_from_this());
        if (!mm || mm->input_value(0).get_node() != root_input.get_node())
            continue;

        const auto [prefix, index] = group_searching_criteria(mm);
        if (prefix.empty()) {
            continue;
        }

        Output<Node> bias;
        if (mm->get_output_target_inputs(0).size() == 1) {
            auto target_input = mm->get_output_target_inputs(0);
            if (auto add = ov::as_type<v1::Add>(target_input.begin()->get_node())) {
                auto idx = 1 - target_input.begin()->get_index();
                bias = add->input_value(idx);
            }
        }

        auto weight = extract_weight(mm->input_value(1));
        if (!weight)
            continue;

        qkv_grouped[prefix].push_back(ProjectionHead{mm, *weight, bias, index});
    }
    return qkv_grouped;
}

Output<Node> match_l2_norm(const Output<Node>& input) {
    auto pow_const = pattern::wrap_type<v0::Constant>();
    auto pow = pattern::wrap_type<v1::Power>({input, pow_const});
    auto reduce_axes = pattern::wrap_type<v0::Constant>();
    auto reduce = pattern::wrap_type<v1::ReduceSum>({pow, reduce_axes});
    auto sqrt = pattern::wrap_type<v0::Sqrt>({reduce});
    auto div = pattern::wrap_type<v1::Divide>({input, sqrt});
    auto scale_const = pattern::wrap_type<v0::Constant>();
    return pattern::wrap_type<v1::Multiply>({div, scale_const});
}

}  // namespace

ov::pass::PackQKVProj::PackQKVProj() {
    MATCHER_SCOPE(PackQKVProj);

    auto input = pattern::any_input();
    auto l2_norm = match_l2_norm(input);
    auto weight = pattern::any_input();
    auto mm = pattern::wrap_type<v0::MatMul>({l2_norm, weight});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();

        // the common part for all head MatMul-s
        const auto root_input = pattern_map.at(l2_norm.get_node_shared_ptr());

        GroupedEntries qkv_heads_grouped = collect_projection_groups(root_input);
        for (auto& [group_name, heads] : qkv_heads_grouped) {
            if (heads.size() < 2)
                continue;

            std::sort(heads.begin(), heads.end(),
                      [](const ProjectionHead& a, const ProjectionHead& b) { return a.index < b.index; });

            OutputVector weights, biases;

            bool all_have_bias = true;
            for (const auto& e : heads) {
                weights.push_back(e.weight);
                if (e.bias.get_node())
                    biases.push_back(e.bias);
                else
                    all_have_bias = false;
            }

            auto concat_w = ov::op::util::make_try_fold<v0::Concat>(weights, 1);
            auto fused_mm = std::make_shared<v0::MatMul>(root_input, concat_w);
            fused_mm->set_friendly_name(group_name + "_fused_mm");

            Output<Node> fused = fused_mm;
            if (all_have_bias) {
                auto concat_b = ov::op::util::make_try_fold<v0::Concat>(biases, 1);
                auto fused_add = std::make_shared<v1::Add>(fused_mm, concat_b);
                fused_add->set_friendly_name(group_name + "_fused_add");
                fused = fused_add;
                ov::replace_node(heads[0].bias.get_node_shared_ptr(), fused.get_node_shared_ptr());
            } else {
                ov::replace_node(heads[0].matmul, fused.get_node_shared_ptr());
            }

            for (size_t i = 1; i < heads.size(); ++i) {
                if (all_have_bias) {
                    auto target_inputs = heads[i].bias.get_target_inputs();
                    for (const auto& target_input : target_inputs) {
                        heads[i].bias.remove_target_input(target_input);
                    }
                } else {
                    auto target_inputs = heads[i].matmul->output(0).get_target_inputs();
                    for (const auto& target_input : target_inputs) {
                        heads[i].matmul->output(0).remove_target_input(target_input);
                    }
                }
            }

            ov::copy_runtime_info(fused.get_node_shared_ptr(), heads[0].matmul);
        }
        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(mm, matcher_name);
    register_matcher(matcher, callback);
}
