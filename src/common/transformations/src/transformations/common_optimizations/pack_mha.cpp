#include "transformations/utils/block_collection.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/pack_mha.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/add.hpp"
#include "openvino/pass/pattern/op/optional.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::pass;
using namespace ov::pass::pattern;

namespace {

std::shared_ptr<ov::op::v1::Add> find_first_add(const std::vector<Output<Node>>& post_sdpa_proj_outputs) {
    std::unordered_set<Node*> post_proj_set;
    for (const auto& out : post_sdpa_proj_outputs) {
        post_proj_set.insert(out.get_node());
    }

    for (const auto& out : post_sdpa_proj_outputs) {
        for (const auto& input : out.get_target_inputs()) {
            auto add = ov::as_type_ptr<ov::op::v1::Add>(input.get_node()->shared_from_this());
            if (!add)
                continue;

            auto lhs = add->input_value(0).get_node();
            auto rhs = add->input_value(1).get_node();

            // Only Add that has both inputs from post_sdpa_projection_block
            if (post_proj_set.count(lhs) && post_proj_set.count(rhs)) {
                return add;
            }
        }
    }

    return {};
}

}

PackMHA::PackMHA() : MultiMatcher("PackMHA") {
    // input -> Normalization Pattern:
    auto norm_input = any_input();
    auto norm_block = blocks::l2_norm_block(norm_input);

    // Projection -> SDPA preprocessing (RoPE?) Pattern:
    auto proj_input = any_input();
    auto proj_block = blocks::qkv_projection_block(proj_input);
    auto pre_sdpa = blocks::sdpa_preprocessing_block(proj_block);

    // SDPA + linear projection Pattern:
    auto q = any_input();
    auto k = any_input();
    auto v = any_input();
    auto reshape_v = wrap_type<v1::Reshape>({v, any_input()});
    auto vT = optional<v1::Transpose>({reshape_v, any_input()});
    auto sdpa = blocks::sdpa_block(q, k, vT);
    auto post_sdpa = blocks::post_sdpa_projection_block(sdpa);

    auto callback = [=](const std::unordered_map<std::shared_ptr<Node>, std::vector<PatternValueMap>>& matches) {
        using namespace ov::pass::pattern::op;

        ov::OutputVector post_sdpa_proj_outputs;
        for (const auto& pm : matches.at(post_sdpa)) {
            post_sdpa_proj_outputs.push_back(ov::as_type<Block>(pm.at(post_sdpa).get_node())->get_outputs()[0]);     
        }
        auto add = find_first_add(post_sdpa_proj_outputs);
        if (!add) {
            std::cout << "No add found" << std::endl;
            return;
        }

        std::cout << "matches size " << matches.size() << std::endl;
        for (const auto& [root, patterns] : matches) {
            for (const auto& m : patterns) {
                // Debug: show what got matched
                for (const auto& [pattern_node, real_value] : m) {
                    std::cout << "Matched: " << pattern_node->get_friendly_name()
                              << " => " << real_value.get_node()->get_friendly_name() << "\n";
                    std::cout<< ov::as_type_ptr<Block>(real_value.get_node_shared_ptr())->get_outputs()[0].get_node()->get_friendly_name() << std::endl;;
                }

                // TODO: Fuse heads, pack Q/K/V, replace SDPA branches
            }
        }


    };

    register_patterns({norm_block, pre_sdpa, post_sdpa},
                      callback,
                      true);
}
