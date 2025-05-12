#include "transformations/utils/block_collection.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/pack_mha.hpp"

using namespace ov;
using namespace ov::pass;
using namespace ov::pass::pattern;

PackMHA::PackMHA() : MultiMatcher("PackMHA") {
    // input -> Normalization Pattern
    auto norm_input = any_input();
    auto norm_block = blocks::l2_norm_block(norm_input);

    // Projection -> SDPA preprocessing Pattern
    auto proj_input = any_input();
    auto proj_weight = any_input();
    auto proj_block = blocks::qkv_projection_block(proj_input, proj_weight);
    auto pre_sdpa = blocks::sdpa_preprocessing_block(proj_block);

    // SDPA pattern + linear projection
    auto q = any_input();
    auto k = any_input();
    auto v = any_input();
    auto sdpa = blocks::sdpa_block(q, k, v);
    auto post = blocks::post_sdpa_proj(sdpa);

    auto callback = [=](const std::unordered_map<std::shared_ptr<Node>, std::vector<PatternValueMap>>& matches) {
        // Core fusion logic here
        // Extract matched subgraphs and replace them with fused equivalent

        for (const auto& [root, patterns] : matches) {
            for (const auto& m : patterns) {
                // Debug: show what got matched
                for (const auto& [pattern_node, real_value] : m) {
                    std::cout << "Matched: " << pattern_node->get_friendly_name()
                              << " => " << real_value.get_node()->get_friendly_name() << "\n";
                }

                // TODO: Fuse heads, pack Q/K/V, replace SDPA branches
            }
        }

        return true;
    };

    register_patterns({norm_block, pre_sdpa, post},
                      callback,
                      true);
}
