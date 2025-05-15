#include "transformations/utils/block_collection.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/pack_mha.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/pattern/op/optional.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::pass;
using namespace ov::pass::pattern;

namespace {

std::vector<std::shared_ptr<Node>> get_sdpa_order(const std::unordered_set<Node*>& post_sdpa_proj) {
    std::vector<std::shared_ptr<Node>> post_sdpa_ordered;

    // find the first Add in the chain:
    // AddN(Add2(Add1(post_sdpa_proj_1, post_sdpa_proj_2), post_sdpa_proj_3), post_sdpa_proj_N + 1)
    ov::op::v1::Add* current_add = nullptr;
    for (const auto& proj_node : post_sdpa_proj) {
        std::cout << "proj_node: " << proj_node << "\n";
        const auto& output = proj_node->output(0);
        const auto& targets = output.get_target_inputs();

        if (targets.size() != 1)
            continue;  // Only proceed if there's a single consumer

        const auto& input = *targets.begin();
        auto add_node = ov::as_type<ov::op::v1::Add>(input.get_node());
        if (!add_node)
            continue;

        auto lhs = add_node->input_value(0).get_node();
        auto rhs = add_node->input_value(1).get_node();

        if (post_sdpa_proj.count(lhs) && post_sdpa_proj.count(rhs)) {
            current_add = add_node;
            post_sdpa_ordered.push_back(lhs->shared_from_this());
            post_sdpa_ordered.push_back(rhs->shared_from_this());
            break;
        }
    }

    if (!current_add) {
        std::cout << "Can't find the first Add in the chain\n";
        return {};
    }

    // find the remaining post_sdpa_proj order, starting from the first Add
    // AddN(Add2(Add1(post_sdpa_proj_1, post_sdpa_proj_2), post_sdpa_proj_3), post_sdpa_proj_N + 1)
    while (true) {
        const auto& targets = current_add->output(0).get_target_inputs();
        if (targets.size() != 1) {
            std::cout << "Unexpected number of targets for Add: " << targets.size() << "\n";
            return {};
        }

        auto next_node_input = targets.begin();
        current_add = ov::as_type<ov::op::v1::Add>(next_node_input->get_node());
        if (!current_add)
            break;

        auto another_idx = 1 - next_node_input->get_index();
        auto input_node = current_add->input_value(another_idx).get_node();
        if (post_sdpa_proj.count(input_node)) {
            post_sdpa_ordered.push_back(input_node->shared_from_this());
        } else {
            std::cout << "Unexpected node in Add chain: " << current_add << "\n";
            break;
        }
    }
    return post_sdpa_ordered;
}

}

PackMHA::PackMHA() : MultiMatcher("PackMHA") {
    // input -> Normalization Pattern:
    auto norm_input = any_input();
    auto norm_block = blocks::l2_norm_block(norm_input);

    // Attention mask
    //auto attention_mask = blocks::attention_mask();

    // Projection -> SDPA preprocessing (RoPE?) Pattern:
    //auto proj_input = any_input();
    auto constant = wrap_type<v0::Constant>();
    auto convert = optional<v0::Convert>(constant);

    auto dq = blocks::dq_constant_block();
    auto mm = wrap_type<v0::MatMul>({norm_block, dq | convert});
    auto bias = optional<v1::Add>({mm, any_input()});
    //auto pre_sdpa = blocks::sdpa_preprocessing_block(proj_block);

    // SDPA + linear projection Pattern:
    auto q = blocks::sdpa_preprocessing_block(any_input());
    auto k = blocks::sdpa_preprocessing_block(any_input());
    auto v = any_input();

    auto reshape_v = wrap_type<v1::Reshape>({v, any_input()});
    auto vT = optional<v1::Transpose>({reshape_v, any_input()});

    auto sdpa = blocks::sdpa_block(q, k, vT);
    auto t2 = wrap_type<v1::Transpose>({sdpa, any_input()});
    auto reshaped = wrap_type<v1::Reshape>({t2, any_input()});
    auto proj = wrap_type<v0::MatMul>({reshaped, any_input()});

    auto callback = [=](const std::unordered_map<std::shared_ptr<Node>, std::vector<PatternValueMap>>& matches) {
        using namespace ov::pass::pattern::op;

        // DEBUG
        std::cout << "matches size " << matches.size() << std::endl;
        for (const auto& [root, patterns] : matches) {
            std::cout << "pattern size" << patterns.size() << std::endl;
            for (const auto& m : patterns) {
                std::cout << "pattern = " << root->get_friendly_name() << std::endl;
                std::cout << m.at(root->shared_from_this()).get_node()->get_friendly_name() << std::endl;
            }
        }
        // DEBUG END

        std::unordered_set<Node*> post_sdpa_proj;
        std::unordered_map<Node*, const PatternValueMap*> node_to_pm;
        for (const auto& pm : matches.at(proj)) {
            auto root_node = pm.at(proj).get_node();
            post_sdpa_proj.insert(root_node);
            node_to_pm[root_node] = &pm;
        }

        auto post_sdpa_proj_ordered = get_sdpa_order(post_sdpa_proj);
        if (post_sdpa_proj_ordered.empty()) {
            return;
        }
    };

    register_patterns({bias, proj},
                      callback,
                      true);
}
