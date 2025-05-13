
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/block.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::pass::pattern::op;

Block::Block(const OutputVector& inputs,
             const OutputVector& outputs,
             const std::string& name)
        : Pattern(inputs, op::Predicate{}),  // base Pattern
          m_inputs(inputs),
          m_outputs(outputs) {
    set_output_type(0, element::dynamic, PartialShape::dynamic());
    set_friendly_name(name);
    extract_named_outputs();
}

std::shared_ptr<Node> Block::clone_with_new_inputs(const OutputVector& new_args) const {
    return std::make_shared<Block>(m_inputs, m_outputs, get_friendly_name());
}

void Block::extract_named_outputs() {
    std::deque<Output<Node>> queue(m_outputs.begin(), m_outputs.end());
    std::unordered_set<Node*> visited;

    while (!queue.empty()) {
        auto output = queue.front();
        queue.pop_front();
        auto node = output.get_node();
        if (!visited.insert(node).second)
            continue;

        auto name = node->get_friendly_name();
        if (!name.empty()) {
            m_named_outputs[name] = output;
        }

        for (const auto& input : node->input_values()) {
            queue.push_back(input);
        }
    }
}

bool Block::match_value(Matcher* matcher,
                        const Output<Node>& pattern_value,
                        const Output<Node>& graph_value) {
    auto block_pattern_root = m_outputs.front();
    auto local_matcher = std::make_shared<Matcher>(block_pattern_root.get_node_shared_ptr(), "BlockMatcher");
    if (!local_matcher->match_value(block_pattern_root, graph_value)) {
        return false;
    }

    auto& local_pm = local_matcher->get_pattern_value_map();

    OutputVector real_inputs, real_outputs;

    for (const auto& input : m_inputs) {
        if (local_pm.count(input.get_node_shared_ptr())) {
            real_inputs.push_back(local_pm.at(input.get_node_shared_ptr()));
        }
    }

    for (const auto& output : m_outputs) {
        if (local_pm.count(output.get_node_shared_ptr())) {
            real_outputs.push_back(local_pm.at(output.get_node_shared_ptr()));
        }
    }

    auto matched_block = std::make_shared<Block>(real_inputs, real_outputs, get_friendly_name());

    auto& pattern_map = matcher->get_pattern_value_map();
    pattern_map[shared_from_this()] = matched_block->output(0);
    return true;
}