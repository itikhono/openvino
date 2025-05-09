

#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/matcher.hpp"

using namespace ov;
using namespace ov::pass::pattern::op;

Block::Block(OutputVector inputs,
             OutputVector outputs,
             std::string name)
        : Pattern(outputs.empty() ? NodeVector{std::make_shared<op::Label>()} : as_node_vector(outputs)),
          m_inputs(std::move(inputs)),
          m_outputs(std::move(outputs)) {
    set_friendly_name(std::move(name));
}

bool Block::match_value(Matcher* matcher,
                        const Output<Node>& pattern_value,
                        const Output<Node>& graph_value) {
    auto saved_state = matcher->start_match();

    if (m_outputs.empty())
        return saved_state.finish(false);

    auto root = m_outputs.front().get_node_shared_ptr();
    if (!root->match_value(matcher, pattern_value, graph_value)) {
        return saved_state.finish(false);
    }

    for (const auto& input : m_inputs) {
        if (!matcher->get_pattern_value_map().count(input.get_node_shared_ptr())) {
            return saved_state.finish(false);
        }
    }

    matcher->get_pattern_value_map()[shared_from_this()] = graph_value;
    return saved_state.finish(true);
}