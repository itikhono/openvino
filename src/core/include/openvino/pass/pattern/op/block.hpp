#pragma once

#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace ov {
namespace pass {
namespace pattern {
namespace op {

/**
 * @brief Block: reusable pattern subgraph with named internal pattern ops.
 */
class OPENVINO_API Block : public Pattern {
public:
    OPENVINO_RTTI("Block");

    Block(const OutputVector& inputs,
          const OutputVector& outputs,
          const std::string& name = "");

    bool match_value(Matcher* matcher,
                     const Output<Node>& pattern_value,
                     const Output<Node>& graph_value) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const std::unordered_map<std::string, Output<Node>>& get_named_outputs() const {
        return m_named_outputs;
    }

    const OutputVector& get_inputs() const {
        return m_inputs;
    }

    const OutputVector& get_outputs() const {
        return m_outputs;
    }

/*    void set_named_outputs(const std::unordered_map<std::string, Output<Node>>& named) {
        m_named_outputs = named;
    }*/

private:
    OutputVector m_inputs;
    OutputVector m_outputs;
    std::unordered_map<std::string, Output<Node>> m_named_outputs;

    void extract_named_outputs();
};

}  // namespace op
}  // namespace pattern
}  // namespace pass
}  // namespace ov