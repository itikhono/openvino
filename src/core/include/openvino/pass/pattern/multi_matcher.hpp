// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace ov {
namespace pass {

/// MultiMatcher: Applies multiple patterns to the model, collects grouped non-overlapping matches,
/// and invokes a single callback with all matches grouped by pattern root.
class OPENVINO_API MultiMatcher : public ov::pass::ModelPass {
public:
    using Callback = std::function<void(
            const std::unordered_map<std::shared_ptr<Node>, std::vector<pattern::PatternValueMap>>&) >;

    OPENVINO_RTTI("MultiMatcher", "0", ModelPass);

    explicit MultiMatcher(const std::string& name = "MultiMatcher");

    /// Register multiple patterns with a unified callback
    void register_patterns(const std::vector<std::shared_ptr<Node>>& patterns, Callback callback, bool strict = false);

    /// Run all matchers once across the model, apply callback on grouped matches
    bool run_on_model(const std::shared_ptr<Model>& model) override;

private:
    struct PatternEntry {
        ov::Output<ov::Node> pattern;
        std::shared_ptr<ov::Node> root_ptr;
        bool strict_mode = false;
    };

    std::string m_name;
    Callback m_callback;
    std::vector<PatternEntry> m_patterns;
    std::unordered_set<Node*> m_matched_nodes;
    std::unordered_set<Node*> m_all_roots;
};

}  // namespace pass
}  // namespace ov