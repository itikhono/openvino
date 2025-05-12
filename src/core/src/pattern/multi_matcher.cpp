// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include "openvino/pass/pattern/multi_matcher.hpp"

using namespace ov::pass;
using namespace ov::pass::pattern;

MultiMatcher::MultiMatcher(const std::string& name)
        : m_name(name) {}

void MultiMatcher::register_patterns(const std::vector<std::shared_ptr<Node>>& patterns,
                                     Callback callback,
                                     bool strict) {
    m_callback = std::move(callback);
    m_patterns.clear();
    m_all_roots.clear();
    for (const auto& p : patterns) {
        m_patterns.push_back(PatternEntry{p->output(0), p, strict});
        m_all_roots.insert(p.get());
    }
}
bool MultiMatcher::run_on_model(const std::shared_ptr<Model>& model) {
    bool changed = false;
    m_matched_nodes.clear();

    std::unordered_map<std::shared_ptr<Node>, std::vector<PatternValueMap>> matches_by_pattern;

    for (const auto& node : model->get_ordered_ops()) {
        for (const auto& pattern : m_patterns) {
            Matcher matcher(pattern.pattern, m_name, pattern.strict_mode);

            if (!matcher.match(node->output(0)))
                continue;

            const auto& match_map = matcher.get_pattern_value_map();
            const auto matched_nodes = matcher.get_matched_nodes();

            bool conflict = std::any_of(matched_nodes.begin(), matched_nodes.end(), [&](const std::shared_ptr<Node>& n) {
                return m_matched_nodes.count(n.get()) > 0 && m_all_roots.count(n.get()) == 0;
            });

            if (conflict)
                continue;

            for (const auto& n : matched_nodes) {
                m_matched_nodes.insert(n.get());
            }

            matches_by_pattern[pattern.root_ptr].push_back(match_map);
            break;
        }
    }

    if (!matches_by_pattern.empty()) {
        m_callback(matches_by_pattern);
        changed = true;
    }

    return changed;
}
