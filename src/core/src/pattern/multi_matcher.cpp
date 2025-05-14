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
    }
}

bool MultiMatcher::run_on_model(const std::shared_ptr<Model>& model) {
    bool changed = false;
    m_matched_nodes.clear();

    std::unordered_map<std::shared_ptr<Node>, std::vector<PatternValueMap>> matches_by_pattern;
    std::cout << "XXXXXXXX cnt of operations: " << model->get_ordered_ops().size() << std::endl;
    int cnt_of_match_started = 0;
    for (const auto& node : model->get_ordered_ops()) {
        for (const auto& pattern : m_patterns) {
            if (m_matched_nodes.count(node.get()) > 0 && m_all_roots.count(node.get()) == 0) {
                std::cout << "Pattern conflict detected for node: " << node << std::endl;
            }
            auto conflict = m_matched_nodes.count(node.get()) > 0 && m_all_roots.count(node.get()) == 0;
            if (conflict) {
                std::cout << "Pattern conflict detected, skipping this node." << std::endl;
                continue;
            }
            cnt_of_match_started++;
            Matcher matcher(pattern.pattern, m_name, pattern.strict_mode);
            if (!matcher.match(node->output(0)))
                continue;


            m_all_roots.insert(node.get());
            const auto& match_map = matcher.get_pattern_value_map();
            const auto matched_nodes = matcher.get_matched_nodes();

            for (const auto& n : matched_nodes) {
                m_matched_nodes.insert(n.get());
            }

            matches_by_pattern[pattern.root_ptr].push_back(match_map);
            std::cout << "XXXXXXXX Pattern Matched" << std::endl;
            break;
        }
    }
    std::cout << "XXXXXXXX cnt of match started: " << cnt_of_match_started << std::endl;

    if (!matches_by_pattern.empty()) {
        m_callback(matches_by_pattern);
        changed = true;
    }

    return changed;
}
