// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/gru_sequence_fusion.hpp"

#include <memory>
#include <openvino/opsets/opset9.hpp>

#include "itt.hpp"
#include "ngraph_ops/augru_cell.hpp"
#include "ngraph_ops/augru_sequence.hpp"

using namespace std;
using namespace ov::opset9;
using namespace ov::pass::pattern;
using namespace ov::op::util;

ov::pass::SequenceFusion::SequenceFusion() {
    MATCHER_SCOPE(SequenceFusion);

    // todo: use RNNCellBase instead of these
    auto augru_cell = wrap_type<ov::op::internal::AUGRUCell>(
        {{any_input(), any_input(), any_input(), any_input(), any_input(), any_input()}});
    auto gru_cell = wrap_type<GRUCell>({any_input(), any_input(), any_input(), any_input(), any_input()});
    auto cell = make_shared<pattern::op::Or>(OutputVector{augru_cell, gru_cell});
    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto cell = m.get_match_root();
        for (const auto& target : cell->get_output_target_inputs(0)) {
            if (target.get_node()->get_type_name() == cell->get_type_name()) {
                return false;
            }
        }

        size_t hidden_size;
        if (auto last_cell = dynamic_pointer_cast<RNNCellBase>(cell)) {
            hidden_size = last_cell->get_hidden_size();
        } else {
            return false;
        }

        int cnt = 1;
        shared_ptr<Node> first_cell = cell;
        OutputVector attention_inputs;
        OutputVector inputs_to_concat;
        auto axis_0 = make_shared<Constant>(element::i64, Shape{}, 0);
        auto axis_1 = make_shared<Constant>(element::i64, Shape{}, 1);

        map<int, set<ov::Input<Node>>> inputs_to_redirect;
        while (true) {
            auto prev_node = first_cell->input_value(1).get_node_shared_ptr();
            if (auto gru_cell = dynamic_pointer_cast<RNNCellBase>(prev_node)) {
                if (gru_cell->get_hidden_size() != hidden_size) {
                    break;
                }
                auto in_X = first_cell->input(0);
                auto in_H = first_cell->input(1);
                inputs_to_concat.push_back(make_shared<Unsqueeze>(in_X.get_source_output(), axis_1));
                if (auto augru_cell = dynamic_pointer_cast<ov::op::internal::AUGRUCell>(gru_cell)) {
                    attention_inputs.push_back(make_shared<Unsqueeze>(augru_cell->input_value(5), axis_1));
                }
                for (const auto& input : prev_node->get_output_target_inputs(0)) {
                    if (input != in_H) {
                        inputs_to_redirect[cnt].insert(input);
                    }
                }
                first_cell = prev_node;
                cnt++;
            } else {
                break;
            }
        }

        if (cnt == 1)
            return false;
        reverse(inputs_to_concat.begin(), inputs_to_concat.end());
        const auto X_in = make_shared<Concat>(inputs_to_concat, 1);
        const auto Ht_in = make_shared<Unsqueeze>(first_cell->input_value(1), axis_1);

        const auto W_in =
            make_shared<Unsqueeze>(first_cell->input_value(2), axis_0);  // TODO: Check if the same for all nodes
        const auto R_in = make_shared<Unsqueeze>(first_cell->input_value(3), axis_0);
        const auto B_in = make_shared<Unsqueeze>(first_cell->input_value(4), axis_0);

        const auto& shape_node = ngraph::op::util::make_try_fold<opset9::ShapeOf>(cell->input_value(0));
        const auto& batch_dimension =
            ngraph::op::util::make_try_fold<opset9::Gather>(shape_node,
                                                            Constant::create(ov::element::i64, {1}, {0}),
                                                            axis_0);
        auto seq_lengths_scalar = Constant::create(ov::element::i64, {}, {cnt});
        auto sequence_lengths_in = ngraph::op::util::make_try_fold<Broadcast>(seq_lengths_scalar, batch_dimension);
        if (auto last_cell = dynamic_pointer_cast<GRUCell>(cell)) {
            cout << "XXXXXXXX GRUSequence pattern detected" << endl;
            const auto gru_sequence = make_shared<GRUSequence>(X_in,
                                                               Ht_in,
                                                               sequence_lengths_in,
                                                               W_in,
                                                               R_in,
                                                               B_in,
                                                               hidden_size,
                                                               op::RecurrentSequenceDirection::FORWARD,
                                                               last_cell->get_activations(),
                                                               last_cell->get_activations_alpha(),
                                                               last_cell->get_activations_beta(),
                                                               last_cell->get_clip(),
                                                               last_cell->get_linear_before_reset());

            auto squeeze_H = make_shared<Squeeze>(gru_sequence->output(1), axis_1);
            replace_node(cell, squeeze_H);
            cout << "XXXXXXXX GRUSequence pattern replaced" << endl;

            if (!inputs_to_redirect.empty()) {
                auto squeeze_Y = make_shared<Squeeze>(gru_sequence->output(0), axis_1);
                auto split = make_shared<Split>(squeeze_Y, axis_1, cnt - 1);

                for (const auto& it : inputs_to_redirect) {
                    for (const auto& in : it.second) {
                        auto squeeze = make_shared<Squeeze>(split->output(cnt - it.first - 1), axis_1);
                        in.replace_source_output(squeeze);
                    }
                }
            }
        } else if (auto last_cell = dynamic_pointer_cast<ov::op::internal::AUGRUCell>(cell)) {
            cout << "XXXXXXXX AUGRUSequence pattern detected" << endl;
            reverse(attention_inputs.begin(), attention_inputs.end());
            const auto A_in = make_shared<Concat>(attention_inputs, 1);
            const auto gru_sequence = make_shared<ov::op::internal::AUGRUSequence>(X_in,
                                                                                   Ht_in,
                                                                                   sequence_lengths_in,
                                                                                   W_in,
                                                                                   R_in,
                                                                                   B_in,
                                                                                   A_in,
                                                                                   hidden_size);

            auto squeeze_H = make_shared<Squeeze>(gru_sequence->output(1), axis_1);
            replace_node(cell, squeeze_H);
            cout << "XXXXXXXX AUGRUSequence pattern replaced" << endl;

            if (!inputs_to_redirect.empty()) {
                auto squeeze_Y = make_shared<Squeeze>(gru_sequence->output(0), axis_1);
                auto split = make_shared<Split>(squeeze_Y, axis_1, cnt - 1);

                for (const auto& it : inputs_to_redirect) {
                    for (const auto& in : it.second) {
                        auto squeeze = make_shared<Squeeze>(split->output(cnt - it.first - 1), axis_1);
                        in.replace_source_output(squeeze);
                    }
                }
            }
        }
        return true;
    };

    auto m = make_shared<Matcher>(cell, matcher_name);
    this->register_matcher(m, callback);
}
