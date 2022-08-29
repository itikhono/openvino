// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/sequence_fusion.hpp"

#include <memory>
#include <openvino/opsets/opset9.hpp>

#include "itt.hpp"
#include "ngraph_ops/augru_cell.hpp"
#include "ngraph_ops/augru_sequence.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset9;
using namespace ov::pass::pattern;
using namespace ov::op::util;

namespace {
bool check_WRB(const shared_ptr<RNNCellBase>& cell_1, const shared_ptr<RNNCellBase>& cell_2) {
    int64_t idx_W = 2, idx_R = 3, idx_B = 4;
    auto lstm_cell_1 = dynamic_pointer_cast<LSTMCell>(cell_1);
    auto lstm_cell_2 = dynamic_pointer_cast<LSTMCell>(cell_2);

    // 2nd input is Cell State
    if (lstm_cell_1 && lstm_cell_2) {
        idx_B++;
        idx_R++;
        idx_W++;
    }

    return cell_1->input_value(idx_W).get_node_shared_ptr().get() ==
               cell_2->input_value(idx_W).get_node_shared_ptr().get() &&
           cell_1->input_value(idx_R).get_node_shared_ptr().get() ==
               cell_2->input_value(idx_R).get_node_shared_ptr().get() &&
           cell_1->input_value(idx_B).get_node_shared_ptr().get() ==
               cell_2->input_value(idx_B).get_node_shared_ptr().get();
}

bool is_equal_cells(const shared_ptr<RNNCellBase>& cell_1, const shared_ptr<RNNCellBase>& cell_2) {
    bool is_equal =
        cell_1->get_type_name() == cell_2->get_type_name() && cell_1->get_hidden_size() == cell_2->get_hidden_size() &&
        cell_1->get_activations() == cell_2->get_activations() &&
        cell_1->get_activations_alpha() == cell_2->get_activations_alpha() &&
        cell_1->get_activations_beta() == cell_2->get_activations_beta() && cell_1->get_clip() == cell_2->get_clip();
    is_equal &= check_WRB(cell_1, cell_2);
    auto gru_cell_1 = dynamic_pointer_cast<GRUCell>(cell_1);
    auto gru_cell_2 = dynamic_pointer_cast<GRUCell>(cell_2);
    if (gru_cell_1 && gru_cell_2) {
        is_equal &= gru_cell_1->get_linear_before_reset() == gru_cell_2->get_linear_before_reset();
    }
    return is_equal;
}

void find_cell_chain(const shared_ptr<RNNCellBase>& current_cell,
                     OutputVector& x_to_concat,
                     OutputVector& attention_to_concat,
                     map<int, set<ov::Input<Node>>>& h_inputs_to_redirect,
                     map<int, set<ov::Input<Node>>>& c_inputs_to_redirect,
                     int& cells_cnt) {
    cells_cnt++;
    auto axis_1 = make_shared<Constant>(element::i64, Shape{}, 1);

    shared_ptr<RNNCellBase> current = current_cell;
    while (true) {
        // check the source node of HiddenState input
        auto prev = current->input_value(1).get_node_shared_ptr();
        if (auto prev_cell = dynamic_pointer_cast<RNNCellBase>(prev)) {
            if (is_equal_cells(prev_cell, current)) {
                break;
            }

            auto in_X = current->input(0);
            x_to_concat.push_back(make_shared<Unsqueeze>(in_X.get_source_output(), axis_1));

            // collect inputs (target_inputs) connected to H output of prev_node except H input of the current node
            auto in_H = current->input(1);
            for (const auto& input : prev_cell->get_output_target_inputs(0)) {
                if (input != in_H) {
                    h_inputs_to_redirect[cells_cnt].insert(input);
                }
            }

            if (auto lstm = dynamic_pointer_cast<LSTMCell>(current)) {
                auto in_C = current->input(2);
                // collect inputs (target_inputs) connected to C output of prev_node except C input of the current node
                for (const auto& input : prev_cell->get_output_target_inputs(1)) {
                    if (input != in_C) {
                        c_inputs_to_redirect[cells_cnt].insert(input);
                    }
                }
            }

            if (auto augru = dynamic_pointer_cast<ov::op::internal::AUGRUCell>(prev_cell)) {
                attention_to_concat.push_back(make_shared<Unsqueeze>(augru->input_value(5), axis_1));
            }

            current = prev_cell;
            cells_cnt++;
        } else {
            break;
        }
    }
    reverse(x_to_concat.begin(), x_to_concat.end());
    reverse(attention_to_concat.begin(), attention_to_concat.end());
}

shared_ptr<Node> create_sequence(const shared_ptr<RNNCellBase>& cell,
                                 const OutputVector& x_to_concat,
                                 const OutputVector& attention_to_concat,
                                 const map<int, set<ov::Input<Node>>>& h_inputs_to_redirect,
                                 const map<int, set<ov::Input<Node>>>& c_inputs_to_redirect,
                                 int cells_cnt) {
    int64_t idx_W = 2, idx_R = 3, idx_B = 4;
    auto lstm_cell_1 = dynamic_pointer_cast<LSTMCell>(cell);
    // 2nd input is Cell State
    if (lstm_cell_1) {
        idx_B++;
        idx_R++;
        idx_W++;
    }

    auto axis_0 = make_shared<Constant>(element::i64, Shape{}, 0);
    auto axis_1 = make_shared<Constant>(element::i64, Shape{}, 1);

    const auto X_in = make_shared<Concat>(x_to_concat, 1);
    const auto Ht_in = make_shared<Unsqueeze>(cell->input_value(1), axis_1);
    const auto W_in = make_shared<Unsqueeze>(cell->input_value(idx_W), axis_0);
    const auto R_in = make_shared<Unsqueeze>(cell->input_value(idx_R), axis_0);
    const auto B_in = make_shared<Unsqueeze>(cell->input_value(idx_B), axis_0);

    const auto& shape_node = ngraph::op::util::make_try_fold<ShapeOf>(cell->input_value(0));
    const auto& batch_dimension =
        ngraph::op::util::make_try_fold<Gather>(shape_node, Constant::create(ov::element::i64, {1}, {0}), axis_0);
    auto seq_lengths_scalar = Constant::create(ov::element::i64, {}, {cells_cnt});
    auto sequence_lengths_in = ngraph::op::util::make_try_fold<Broadcast>(seq_lengths_scalar, batch_dimension);

    shared_ptr<Node> sequence;
    if (dynamic_pointer_cast<LSTMCell>(cell)) {
        cout << "XXXXXXXX LSTMSequence pattern detected" << endl;
        const auto Ct_in = make_shared<Unsqueeze>(cell->input_value(2), axis_1);
        sequence = make_shared<LSTMSequence>(X_in,
                                             Ht_in,
                                             Ct_in,
                                             sequence_lengths_in,
                                             W_in,
                                             R_in,
                                             B_in,
                                             cell->get_hidden_size(),
                                             ov::op::RecurrentSequenceDirection::FORWARD,
                                             cell->get_activations_alpha(),
                                             cell->get_activations_beta(),
                                             cell->get_activations(),
                                             cell->get_clip());
        if (!c_inputs_to_redirect.empty()) {
            auto squeeze_Y = make_shared<Squeeze>(sequence->output(0), axis_1);
            auto split = make_shared<Split>(squeeze_Y, axis_1, cells_cnt - 1);

            for (const auto& it : c_inputs_to_redirect) {
                for (const auto& in : it.second) {
                    auto squeeze = make_shared<Squeeze>(split->output(cells_cnt - it.first - 1), axis_1);
                    in.replace_source_output(squeeze);
                }
            }
        }
        auto squeeze_C = make_shared<Squeeze>(sequence->output(1), axis_1);
        replace_node(cell, squeeze_C);
    } else if (auto gru_cell = dynamic_pointer_cast<GRUCell>(cell)) {
        cout << "XXXXXXXX GRUSequence pattern detected" << endl;
        sequence = make_shared<GRUSequence>(X_in,
                                            Ht_in,
                                            sequence_lengths_in,
                                            W_in,
                                            R_in,
                                            B_in,
                                            cell->get_hidden_size(),
                                            ov::op::RecurrentSequenceDirection::FORWARD,
                                            cell->get_activations(),
                                            cell->get_activations_alpha(),
                                            cell->get_activations_beta(),
                                            cell->get_clip(),
                                            gru_cell->get_linear_before_reset());
    } else if (dynamic_pointer_cast<RNNCell>(cell)) {
        cout << "XXXXXXXX RNNSequence pattern detected" << endl;
        sequence = make_shared<RNNSequence>(X_in,
                                            Ht_in,
                                            sequence_lengths_in,
                                            W_in,
                                            R_in,
                                            B_in,
                                            cell->get_hidden_size(),
                                            ov::op::RecurrentSequenceDirection::FORWARD,
                                            cell->get_activations(),
                                            cell->get_activations_alpha(),
                                            cell->get_activations_beta(),
                                            cell->get_clip());
    } else if (dynamic_pointer_cast<ov::op::internal::AUGRUCell>(cell)) {
        cout << "XXXXXXXX AUGRUSequence pattern detected" << endl;
        const auto A_in = make_shared<Concat>(attention_to_concat, 1);
        const auto gru_sequence = make_shared<ov::op::internal::AUGRUSequence>(X_in,
                                                                               Ht_in,
                                                                               sequence_lengths_in,
                                                                               W_in,
                                                                               R_in,
                                                                               B_in,
                                                                               A_in,
                                                                               cell->get_hidden_size());
    } else {
        // cell is not supported;
        return nullptr;
    }

    auto squeeze_H = make_shared<Squeeze>(sequence->output(1), axis_1);
    replace_node(cell, squeeze_H);
    cout << "XXXXXXXX Sequence pattern replaced" << endl;

    if (!h_inputs_to_redirect.empty()) {
        auto squeeze_Y = make_shared<Squeeze>(sequence->output(0), axis_1);
        auto split = make_shared<Split>(squeeze_Y, axis_1, cells_cnt - 1);

        for (const auto& it : h_inputs_to_redirect) {
            for (const auto& in : it.second) {
                auto squeeze = make_shared<Squeeze>(split->output(cells_cnt - it.first - 1), axis_1);
                in.replace_source_output(squeeze);
            }
        }
    }
    return squeeze_H;
}
}  // namespace

ov::pass::SequenceFusion::SequenceFusion() {
    MATCHER_SCOPE(SequenceFusion);

    auto cell = wrap_type<RNNCellBase>();
    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto cell = m.get_match_root();
        shared_ptr<RNNCellBase> current_cell = dynamic_pointer_cast<RNNCellBase>(cell);
        if (!current_cell) {
            return false;
        }

        // check that this is the last Cell in the chain
        // GRUCell -> GRUCell (the last cell) -> OtherNode
        // GRUCell (hidden_size = 128) -> GRUCell (hs = 128, the last) -> GRUCell (hs = 64)
        for (const auto& target : cell->get_output_target_inputs(0)) {
            auto cell_1 = dynamic_pointer_cast<RNNCellBase>(target.get_node()->shared_from_this());
            if (cell_1 && is_equal_cells(cell_1, current_cell)) {
                return false;
            }
        }

        int cells_cnt = 0;
        OutputVector x_to_concat;
        OutputVector attention_to_concat;
        map<int, set<ov::Input<Node>>> h_inputs_to_redirect;
        map<int, set<ov::Input<Node>>> c_inputs_to_redirect;

        // detect chain (Cell->Cell->Cell->..)
        find_cell_chain(current_cell,
                        x_to_concat,
                        attention_to_concat,
                        h_inputs_to_redirect,
                        c_inputs_to_redirect,
                        cells_cnt);

        // no reasons to create sequence if the single cell detected.
        // investigate optimal cnt of cells
        int optimal_cnt_of_cells = 2;
        if (cells_cnt < optimal_cnt_of_cells)
            return false;

        auto node = create_sequence(current_cell,
                                    x_to_concat,
                                    attention_to_concat,
                                    h_inputs_to_redirect,
                                    c_inputs_to_redirect,
                                    cells_cnt);
        if (node == nullptr) {
            return false;
        }

        return true;
    };

    auto m = make_shared<Matcher>(cell, matcher_name);
    this->register_matcher(m, callback);
}
