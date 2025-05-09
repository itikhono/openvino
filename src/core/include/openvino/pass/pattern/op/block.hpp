
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include <vector>

#include "openvino/pass/pattern/op/pattern.hpp"

namespace ov::pass::pattern::op {

class OPENVINO_API Block : public Pattern {
public:
    OPENVINO_OP("PatternBlock");

    Block(const OutputVector& inputs, const OutputVector& outputs, const std::string& name = "");

    const OutputVector& get_inputs() const;
    const OutputVector& get_outputs() const;
    const std::string& get_name() const;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector&) const override;

private:
    OutputVector m_inputs;
    OutputVector m_outputs;
    std::string m_name;
};

}  // namespace ov::pass::pattern::op