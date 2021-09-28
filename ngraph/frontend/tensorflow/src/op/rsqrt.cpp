// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

#if 0

namespace tensorflow {
namespace ngraph_bridge {

OutputVector TranslateRsqrtOp(
        const NodeContext& node) {
    return TranslateUnaryOp(
            op, static_input_map, ng_op_map, [&op](Output<Node> n) {
                // Create a constant tensor populated with the value -1/2.
                // (1/sqrt(x) = x^(-1/2))
                auto et = n.get_element_type();
                auto shape = n.get_shape();
                std::vector<std::string> constant_values(shape_size(shape), "-0.5");
                auto ng_exponent = ConstructNgNode<opset::Constant>(
                        node.get_name(), et, shape, constant_values);

                // Raise each element of the input to the power -0.5.
                return ConstructNgNode<opset::Power>(node.get_name(), n, ng_exponent);
            });
}
}
}

#endif