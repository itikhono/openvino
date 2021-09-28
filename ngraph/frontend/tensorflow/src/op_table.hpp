// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <default_opset.h>

#include <functional>
#include <map>
#include <ngraph/output_vector.hpp>
#include <string>

#include "ngraph_conversions.h"
#include "node_context.hpp"
#include "utils.h"

namespace tensorflow {
namespace ngraph_bridge {
using OutPortName = std::string;
using NamedOutputs = ngraph::OutputVector;
using CreatorFunction = std::function<NamedOutputs(const ngraph::frontend::tf::NodeContext&)>;

const std::map<const std::string, const CreatorFunction> get_supported_ops();

}  // namespace ngraph_bridge
}  // namespace tensorflow
