// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"
#include "openvino/pass/serialize.hpp"

#ifndef IN_OV_COMPONENT
#    define IN_OV_COMPONENT
#    define WAS_OV_LIBRARY_DEFINED
#endif

#include "gna/gna_config.hpp"
#include "gpu/gpu_config.hpp"

#ifdef WAS_OV_LIBRARY_DEFINED
#    undef IN_OV_COMPONENT
#    undef WAS_OV_LIBRARY_DEFINED
#endif

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/slog.hpp"

#include "benchmark_app.hpp"
#include "infer_request_wrap.hpp"
#include "inputs_filling.hpp"
#include "remote_tensors_filling.hpp"
#include "statistics_report.hpp"
#include "utils.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/concat.hpp"

// clang-format on

/**
 * @brief The entry point of the benchmark application
 */
 

std::shared_ptr<ov::Model> get_query_model() {
    auto const_to_concat = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 2}, 0);

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 2});
    //auto mem_i = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 2}, 0);
    auto mem_i = std::make_shared<ov::op::v3::ShapeOf>(input);
    auto const_0 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 0);
    auto const_1 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 1);
    auto random = std::make_shared<ov::op::v8::RandomUniform>(mem_i, const_0, const_1, ov::element::f32);
    auto mem_r = std::make_shared<ov::op::v3::ReadValue>(random, "id");
    auto mul   = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{mem_r, const_to_concat}, 0);
    auto mem_w = std::make_shared<ov::op::v3::Assign>(mul, "id");
    auto res = std::make_shared<ov::op::v0::Result>(mul);
    return std::make_shared<ov::Model>(ov::ResultVector{res}, ov::SinkVector{mem_w}, ov::ParameterVector{input});
}

int main() {
    auto core = ov::Core();
    //auto model = core.read_model("path_to_the_model");
    auto model = get_query_model();
    std::cout << "Step 1 XXXX" << std::endl;
    serialize(model, "/home/itikhonov/OpenVINO/tmp/serialized/assign_read_concat.xml",
              "/home/itikhonov/OpenVINO/tmp/serialized/assign_read_concat.bin");
    auto compiled = core.compile_model(model, "Template");
    std::cout << "Step 2 XXXX" << std::endl;
    auto infer_request = compiled.create_infer_request();
    auto query_state = infer_request.query_state();
}
