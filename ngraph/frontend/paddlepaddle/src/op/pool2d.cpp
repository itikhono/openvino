//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <ngraph/opsets/opset6.hpp>
#include "pool2d.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

// helper func - get pad_begin and pad_end
static void get_paddings(const NodeContext& node, ngraph::Shape& pad_begin, ngraph::Shape& pad_end, ngraph::op::PadType &auto_pad) {
    //
    auto pad_algo = node.get_attribute<std::string>("padding_algorithm");
    if (pad_algo == "SAME") {
        auto_pad = ngraph::op::PadType::SAME_UPPER;
    } else if (pad_algo == "VALID") {
        auto_pad = ngraph::op::PadType::VALID;
    } else if ((pad_algo == "EXPLICIT") ||
             pad_algo.empty()) { //adaptive_maxpool with no such attr.
        auto_pad = ngraph::op::PadType::EXPLICIT;
    }else {
        // FIXME
        throw std::runtime_error("Unsupported pooling padding_algorithm " + pad_algo);
    }

    /*If pool padding size is a tuple or list, it could be in three forms: 
    [pad_height, pad_width] or [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right], 
    and when data_format is “NCHW”, pool_padding can be in the form [[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]. 
    when data_format is “NHWC”, pool_padding can be in the form [[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]. 
    Otherwise, the pool padding size will be a square of an int.*/
    auto paddings = node.get_attribute<std::vector<int32_t>>("paddings");
    auto data_format = node.get_attribute<std::string>("data_format");  

    switch (paddings.size())
    {
    case 1:
        pad_begin = Shape(2, paddings[0]);
        pad_end = pad_begin;
        break;
    case 2:
        pad_begin = Shape{static_cast<uint64_t>(paddings[0]), static_cast<uint64_t>(paddings[1])};
        pad_end = pad_begin;
        break;
    case 4:
        pad_begin = Shape{static_cast<uint64_t>(paddings[0]), static_cast<uint64_t>(paddings[2])};
        pad_end = Shape(static_cast<uint64_t>(paddings[1]), static_cast<uint64_t>(paddings[3]));
        break;
    case 8:     
        if (data_format == "NCHW") {
            pad_begin = Shape{static_cast<uint64_t>(paddings[4]), static_cast<uint64_t>(paddings[6])};
            pad_end = Shape(static_cast<uint64_t>(paddings[5]), static_cast<uint64_t>(paddings[7]));
        } else if (data_format == "NHWC") {
            pad_begin = Shape{static_cast<uint64_t>(paddings[2]), static_cast<uint64_t>(paddings[4])};
            pad_end = Shape(static_cast<uint64_t>(paddings[3]), static_cast<uint64_t>(paddings[5]));             
        } else {
            throw std::runtime_error("Unsupported pooling data_format " + data_format);
        }
        break;
    default:
        throw std::runtime_error("Unsupported pooling paddings " + paddings.size());
        break;
    }  
}
  

NamedOutputs pool2d (const NodeContext& node) {
    // TODO : resolve padding according to spec
    auto data = node.get_ng_input("X"); 

    auto pooling_type = node.get_attribute<std::string>("pooling_type");
    auto global_pooling = node.get_attribute<bool>("global_pooling");
    auto adaptive = node.get_attribute<bool>("adaptive");
    auto kernel_shape = node.get_attribute<std::vector<int32_t>>("ksize"); // FIXME: int/list?
    
    auto rounding_type = node.get_attribute<bool>("ceil_mode")
                                ? ngraph::op::RoundingType::CEIL
                                : ngraph::op::RoundingType::FLOOR;

    if (pooling_type.empty()) { // TODO: to check op.type "max_pool2d_with_index"
        pooling_type = "max";
    }            

    PDPD_ASSERT((pooling_type == "max") || (pooling_type == "avg"),
                    "pool2d: not supported pooling type !");
    PDPD_ASSERT(kernel_shape.size()==1 || kernel_shape.size()==2,
                    "pool2d: ksize must be 1 or 2!");                 

    PartialShape input_shape = data.get_partial_shape();
    PDPD_ASSERT(input_shape.rank().is_static(), "pool2d: X rank must be static!");
    int32_t input_rank = input_shape.rank().get_length();  
    uint64_t input_h = input_shape[input_rank-2].get_length();
    uint64_t input_w =  input_shape[input_rank-1].get_length();
    std::cout << input_rank << "," << input_h << "," << input_w << std::endl;
   
    auto auto_pad = ngraph::op::PadType::EXPLICIT;
    ngraph::Shape pad_begin, pad_end;
    get_paddings(node, pad_begin, pad_end, auto_pad);

    if (global_pooling || (adaptive && 
                        std::any_of(kernel_shape.begin(), 
                                    kernel_shape.end(), 
                                    [](int32_t i){return i==1;}))) { 
        if (pooling_type == "max") {
            return node.default_single_output_mapping({std::make_shared<ngraph::opset6::MaxPool>(
                        data,
                        ngraph::Strides({1,1}),
                        ngraph::Shape{0,0}, //FIXME pads_begin
                        ngraph::Shape{0,0}, //pads_end
                        ngraph::Shape{input_h,input_w})}, {"Out"});
        } else {
            // TODO : resolve axes according to rank
            auto axes = ngraph::opset6::Constant::create(ngraph::element::i64, {2}, {input_rank-2, input_rank-1});
            return node.default_single_output_mapping({std::make_shared<ngraph::opset6::ReduceMean>(data, axes, true)}, {"Out"});
        }                    
    } else if (adaptive) {        
        uint64_t pool_size_Height, pool_size_Width;
        if (kernel_shape.size()==1) {
            pool_size_Height = pool_size_Width = kernel_shape[0];
        } else {
            pool_size_Height = kernel_shape[0];
            pool_size_Width = kernel_shape[1];
        }

        uint64_t stride_h = int64_t(input_h / pool_size_Height);
        uint64_t stride_w = int64_t(input_h / pool_size_Width);
        uint64_t kernel_h = input_h - (pool_size_Height - 1) * stride_h;
        uint64_t kernel_w = input_w - (pool_size_Width - 1) * stride_w;

        if ( stride_h < 1 ||  stride_w < 1) { // upsampling?
            throw std::runtime_error("Unsupported pooling adaptive type!");
        }
 
        if (pooling_type == "max") {
            return node.default_single_output_mapping({std::make_shared<ngraph::opset6::MaxPool>(
                        data,
                        ngraph::Strides{stride_h, stride_w},
                        pad_begin, pad_end,
                        ngraph::Shape{kernel_h, kernel_w},
                        rounding_type,
                        auto_pad)}, {"Out"});
        } else {
            bool exclude_pad = node.get_attribute<bool>("exclusive") ? true : false;
            return node.default_single_output_mapping({std::make_shared<ngraph::opset6::AvgPool>(
                        data,
                        ngraph::Strides{stride_h, stride_w},
                        pad_begin, pad_end,
                        ngraph::Shape{kernel_h, kernel_w},
                        exclude_pad,
                        rounding_type,
                        auto_pad)}, {"Out"});
        }          
    } else {
        auto strides = node.get_attribute<std::vector<int32_t>>("strides");
        auto paddings = node.get_attribute<std::vector<int32_t>>("paddings");
        uint64_t kernel_h, kernel_w;
        if (kernel_shape.size() == 1) {
            kernel_h = kernel_w = kernel_shape[0];
        } else {
            kernel_h = kernel_shape[0];
            kernel_w = kernel_shape[1];
        }

        if ((input_h > 0) && 
            (input_h+paddings[0] < kernel_h)) {
                kernel_h = input_h+paddings[0];
        }
        if ((input_w > 0) && 
            (input_w+paddings[1] < kernel_w)) {
                kernel_w = input_w+paddings[1];
        }        

        if (pooling_type == "max") {
            return node.default_single_output_mapping({std::make_shared<ngraph::opset6::MaxPool>(
                        data,
                        ngraph::Strides(strides.begin(), strides.end()),
                        pad_begin, pad_end,
                        ngraph::Shape{kernel_h, kernel_w},
                        rounding_type,
                        auto_pad)}, {"Out"});
        } else {
            bool exclude_pad = node.get_attribute<bool>("exclusive") ? true : false;
            return node.default_single_output_mapping({std::make_shared<ngraph::opset6::AvgPool>(
                        data,
                        ngraph::Strides(strides.begin(), strides.end()),
                        pad_begin, pad_end,
                        ngraph::Shape{kernel_h, kernel_w},
                        exclude_pad,
                        rounding_type,
                        auto_pad)}, {"Out"});
        }                                  
    }
}

}}}}