// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

// Translate Conv3D Op
namespace tensorflow {
namespace ngraph_bridge {

OutputVector TranslateConv3DOp(const NodeContext& node) {
    auto ng_input = node.get_ng_input(0), ng_filter = node.get_ng_input(1);

    auto tf_strides = node.get_attribute<std::vector<int32_t>>("strides");
    auto tf_dilations = node.get_attribute<std::vector<int32_t>>("dilations");
    auto tf_padding_type = node.get_attribute<std::string>("padding");
    auto tf_data_format = node.get_attribute<std::string>("data_format");

    if (tf_data_format != "NDHWC" && tf_data_format != "NCDHW") {
        throw errors::InvalidArgument("Conv3D data format is neither NDHWC nor NCDHW");
    }

    bool is_ndhwc = (tf_data_format == "NDHWC");

    // TODO: in 3D
    // TF Kernel Test Checks
    // // Strides in the batch and depth dimension is not supported
    // if (tf_strides[0] != 1 || tf_strides[is_nhwc ? 3 : 1] != 1) {
    //   return errors::InvalidArgument(
    //       "Strides in batch and depth dimensions is not supported: ",
    //       op->type_string());
    // }

    NGRAPH_VLOG(3) << join(tf_strides);
    NGRAPH_VLOG(3) << join(tf_dilations);
    NGRAPH_VLOG(3) << tf_padding_type;
    NGRAPH_VLOG(3) << tf_data_format;

    Strides ng_strides(3);
    Strides ng_dilations(3);
    Shape ng_image_shape(3);
    Shape ng_kernel_shape(3);

    NHWCtoHW(is_ndhwc, tf_strides, ng_strides);
    NHWCtoHW(is_ndhwc, ng_input.get_shape(), ng_image_shape);
    NHWCtoHW(is_ndhwc, tf_dilations, ng_dilations);
    NHWCtoNCHW(node.get_name(), is_ndhwc, ng_input);

    NGRAPH_VLOG(3) << "ng_strides: " << join(ng_strides);
    NGRAPH_VLOG(3) << "ng_dilations: " << join(ng_dilations);
    NGRAPH_VLOG(3) << "ng_image_shape: " << join(ng_image_shape);

    auto& ng_filter_shape = ng_filter.get_shape();
    ng_kernel_shape[0] = ng_filter_shape[0];
    ng_kernel_shape[1] = ng_filter_shape[1];
    ng_kernel_shape[2] = ng_filter_shape[2];
    Transpose3D<4, 3, 0, 1, 2>(ng_filter);
    SetTracingInfo(node.get_name(), ng_filter);

    NGRAPH_VLOG(3) << "ng_kernel_shape: " << join(ng_kernel_shape);

    CoordinateDiff ng_padding_below;
    CoordinateDiff ng_padding_above;
    MakePadding(tf_padding_type,
                ng_image_shape,
                ng_kernel_shape,
                ng_strides,
                ng_dilations,
                ng_padding_below,
                ng_padding_above);

    Output<Node> ng_conv = ConstructNgNode<opset::Convolution>(node.get_name(),
                                                               ng_input,
                                                               ng_filter,
                                                               ng_strides,
                                                               ng_padding_below,
                                                               ng_padding_above,
                                                               ng_dilations);

    NCHWtoNHWC(node.get_name(), is_ndhwc, ng_conv);
    return {ng_conv};
}
}  // namespace ngraph_bridge
}  // namespace tensorflow