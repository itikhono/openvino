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

#include <frontend_manager/frontend_manager.hpp>
#include <tensorflow_frontend/frontend.hpp>

extern "C" NGRAPH_HELPER_DLL_EXPORT ngraph::frontend::FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

extern "C" NGRAPH_HELPER_DLL_EXPORT void* GetFrontEndData() {
    auto res = new ngraph::frontend::FrontEndPluginInfo();
    res->m_name = "tf";
    res->m_creator = []() {
        return std::make_shared<ngraph::frontend::FrontEndTF>();
    };
    return res;
}
