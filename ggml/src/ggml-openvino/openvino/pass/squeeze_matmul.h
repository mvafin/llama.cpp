// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ggml {
namespace pass {

// For quantized models on NPU, NPUW expects the activation to be 3d in DQ (DynamicQuantization)
// optimization (e.g. DQMatMulGQ2i). This pass squeezes a leading unit batch dim off a 4d
// activation feeding a MatMul against 2d weights. Applied by the backend on the static (NPU)
// path after frontend conversion; it is a runtime/device concern, not part of the frontend.
class SqueezeMatmul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ggml::pass::SqueezeMatmul")
    SqueezeMatmul();
};

}  // namespace pass
}  // namespace ggml
