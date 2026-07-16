// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ggml {
namespace pass {

// Stateful lowering of the gguf frontend's SetRows placeholder op. The frontend emits a SetRows
// for every ggml SET_ROWS and by default lowers it to the stateless ScatterUpdate form. When
// stateful execution is requested, the backend registers this pass (via
// ov::frontend::DecoderTransformationExtension) so it runs in the frontend's normalization stage
// *before* the default stateless lowering. It converts only the SetRows that are attention KV
// writes -- destination is a Parameter whose updated value is read by attention (via the
// Reshape -> Slice -> Transpose chain) -- into an OpenVINO stateful subgraph:
//   SetRows(new_kv, idx, cache_Param) -> ReadValue(var) -> Concat(ReadValue, new_kv, axis=2)
//                                        -> {cache read, Assign(var)}
// leaving non-KV SetRows (e.g. MoE routing) for the default stateless lowering. The cache
// Parameter/Result are removed; the read path is reconnected past the stateless windowing Slices
// so stateful_sdpa_fusion matches; and the attention mask is resliced to the grown KV length.
class LlamaCppToStateful : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ggml::pass::LlamaCppToStateful")
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace ggml
