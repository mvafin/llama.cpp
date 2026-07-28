// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <set>
#include <string>

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
//
// The sliding-window caches of an iSWA model (gemma3/gemma4) are excluded via skip_caches: see the
// comment on the constructor.
class LlamaCppToStateful : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ggml::pass::LlamaCppToStateful")

    LlamaCppToStateful() = default;

    // skip_caches: cache Parameter names to leave on the stateless path. An append-grown state is
    // only equivalent to ggml's cache while that cache is filled in position order, which is not
    // true of a sliding-window cache: once the sequence passes n_swa, llama.cpp prunes out-of-window
    // cells and reuses their slots, so mask column j denotes a *slot*, not position j, and the mask
    // stops growing (its width saturates at the window). Appending would then desynchronize the KV
    // length from the mask width. Those layers stay stateless, where the ScatterUpdate writes land
    // at exactly the slots ggml chose.
    explicit LlamaCppToStateful(std::set<std::string> skip_caches) : m_skip_caches(std::move(skip_caches)) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::set<std::string> m_skip_caches;
};

}  // namespace pass
}  // namespace ggml
