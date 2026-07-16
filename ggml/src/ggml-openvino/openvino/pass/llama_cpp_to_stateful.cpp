// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llama_cpp_to_stateful.h"

#include <memory>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/frontend/gguf/set_rows_op.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/variable.hpp"

namespace ggml {
namespace pass {

namespace {

std::shared_ptr<ov::Node> find_by_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& n : model->get_ops()) {
        if (n->get_friendly_name() == name) {
            return n;
        }
    }
    return nullptr;
}

// The attention read of a KV cache is the chain
//   SetRows_output -> Reshape (merged [1,1,seq,emb] -> split [1,seq,H,D]) -> Slice -> Slice ->
//   Transpose -> ... -> SDPA
// The two Slices window the fixed preallocated cache by attention_size / seq_active; on an
// append-grown stateful cache they are no-ops that break stateful_sdpa_fusion. We keep the
// relayout Reshape and drop the Slices by reconnecting the Transpose directly to the Reshape.
// This finds that (Reshape, Transpose) pair downstream of a SetRows output, or {nullptr,nullptr}
// for a non-KV SetRows (e.g. MoE routing), which is left for the default stateless lowering.
struct ReadChain {
    std::shared_ptr<ov::op::v1::Reshape> relayout;
    std::shared_ptr<ov::op::v1::Transpose> transpose;
};

ReadChain find_read_chain(const std::shared_ptr<ov::Node>& set_rows) {
    for (const auto& t0 : set_rows->output(0).get_target_inputs()) {
        auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(t0.get_node()->shared_from_this());
        if (!reshape) {
            continue;
        }
        // Walk Reshape -> (Slice)* -> Transpose.
        std::shared_ptr<ov::Node> cur = reshape;
        for (int depth = 0; depth < 4 && cur; ++depth) {
            std::shared_ptr<ov::Node> next;
            for (const auto& t : cur->output(0).get_target_inputs()) {
                auto consumer = t.get_node()->shared_from_this();
                if (auto tr = ov::as_type_ptr<ov::op::v1::Transpose>(consumer)) {
                    return {reshape, tr};
                }
                if (ov::as_type_ptr<ov::op::v8::Slice>(consumer)) {
                    next = consumer;
                }
            }
            cur = next;
        }
    }
    return {nullptr, nullptr};
}

// Reslice the attention mask for the append-grown KV: slice over the last two axes to
// [token_len_per_seq, last_pos + 1] instead of the stateless fixed-attention_size slice.
void rewire_stateful_mask(const std::shared_ptr<ov::Model>& model, const std::string& mask_sliced_name) {
    auto sliced = find_by_name(model, mask_sliced_name);
    if (!sliced) {
        return;
    }
    auto mask = find_by_name(model, "self_kq_mask");
    auto token_len = find_by_name(model, "token_len_per_seq");
    auto inp_pos = find_by_name(model, "inp_pos");
    if (!mask || !token_len || !inp_pos) {
        return;
    }

    auto i64 = ov::element::i64;
    auto zero_2d = ov::op::v0::Constant::create(i64, {2}, {0, 0});
    auto one_2d = ov::op::v0::Constant::create(i64, {2}, {1, 1});
    auto three_1d = ov::op::v0::Constant::create(i64, {1}, {3});
    auto neg_one_1d = ov::op::v0::Constant::create(i64, {1}, {-1});
    auto axes = ov::op::v0::Constant::create(i64, {2}, {-2, -1});

    auto gather_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, neg_one_1d, three_1d);
    auto reshaped_pos =
        std::make_shared<ov::op::v1::Reshape>(gather_pos, ov::op::v0::Constant::create(i64, {1}, {1}), false);
    auto pos_inc =
        std::make_shared<ov::op::v1::Add>(reshaped_pos, ov::op::v0::Constant::create(ov::element::i32, {1}, {1}));
    auto stop = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{token_len, std::make_shared<ov::op::v1::ConvertLike>(pos_inc, token_len)}, 0);
    auto new_slice = std::make_shared<ov::op::v8::Slice>(mask, zero_2d, stop, one_2d, axes);
    auto new_mask = std::make_shared<ov::op::v0::Convert>(new_slice, ov::element::f16);
    new_mask->set_friendly_name(mask_sliced_name);
    ov::replace_node(sliced, new_mask);
}

}  // namespace

bool LlamaCppToStateful::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::shared_ptr<ov::frontend::gguf::SetRows>> kv_writes;
    std::vector<ReadChain> read_chains;

    for (const auto& node : model->get_ops()) {
        auto set_rows = ov::as_type_ptr<ov::frontend::gguf::SetRows>(node);
        if (!set_rows) {
            continue;
        }
        // Only KV-cache writes (dst is a Parameter, output read by attention) become stateful.
        // Non-KV SetRows (e.g. MoE) are left for the frontend's default stateless lowering.
        if (!ov::as_type_ptr<ov::op::v0::Parameter>(set_rows->input_value(2).get_node_shared_ptr())) {
            continue;
        }
        auto chain = find_read_chain(set_rows);
        if (!chain.transpose) {
            continue;
        }
        kv_writes.push_back(set_rows);
        read_chains.push_back(chain);
    }
    if (kv_writes.empty()) {
        return false;
    }

    std::vector<std::shared_ptr<ov::op::v0::Parameter>> params_to_remove;
    std::vector<std::shared_ptr<ov::op::v0::Result>> results_to_remove;
    ov::SinkVector new_sinks;

    for (size_t i = 0; i < kv_writes.size(); ++i) {
        auto set_rows = kv_writes[i];
        auto chain = read_chains[i];
        auto new_kv = set_rows->input_value(0);  // [1, 1, seq, emb]
        auto cache_param = ov::as_type_ptr<ov::op::v0::Parameter>(set_rows->input_value(2).get_node_shared_ptr());

        const auto& cache_name = cache_param->get_friendly_name();
        const auto& ps = cache_param->get_partial_shape();
        const auto et = cache_param->get_element_type();
        auto var = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape{ps[0], ps[1], ov::Dimension::dynamic(), ps[3]}, et, cache_name});

        // Empty-initialized state; the current step's K/V is appended along the sequence axis.
        auto init = ov::op::v0::Constant::create(et, ov::Shape{1, 1, 0, static_cast<size_t>(ps[3].get_length())}, {});
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(init, var);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{read_value, new_kv}, 2);
        concat->set_friendly_name(set_rows->get_friendly_name());
        new_sinks.push_back(std::make_shared<ov::op::v6::Assign>(concat, var));

        // Any remaining consumers of the SetRows output (e.g. the cache Result, and the read-chain
        // relayout Reshape) now read the appended Concat.
        for (auto target : set_rows->output(0).get_target_inputs()) {
            target.replace_source_output(concat);
        }
        // Bypass the stateless read windowing: feed the relayout Reshape (which produces the split
        // [1, seq, H, D] layout the read Transpose expects) directly into the Transpose, dropping
        // the attention_size/seq_active Slices. This yields the ReadValue->Concat->Reshape->
        // Transpose->SDPA shape the CPU plugin's stateful_sdpa_fusion matches.
        chain.transpose->input(0).replace_source_output(chain.relayout->output(0));

        params_to_remove.push_back(cache_param);
        for (const auto& r : model->get_results()) {
            if (r->get_friendly_name() == cache_name) {
                results_to_remove.push_back(r);
                break;
            }
        }
        model->add_variables({var});
    }

    for (const auto& r : results_to_remove) {
        model->remove_result(r);
    }
    model->add_sinks(new_sinks);
    for (const auto& p : params_to_remove) {
        model->remove_parameter(p);
    }

    rewire_stateful_mask(model, "KQ_mask_sliced");
    rewire_stateful_mask(model, "KQ_mask_swa_sliced");

    model->validate_nodes_and_infer_types();
    return true;
}

}  // namespace pass
}  // namespace ggml
