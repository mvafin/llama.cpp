#include "utils.h"

#include "ggml-impl.h"
#include "ggml-openvino-extra.h"
#include "ggml-openvino/ggml-decoder.h"
#include "ggml.h"
#include "model-cache.h"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/pass/fuse_to_conv.h"
#include "openvino/pass/llama_cpp_to_stateful.h"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/squeeze_matmul.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <map>
#include <iostream>
#include <memory>
#include <openvino/core/any.hpp>
#include <openvino/core/graph_util.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/tensor.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Suppress  deprecation warning for ov::Tensor::data()
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

// Debug seam (defined below): element-order full dump for element-wise cross-backend diff.
void maybe_dump_full_tensor(const std::string & name, const ov::Tensor & tensor);

static std::optional<ov::Tensor> try_make_kv_sliced_tensor(const std::shared_ptr<GgmlOvDecoder> & decoder,
                                                           const std::string & name,
                                                           const ggml_tensor * tensor) {
    if (ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE") || decoder->m_is_static ||
        decoder->m_is_stateful || tensor->op != GGML_OP_NONE || tensor->view_src != nullptr) {
        return std::nullopt;
    }
    const auto * op = decoder->get_tensor_used_op(tensor);
    if (!GgmlOvDecoder::is_kvcache(tensor, op)) {
        return std::nullopt;
    }
    const auto compute = decoder->get_compute_params();
    if (compute.n_seq_active != 1 || compute.seq_active_start != 0) {
        return std::nullopt;
    }
    int layer;
    try {
        layer = extract_layer_from_name(name);
    } catch (...) {
        return std::nullopt;
    }
    if (decoder->is_swa_layer(layer)) {
        return std::nullopt;
    }
    const int ctx_per_seq = decoder->get_model_params().ctx_per_seq;
    const int n_kv = compute.attention_size;
    auto shape = decoder->get_shape(tensor);
    if (ctx_per_seq <= 0 || n_kv <= 0 || n_kv >= ctx_per_seq || shape.size() != 4 || shape[0] != 1 ||
        shape[1] != 1 || static_cast<int>(shape[2]) != ctx_per_seq) {
        return std::nullopt;
    }
    shape[2] = static_cast<size_t>(n_kv);
    return ov::Tensor(decoder->get_ov_type(tensor), shape, tensor->data);
}

static uint64_t ggml_openvino_model_cache_extra_cfg(const std::string & device, bool stateful) {
    const char * manual_gqa_env = ggml_openvino_getenv_str("GGML_OPENVINO_MANUAL_GQA_ATTN");
    const bool manual_gqa = manual_gqa_env ? ggml_openvino_getenv_int("GGML_OPENVINO_MANUAL_GQA_ATTN") > 0
                                           : device == "GPU";
    uint64_t value = stateful ? 1u : 0u;
    value = value * 131 + (ggml_openvino_reduce_compile_mem_enabled() ? 1u : 0u);
    value = value * 131 + (ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_KV_SLICE") ? 1u : 0u);
    value = value * 131 + (manual_gqa ? 1u : 0u);
    return value;
}

enum ggml_status ov_graph_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    try {
        if (getenv("GGML_OPENVINO_DUMP_CGRAPH")) {
            static int dump_counter = 0;
            std::string filename = "cgraph_ov_" + std::to_string(dump_counter++) + ".txt";
            GGML_LOG_INFO("OV subgraph #%d: n_nodes=%d -> %s\n", dump_counter - 1, cgraph->n_nodes, filename.c_str());
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        const auto is_static = ggml_openvino_is_npu() || ggml_openvino_getenv_int("GGML_OPENVINO_FORCE_STATIC");

        GGML_ASSERT(ctx->runtime_context != nullptr);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);

        return is_static ? ov_graph_compute_static(cgraph, r_ctx) : ov_graph_compute_dynamic(cgraph, r_ctx);
    } catch (const ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        return GGML_STATUS_FAILED;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        return GGML_STATUS_FAILED;
    } catch (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        return GGML_STATUS_FAILED;
    }
}

ov::Tensor create_ov_output_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                   std::shared_ptr<ov::InferRequest> infer_request,
                                   int output_index,
                                   const ggml_tensor * ggml_tensor) {
    if (auto sliced = try_make_kv_sliced_tensor(ggml_decoder, ggml_tensor->name, ggml_tensor)) {
        return *sliced;
    }
    auto output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    void * output_data = ggml_tensor->data;
    if (ggml_decoder->m_is_static) {
        output_shape = infer_request->get_output_tensor(output_index).get_shape();
    } else {
        if (ggml_tensor->op == GGML_OP_CPY && ggml_tensor->view_src != nullptr &&
            ggml_nbytes(ggml_tensor) != ggml_nbytes(ggml_tensor->view_src)) {
            output_shape = ggml_decoder->get_shape(ggml_tensor->view_src);
            output_data = ggml_tensor->view_src->data;
        } else {
            output_shape = ggml_decoder->get_shape(ggml_tensor);
        }
        // A no-op in-place cache write produces a statically 0-element OV Result, but get_shape()
        // returns the view_src's full (non-empty) cache shape. qwen3-next's recurrent-state reorder
        // (inp->s_copy: GET_ROWS active seqs -> CPY into cache_r view) has 0 active sequences during
        // single-sequence generation, so its CPY output is [.,.,0,.]. Binding the full cache shape
        // makes set_output_tensor reject the size-0-vs-N mismatch. Bind the 0-element port shape
        // instead: the copy writes nothing, which is exactly ggml's no-op semantics.
        const auto & port_ps = infer_request->get_compiled_model().output(output_index).get_partial_shape();
        if (port_ps.is_static() && ov::shape_size(port_ps.to_shape()) == 0) {
            output_shape = port_ps.to_shape();
        }
    }

    ov::Tensor output_tensor(output_type, output_shape, output_data);
    return output_tensor;
}

namespace {
// graph_key (n_nodes + first/last node name) is too coarse to distinguish graphs that share that
// shape but differ in every tensor's type/size -- e.g. test-backend-ops's ADD_ID sweep, which
// builds a fresh {a, b, ids, (view_of_ids), out} graph per {type_a, type_b, n_embd, n_experts,
// n_experts_used, n_token} combination, all with identical node count and "out" as both first and
// last node name. can_reuse_dynamically() only compares RoPE params (irrelevant to non-LLM
// graphs), so those combinations collide on the same cache entry and the compiled model gets fed
// a tensor of the wrong precision/shape -- observed as an OpenVINO ParameterMismatch exception or
// worse, a crash. Guard the reuse decision with a cheap structural check: every previously
// registered model input's element type must match the same-named tensor in the incoming cgraph,
// and any statically-known shape dimension must agree too (dynamic dims -- the token axis a real
// decode loop varies every step -- are always allowed to differ; that's the whole point of the
// cache).
bool ggml_decoder_inputs_compatible(const std::shared_ptr<GgmlOvDecoder> & decoder, const ggml_cgraph * cgraph) {
    std::unordered_map<std::string, const ggml_tensor *> by_name;
    auto visit = [&](const ggml_tensor * t) {
        if (t != nullptr && t->name[0] != '\0') {
            by_name.emplace(t->name, t);
        }
    };
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const auto * node = cgraph->nodes[i];
        visit(node);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            visit(node->src[j]);
        }
    }

    for (const auto & input : decoder->get_model_inputs()) {
        auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(input.second);
        if (!param) {
            continue;
        }
        auto it = by_name.find(input.first);
        if (it == by_name.end()) {
            // A previously registered model input has no same-named counterpart in the incoming
            // cgraph at all -- these are two structurally unrelated graphs that merely collided on
            // graph_key (e.g. one earlier op's test case and a later, differently-shaped one, both
            // ending on a node named "out"). Never reuse across that.
            return false;
        }
        if (GgmlOvDecoder::get_ov_type(it->second) != param->get_element_type()) {
            return false;
        }
        const auto & pshape = param->get_partial_shape();
        if (pshape.rank().is_static()) {
            auto shape = GgmlOvDecoder::get_shape(it->second);
            if (shape.size() != pshape.size()) {
                return false;
            }
            for (size_t d = 0; d < shape.size(); d++) {
                if (pshape[d].is_static() && static_cast<int64_t>(shape[d]) != pshape[d].get_length()) {
                    return false;
                }
            }
        }
    }
    return true;
}
}  // namespace

enum ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx) {
    auto & core = ov_singleton_core();
    const auto & config = ggml_openvino_get_compile_config();
    const auto & device = r_ctx->device;
    // Meta renames fragment inputs and can split attention from the canonical inp_pos input. Such
    // a fragment cannot keep an independent KV state synchronized, so retain stateless cache I/O.
    const auto * inp_pos = find_inp_pos_tensor(cgraph);
    const bool stateful = r_ctx->stateful && inp_pos != nullptr && strcmp(inp_pos->name, "inp_pos") == 0;
    static auto is_static = false;

    if (is_naive(cgraph)) {
        return naive_compute(cgraph, core, device, config);
    }

    auto start_time = ggml_time_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    graph_key key(cgraph);
    bool cache_hit;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;

    {
        std::shared_ptr<decoder_runtime_ctx> entry;
        ModelParams old_m_params;

        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            auto it = r_ctx->decoder_cache.find(key);
            cache_hit = it != r_ctx->decoder_cache.end();
            if (cache_hit) {
                entry = it->second;
            } else {
                auto mutex = std::make_shared<std::mutex>();
                entry = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[key] = entry;
            }
        }

        std::lock_guard<std::mutex> lock(*(entry->mutex));

        if (cache_hit) {
            ggml_decoder = entry->ptr;
            old_m_params = ggml_decoder->get_model_params();
            cache_hit = old_m_params.can_reuse_dynamically(m_params) &&
                        ggml_decoder_inputs_compatible(ggml_decoder, cgraph);
        }

        if (cache_hit) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->set_compute_params(c_params);
            ggml_decoder->set_model_params(m_params);
            if (old_m_params.kv_buffer_changed(m_params)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->add_extra_inputs();
            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(key);
            }

            // A linear-attention-only subgraph (qwen3-next's GDN layers, split off from the
            // full-attention layers) has no ROPE, hence no inp_pos, and carries only recurrent-state
            // caches, which are stateless (ggml owns the buffer, round-tripped as I/O). The SDPA-KV
            // stateful bookkeeping below applies only to attention subgraphs, so skip it when there is
            // no position input rather than treating its absence as fatal.
            const auto * inp_pos = stateful ? find_inp_pos_tensor(cgraph) : nullptr;
            if (stateful && inp_pos != nullptr) {
                int32_t * pos_data = (int32_t *) inp_pos->data;
                // Advance by the number of tokens actually in this ubatch, not by the allocated size of
                // the position buffer: ggml sizes inp_pos to n_batch and only fills the first input_len
                // entries, so pos_shape[3] overstates the count whenever a ubatch is smaller than
                // n_batch (chunked prefill, and every decode step). Overstating it desynchronizes
                // stateful_kv_size from pos_data[0], which sends the next call into the cache-trim
                // branch below with an out-of-range ROI end.
                const size_t n_tokens = static_cast<size_t>(ggml_decoder->get_input_len());
                if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
                    GGML_LOG_INFO("OpenVINO stateful cache hit: pos=%d, tokens=%zu, tracked=%zu\n",
                                  pos_data[0],
                                  n_tokens,
                                  entry->stateful_kv_size);
                }
                if (pos_data[0] == 0) {
                    infer_request->reset_state();
                    entry->stateful_kv_size = n_tokens;
                } else if (entry->stateful_kv_size == static_cast<size_t>(pos_data[0])) {
                    entry->stateful_kv_size += n_tokens;
                } else {
                    auto states = infer_request->query_state();
                    for (auto state : states) {
                        auto state_tensor = state.get_state();
                        auto state_tensor_shape = state_tensor.get_shape();
                        if (static_cast<uint32_t>(pos_data[0]) > entry->stateful_kv_size) {
                            std::string state_name;
                            try {
                                state_name = r_ctx->kv_state_input_name_map.at(state.get_name());
                            } catch (...) {
                                GGML_LOG_ERROR("GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                return GGML_STATUS_FAILED;
                            }
                            auto kv_tensor = get_ov_input_tensor(ggml_decoder, state_name);
                            const auto kv_shape = kv_tensor.get_shape();
                            ov::Coordinate begin = {0, 0, 0, 0};
                            ov::Coordinate end = {kv_shape[0], kv_shape[1], static_cast<size_t>(pos_data[0]), kv_shape[3]};
                            state_tensor = ov::Tensor(kv_tensor, begin, end);
                            state_tensor_shape = state_tensor.get_shape();
                        }
                        ov::Coordinate begin = {0, 0, 0, 0};
                        ov::Coordinate end = {state_tensor_shape[0], state_tensor_shape[1],
                                              static_cast<size_t>(pos_data[0]), state_tensor_shape[3]};
                        ov::Tensor new_state_tensor(state_tensor, begin, end);
                        state.set_state(new_state_tensor);
                    }
                    entry->stateful_kv_size = static_cast<size_t>(pos_data[0]) + n_tokens;
                }
            }

            decoder_end_time = ggml_time_us();
            conversion_end_time = decoder_end_time;
            compile_end_time = decoder_end_time;
        } else {
            if (ggml_openvino_weight_buffers_released()) {
                GGML_ABORT("ggml-openvino: a new graph requires compilation after host weights were released");
            }
            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(key);
            }

            std::shared_ptr<ov::Model> model;
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;  // weights flow as GGML_OP_NONE leaf nodes

            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static, stateful);
            decoder_end_time = ggml_time_us();

            const std::string model_cache_dir = ggml_openvino_model_cache_dir();
            const uint64_t model_fp = model_cache_dir.empty()
                                          ? 0
                                          : ggml_openvino_model_fingerprint(cgraph,
                                                                            device,
                                                                            true,
                                                                            m_params.rope_params,
                                                                            15,
                                                                            ggml_openvino_model_cache_extra_cfg(
                                                                                device,
                                                                                stateful));
            const std::string blob_path = model_fp == 0 ? std::string()
                                                        : ggml_openvino_model_cache_blob_path(model_cache_dir, model_fp);
            const std::string manifest_path =
                model_fp == 0 ? std::string() : ggml_openvino_model_cache_manifest_path(model_cache_dir, model_fp);
            ov::AnyMap model_cache_config = config;
            if (model_fp != 0) {
                model_cache_config.erase("CACHE_DIR");
                model_cache_config.erase("CACHE_MODE");
            }

            ov::CompiledModel compiled_model;
            bool imported = false;
            if (model_fp != 0 && ggml_openvino_model_cache_verify_manifest(manifest_path, cgraph, model_fp)) {
                std::ifstream blob(blob_path, std::ios::binary);
                if (blob.is_open()) {
                    try {
                        auto remote_context = ggml_openvino_get_remote_context();
                        compiled_model = remote_context.has_value()
                                             ? core.import_model(blob, remote_context.value(), model_cache_config)
                                             : core.import_model(blob, device, model_cache_config);
                        imported = true;
                        GGML_LOG_INFO("ggml-openvino: model cache HIT %s\n", blob_path.c_str());
                    } catch (const std::exception & e) {
                        GGML_LOG_WARN("ggml-openvino: model cache import failed: %s\n", e.what());
                    }
                }
            }

            // The frontend always emits the stateless model and lowers its SetRows ops to the
            // stateless ScatterUpdate form by default. For stateful execution, register the
            // stateful SetRows lowering as a transformation extension: the frontend runs it in the
            // normalization stage (ahead of the default lowering), yielding an OpenVINO stateful
            // model. Statefulness thus stays a backend concern with no is_stateful in the frontend.
            if (!imported) {
                ov::frontend::gguf::FrontEnd frontend;
                if (stateful) {
                    frontend.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
                        ggml::pass::LlamaCppToStateful(ggml_decoder->get_swa_kv_names())));
                }
                model = frontend.convert(
                    frontend.load(std::static_pointer_cast<ov::frontend::gguf::GgufDecoder>(ggml_decoder)));
                {
                    ov::pass::Manager manager;
                    manager.register_pass<ov::frontend::ggml::pass::FuseToConv>();
                    manager.run_passes(model);
                }
                conversion_end_time = ggml_time_us();

                if (getenv("GGML_OPENVINO_DUMP_IR")) {
                    char timestamped_filename[64];
                    auto timestamp = (long long) ggml_time_us();
                    snprintf(timestamped_filename, sizeof(timestamped_filename), "model_%lld.xml", timestamp);
                    ov::serialize(model, timestamped_filename);
                }

                auto remote_context = ggml_openvino_get_remote_context();
                compiled_model = remote_context.has_value()
                                     ? core.compile_model(model, remote_context.value(), model_cache_config)
                                     : core.compile_model(model, device, model_cache_config);
                if (model_fp != 0) {
                    const std::string blob_tmp = blob_path + ".tmp";
                    const std::string manifest_tmp = manifest_path + ".tmp";
                    if (ggml_openvino_model_cache_write_manifest(manifest_tmp, cgraph, model_fp)) {
                        std::ofstream blob(blob_tmp, std::ios::binary | std::ios::trunc);
                        if (blob.is_open()) {
                            compiled_model.export_model(blob);
                            blob.close();
                            if (blob.good() && std::rename(blob_tmp.c_str(), blob_path.c_str()) == 0 &&
                                std::rename(manifest_tmp.c_str(), manifest_path.c_str()) == 0) {
                                GGML_LOG_INFO("ggml-openvino: model cache WROTE %s\n", blob_path.c_str());
                            } else {
                                std::remove(blob_tmp.c_str());
                                std::remove(manifest_tmp.c_str());
                            }
                        }
                    }
                }
            } else {
                conversion_end_time = decoder_end_time;
            }
            compile_end_time = ggml_time_us();
            infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
            entry->ptr = ggml_decoder;

            std::vector<std::string> ov_input_names;
            std::vector<std::string> ov_output_names;
            // Derive input names from the actual compiled model, not the pre-compilation `model`:
            // a plugin transformation can drop a Parameter that becomes disconnected during its
            // own internal optimization (e.g. GatedDeltaNetFusion removing now-unused
            // recurrent-state bookkeeping inputs), which core.compile_model() applies to its own
            // clone without touching `model`. Binding tensors by index below must match the
            // compiled model's actual input count and order.
            for (const auto & ov_input : compiled_model.inputs()) {
                ov_input_names.push_back(ov_input.get_node()->get_friendly_name());
            }
            // LlamaCppToStateful removes the explicit KV-cache Results after replacing them with
            // Assign sinks. Preserve the remaining model Result order so output index i is still
            // paired with the right ggml tensor; using the decoder's pre-pass list here shifts all
            // later bindings (Falcon then tries to bind an f16 cache to the f32 logits port).
            // Without the stateful pass, retain the decoder-owned names to avoid Result-name
            // disambiguation when a cache is both a Parameter and an output.
            if (stateful && !imported) {
                for (const auto & result : model->get_results()) {
                    ov_output_names.push_back(result->get_friendly_name());
                }
            } else if (imported) {
                for (const auto & output : compiled_model.outputs()) {
                    ov_output_names.push_back(output.get_node()->get_friendly_name());
                }
            } else {
                ov_output_names = ggml_decoder->get_model_output_names();
            }

            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[key] = infer_request;
                r_ctx->ov_input_names_cache[key] = std::move(ov_input_names);
                r_ctx->ov_output_names_cache[key] = std::move(ov_output_names);
            }

            // See the note above: a linear-attention-only subgraph has no inp_pos and no SDPA-KV
            // state, so skip initializing the stateful KV bookkeeping for it.
            if (stateful) {
                const auto * inp_pos = find_inp_pos_tensor(cgraph);
                if (inp_pos != nullptr) {
                    const auto kv_param_res_names = ggml_decoder->get_kv_param_res_names();
                    for (const auto& pair : kv_param_res_names) {
                        // Current backend-owned state variables use the cache Parameter name as
                        // their id. Keep the concatenated legacy key too for imported/older graphs.
                        r_ctx->kv_state_input_name_map[pair.first] = pair.first;
                        r_ctx->kv_state_input_name_map[pair.first + pair.second] = pair.first;
                    }

                    const auto start_pos = static_cast<size_t>(static_cast<int32_t *>(inp_pos->data)[0]);
                    const auto n_tokens = static_cast<size_t>(ggml_decoder->get_input_len());
                    if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
                        GGML_LOG_INFO("OpenVINO stateful cache miss: pos=%zu, tokens=%zu\n", start_pos, n_tokens);
                    }
                    if (start_pos == 0) {
                        infer_request->reset_state();
                    } else {
                        // A cache miss can occur mid-sequence when llama.cpp rebuilds an otherwise
                        // equivalent cgraph. Seed the new request from ggml's authoritative cache;
                        // an empty ReadValue state plus positions starting at start_pos would make
                        // K/V length n_tokens while the causal mask spans start_pos+n_tokens.
                        for (auto state : infer_request->query_state()) {
                            auto map_it = r_ctx->kv_state_input_name_map.find(state.get_name());
                            if (map_it == r_ctx->kv_state_input_name_map.end()) {
                                GGML_LOG_ERROR("GGML OpenVINO backend stateful inference failed: no input found for state %s\n",
                                               state.get_name().c_str());
                                return GGML_STATUS_FAILED;
                            }
                            auto kv_tensor = get_ov_input_tensor(ggml_decoder, map_it->second);
                            const auto kv_shape = kv_tensor.get_shape();
                            if (start_pos > kv_shape[2]) {
                                GGML_LOG_ERROR("GGML OpenVINO backend stateful inference failed: position %zu exceeds cache %zu\n",
                                               start_pos,
                                               kv_shape[2]);
                                return GGML_STATUS_FAILED;
                            }
                            ov::Coordinate begin = {0, 0, 0, 0};
                            ov::Coordinate end = {kv_shape[0], kv_shape[1], start_pos, kv_shape[3]};
                            state.set_state(ov::Tensor(kv_tensor, begin, end));
                        }
                    }
                    entry->stateful_kv_size = start_pos + n_tokens;
                }
            }
        }

        std::vector<std::string> ov_input_names;
        std::vector<std::string> ov_output_names;
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            ov_input_names = r_ctx->ov_input_names_cache[key];
            ov_output_names = r_ctx->ov_output_names_cache[key];
        }

        for (size_t i = 0; i < ov_input_names.size(); i++) {
            auto param_name = ov_input_names[i];
            auto input_tensor = get_ov_input_tensor(ggml_decoder, param_name);
            infer_request->set_input_tensor(i, input_tensor);

            if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
                print_input_tensor_info(param_name, input_tensor);
            }
        }

        for (size_t i = 0; i < ov_output_names.size(); i++) {
            auto * ggml_tensor = ggml_decoder->get_model_outputs().at(ov_output_names[i]);
            // A zero-element ggml output (e.g. a 0-active-sequence recurrent-state cache writeback) has
            // nothing to receive; set_output_tensor would reject the size-0 ggml tensor against the
            // model's size-1 port. Skip the bind and let OV allocate its own (unused) output buffer.
            // Test element count, not ggml_nbytes: a [12288,0,1,1] tensor has 0 elements but the
            // (ne[0]-1)*nb[0] term keeps ggml_nbytes positive, so an nbytes check would not fire.
            if (ggml_nelements(ggml_tensor) == 0) {
                continue;
            }
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        infer_request->infer();
        infer_end_time = ggml_time_us();

        if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
            for (size_t i = 0; i < ov_output_names.size(); i++) {
                const auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
            }
        }
        for (size_t i = 0; i < ov_output_names.size(); i++) {
            maybe_dump_full_tensor(ov_output_names[i], infer_request->get_output_tensor(i));
        }

        if (getenv("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %ld ms \n", (decoder_end_time - start_time) / 1000);
            if (!cache_hit) {
                GGML_LOG_INFO("  - Graph conversion time: %ld ms \n", (conversion_end_time - decoder_end_time) / 1000);
                GGML_LOG_INFO("  - Graph compile time: %ld ms \n", (compile_end_time - conversion_end_time) / 1000);
            }
            GGML_LOG_INFO("  - Graph inference time: %ld ms \n", (infer_end_time - compile_end_time) / 1000);
        }
    }

    if (cache_hit && ggml_openvino_release_weights_enabled(device) &&
        !ggml_openvino_weight_buffers_released()) {
        ggml_openvino_release_weight_buffers();
    }

    return GGML_STATUS_SUCCESS;
}

enum ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx) {
    auto & core = ov_singleton_core();

    auto get_prefill_chunk_size = [] {
        const char * chunk_size_str = getenv("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
        if (chunk_size_str && atoi(chunk_size_str) > 0) {
            return atoi(chunk_size_str);
        }
        return 256;
    };

    static std::string device = ggml_openvino_get_device_name();
    static auto is_static = true;
    static auto stateful = false;
    static auto prefill_chunk_size = get_prefill_chunk_size();
    const auto & config = ggml_openvino_get_compile_config();

    if (is_naive(cgraph)) {
        return naive_compute(cgraph, core, device, config);
    }

    auto start_time = ggml_time_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    const auto * inp_pos = find_inp_pos_tensor(cgraph);
    const auto is_prefill = inp_pos != nullptr ? get_is_prefill(cgraph, inp_pos) : c_params.input_len > 1;
    graph_key key(cgraph);
    bool cache_hit;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;

    std::shared_ptr<decoder_runtime_ctx> entry;
    ModelParams old_m_params;

    {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        auto it = r_ctx->decoder_cache.find(key);
        cache_hit = it != r_ctx->decoder_cache.end();
        if (cache_hit) {
            entry = it->second;
        } else {
            auto mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[key] = entry;
        }
    }

    std::lock_guard<std::mutex> lock(*(entry->mutex));

    if (cache_hit) {
        ggml_decoder = entry->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_hit = old_m_params.can_reuse_statically(m_params) &&
                    ggml_decoder_inputs_compatible(ggml_decoder, cgraph);
    }

    if (cache_hit) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->set_model_params(m_params);
        ggml_decoder->set_compute_params(c_params);
        if (old_m_params.kv_buffer_changed(m_params)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->add_extra_inputs();
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(key) : r_ctx->infer_request_cache.at(key);
        }

        decoder_end_time = ggml_time_us();
        conversion_end_time = decoder_end_time;
        compile_end_time = decoder_end_time;
    } else {
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(key);
            r_ctx->infer_request_cache_prefill.erase(key);
        }

        std::shared_ptr<ov::Model> model;
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;  // weights flow as GGML_OP_NONE leaf nodes

        auto ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights,
                                                                    is_static, stateful, true, prefill_chunk_size);
        auto ggml_decoder_decode = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                                   stateful, false, prefill_chunk_size);
        decoder_end_time = ggml_time_us();

        std::shared_ptr<ov::Model> model_prefill;
        std::shared_ptr<ov::Model> model_decode;
        ov::CompiledModel compiled_model_prefill;
        ov::CompiledModel compiled_model_decode;
        std::shared_ptr<ov::InferRequest> infer_request_prefill;
        std::shared_ptr<ov::InferRequest> infer_request_decode;
        int64_t prefill_conversion_end_time;
        int64_t decode_conversion_end_time;
        int64_t prefill_compile_end_time;
        int64_t decode_compile_end_time;
        const bool dump_ir = getenv("GGML_OPENVINO_DUMP_IR") != nullptr;
        const auto dump_timestamp = static_cast<long long>(ggml_time_us());
        auto build_static_model = [&](const std::shared_ptr<GgmlOvDecoder> & decoder,
                                      const char * tag,
                                      std::shared_ptr<ov::Model> & output_model,
                                      ov::CompiledModel & output_compiled,
                                      std::shared_ptr<ov::InferRequest> & output_request,
                                      int64_t & output_conversion_end,
                                      int64_t & output_compile_end) {
            ov::frontend::gguf::FrontEnd frontend;
            output_model = frontend.convert(
                frontend.load(std::static_pointer_cast<ov::frontend::gguf::GgufDecoder>(decoder)));
            ov::pass::Manager manager;
            manager.register_pass<ov::frontend::ggml::pass::FuseToConv>();
            manager.register_pass<ggml::pass::SqueezeMatmul>();
            manager.run_passes(output_model);
            output_conversion_end = ggml_time_us();
            if (dump_ir) {
                char filename[64];
                snprintf(filename, sizeof(filename), "model_%s_%lld.xml", tag, dump_timestamp);
                ov::serialize(output_model, filename);
            }
            auto remote_context = ggml_openvino_get_remote_context();
            output_compiled = remote_context.has_value()
                                  ? core.compile_model(output_model, remote_context.value(), config)
                                  : core.compile_model(output_model, device, config);
            output_request = std::make_shared<ov::InferRequest>(output_compiled.create_infer_request());
            output_compile_end = ggml_time_us();
        };
        auto prefill_future = std::async(std::launch::async,
                                         build_static_model,
                                         ggml_decoder_prefill,
                                         "prefill",
                                         std::ref(model_prefill),
                                         std::ref(compiled_model_prefill),
                                         std::ref(infer_request_prefill),
                                         std::ref(prefill_conversion_end_time),
                                         std::ref(prefill_compile_end_time));
        auto decode_future = std::async(std::launch::async,
                                        build_static_model,
                                        ggml_decoder_decode,
                                        "decode",
                                        std::ref(model_decode),
                                        std::ref(compiled_model_decode),
                                        std::ref(infer_request_decode),
                                        std::ref(decode_conversion_end_time),
                                        std::ref(decode_compile_end_time));
        prefill_future.get();
        decode_future.get();
        conversion_end_time = std::max(prefill_conversion_end_time, decode_conversion_end_time);
        compile_end_time = std::max(prefill_compile_end_time, decode_compile_end_time);

        model = is_prefill ? model_prefill : model_decode;
        ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
        infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
        entry->ptr = ggml_decoder;

        std::vector<std::string> ov_input_names;
        std::vector<std::string> ov_output_names;
        // See the single-model compile path above for why inputs are read from the compiled
        // model's actual ports, and outputs from the decoder's own get_model_output_names().
        const auto & compiled_model = is_prefill ? compiled_model_prefill : compiled_model_decode;
        for (const auto & ov_input : compiled_model.inputs()) {
            ov_input_names.push_back(ov_input.get_node()->get_friendly_name());
        }
        ov_output_names = ggml_decoder->get_model_output_names();

        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache_prefill[key] = infer_request_prefill;
            r_ctx->infer_request_cache[key] = infer_request_decode;
            r_ctx->ov_input_names_cache[key] = std::move(ov_input_names);
            r_ctx->ov_output_names_cache[key] = std::move(ov_output_names);
        }
    }

    std::vector<std::string> ov_input_names_local;
    std::vector<std::string> ov_output_names_local;
    {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        ov_input_names_local = r_ctx->ov_input_names_cache[key];
        ov_output_names_local = r_ctx->ov_output_names_cache[key];
    }

    if (is_prefill) {
        const auto inp_len = c_params.input_len;
        for (int chunk_index = 0; chunk_index * prefill_chunk_size < inp_len; chunk_index++) {
            for (size_t i = 0; i < ov_input_names_local.size(); i++) {
                auto param_name = ov_input_names_local[i];
                auto input_tensor = get_ov_input_tensor_static_prefill(ggml_decoder, param_name, chunk_index);
                infer_request->set_input_tensor(i, input_tensor);

                if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
                    const auto input_tensor = infer_request->get_input_tensor(i);
                    print_input_tensor_info(param_name, input_tensor);
                }
            }

            for (size_t i = 0; i < ov_output_names_local.size(); i++) {
                auto * ggml_tensor = ggml_decoder->get_model_outputs().at(ov_output_names_local[i]);
                if (ggml_nelements(ggml_tensor) == 0) {
                    continue;
                }
                auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->set_output_tensor(i, output_tensor);
            }

            infer_request->infer();

            if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
                for (size_t i = 0; i < ov_output_names_local.size(); i++) {
                    const auto output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
                }
            }
            for (size_t i = 0; i < ov_output_names_local.size(); i++) {
                maybe_dump_full_tensor(ov_output_names_local[i], infer_request->get_output_tensor(i));
            }
        }
        infer_end_time = ggml_time_us();
    } else {
        for (size_t i = 0; i < ov_input_names_local.size(); i++) {
            auto param_name = ov_input_names_local[i];
            auto input_tensor = get_ov_input_tensor_static_decode(ggml_decoder, param_name);
            infer_request->set_input_tensor(i, input_tensor);

            if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
                const auto input_tensor = infer_request->get_input_tensor(i);
                print_input_tensor_info(param_name, input_tensor);
            }
        }

        for (size_t i = 0; i < ov_output_names_local.size(); i++) {
            auto * ggml_tensor = ggml_decoder->get_model_outputs().at(ov_output_names_local[i]);
            if (ggml_nelements(ggml_tensor) == 0) {
                continue;
            }
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        infer_request->infer();
        infer_end_time = ggml_time_us();

        if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
            for (size_t i = 0; i < ov_output_names_local.size(); i++) {
                const auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
            }
        }
        for (size_t i = 0; i < ov_output_names_local.size(); i++) {
            maybe_dump_full_tensor(ov_output_names_local[i], infer_request->get_output_tensor(i));
        }
    }

    if (getenv("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %ld ms \n", (decoder_end_time - start_time) / 1000);
        if (!cache_hit) {
            GGML_LOG_INFO("  - Graph conversion time: %ld ms \n", (conversion_end_time - decoder_end_time) / 1000);
            GGML_LOG_INFO("  - Graph compile time: %ld ms \n", (compile_end_time - conversion_end_time) / 1000);
        }
        GGML_LOG_INFO("  - Graph inference time: %ld ms \n", (infer_end_time - compile_end_time) / 1000);
    }

    return GGML_STATUS_SUCCESS;
}

bool is_naive(ggml_cgraph * cgraph) {
    constexpr int naive_graph_size_threshold = 20;
    int count = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    return count < naive_graph_size_threshold;
}

enum ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config) {
    if (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        return GGML_STATUS_SUCCESS;
    }

    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;  // weights flow as GGML_OP_NONE leaf nodes
    auto decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    // This bare-cgraph decoder exposes no "rope_config", so the frontend builds no shared LLM
    // scaffolding (the former naive=true path) -- see InputModel::get_rope_config.
    ov::frontend::gguf::FrontEnd frontend;
    auto model = frontend.convert(
        frontend.load(std::static_pointer_cast<ov::frontend::gguf::GgufDecoder>(decoder)));
    {
        ov::pass::Manager manager;
        manager.register_pass<ov::frontend::ggml::pass::FuseToConv>();
        manager.run_passes(model);
    }
    if (getenv("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    auto remote_context = ggml_openvino_get_remote_context();
    if (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        core.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE));
    } else {
        core.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
    }
    if (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), config).create_infer_request());
    } else {
        infer_request =
            std::make_shared<ov::InferRequest>(core.compile_model(model, device, config).create_infer_request());
    }

    auto ov_params = model->get_parameters();
    for (size_t i = 0; i < ov_params.size(); i++) {
        auto param_name = ov_params[i]->get_friendly_name();
        auto input_tensor = get_ov_input_tensor(decoder, param_name);
        infer_request->set_input_tensor(i, input_tensor);
    }

    infer_request->infer();

    // A single view-like op can have overlapping input/output storage. Let OpenVINO allocate its
    // output, then copy it after inference so the input is not overwritten while it is still read.
    auto ov_results = model->get_results();
    for (size_t i = 0; i < ov_results.size(); i++) {
        auto * ggml_tensor = decoder->get_model_outputs().at(ov_results[i]->get_friendly_name());
        if (ggml_nelements(ggml_tensor) == 0) {
            continue;
        }
        auto output_tensor = infer_request->get_output_tensor(i);
        const size_t copy_size = std::min(ggml_nbytes(ggml_tensor), output_tensor.get_byte_size());
        std::memcpy(ggml_tensor->data, output_tensor.data(), copy_size);
    }
    return GGML_STATUS_SUCCESS;
}

namespace {
ov::Tensor convert_ggml_input_to_ov(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & name) {
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(name);

    // A view never carries its own extra (see ggml_backend_openvino_buffer_init_tensor); resolve
    // view_src's *current* extra live here instead of relying on a value cached at some earlier
    // point, which could have gone stale if view_src's extra was replaced since (e.g. its data was
    // (re)written via ggml_backend_openvino_buffer_set_tensor, which installs a fresh extra object
    // and frees the old one).
    const auto * extra_src = ggml_tensor;
    while (extra_src->extra == nullptr && extra_src->view_src != nullptr) {
        extra_src = extra_src->view_src;
    }

    if (extra_src->extra != nullptr) {
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(extra_src->extra);
        if (extra_base->type != ggml_openvino_extra_base::Type::TENSOR) {
            throw std::runtime_error("ggml tensor extra is not of type TENSOR for input: " + name);
        }
        auto * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
        const ov::Tensor & ext = *tensor_extra->tensor;
        // A ggml VIEW/RESHAPE shares its view_src's extra, whose ov::Tensor carries the *allocation*
        // shape of the base buffer (e.g. the flat [1,1,1,24576] cache_r_l0 or the fused [1,1,5,8192]
        // QKV), not this view's own logical shape. When such a view crosses an OV subgraph boundary as
        // an input, the consuming subgraph's Parameter has the view's logical shape, so binding the raw
        // extra tensor fails the shape check. Reconcile the two:
        ov::Shape want = ggml_decoder->get_shape(ggml_tensor);
        if (ext.get_shape() != want) {
            if (ggml_is_contiguous(ggml_tensor) && ext.data() == ggml_tensor->data &&
                ov::shape_size(ext.get_shape()) == ov::shape_size(want)) {
                // Contiguous reshape/reinterpret of the whole buffer: re-wrap the same data.
                return ov::Tensor(ext.get_element_type(), want, ext.data());
            }
            // Non-contiguous (strided) view -- e.g. qwen3-next's Qcur_view selecting the first 256 of
            // each 512-wide fused-QKV row, or an offset sub-block. An ov::Tensor cannot wrap strided
            // memory, so materialize the view's logical contents into a fresh contiguous tensor,
            // copying with ggml's byte strides (nb[]). ggml tensors are at most 4D.
            ov::Tensor dense(ext.get_element_type(), want);
            const size_t esz = ggml_element_size(ggml_tensor);
            const auto * src_base = static_cast<const char *>(ggml_tensor->data);
            auto * dst = static_cast<char *>(dense.data());
            const int64_t n0 = ggml_tensor->ne[0], n1 = ggml_tensor->ne[1];
            const int64_t n2 = ggml_tensor->ne[2], n3 = ggml_tensor->ne[3];
            const size_t nb0 = ggml_tensor->nb[0], nb1 = ggml_tensor->nb[1];
            const size_t nb2 = ggml_tensor->nb[2], nb3 = ggml_tensor->nb[3];
            size_t dst_off = 0;
            for (int64_t i3 = 0; i3 < n3; ++i3) {
                for (int64_t i2 = 0; i2 < n2; ++i2) {
                    for (int64_t i1 = 0; i1 < n1; ++i1) {
                        const char * row = src_base + i3 * nb3 + i2 * nb2 + i1 * nb1;
                        if (nb0 == esz) {
                            std::memcpy(dst + dst_off, row, n0 * esz);
                            dst_off += n0 * esz;
                        } else {
                            for (int64_t i0 = 0; i0 < n0; ++i0) {
                                std::memcpy(dst + dst_off, row + i0 * nb0, esz);
                                dst_off += esz;
                            }
                        }
                    }
                }
            }
            return dense;
        }
        return ext;
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    auto * input_data = ggml_tensor->data;
    ov::Shape input_shape;
    if (ggml_tensor->op == GGML_OP_VIEW) {
        // This case is added to make test-backend-ops work
        input_shape = ggml_decoder->get_shape(ggml_tensor->view_src);
    } else {
        input_shape = ggml_decoder->get_shape(ggml_tensor);
    }
    auto input_tensor = ov::Tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape, input_data);
    return input_tensor;
}
}  // namespace

ov::Tensor get_ov_input_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & param_name) {
    ov::Tensor input_tensor;
    if (ggml_decoder->get_model_extra_inputs().find(param_name) != ggml_decoder->get_model_extra_inputs().end()) {
        input_tensor = *ggml_decoder->get_model_extra_input_values().at(param_name);
    } else {
        input_tensor = convert_ggml_input_to_ov(ggml_decoder, param_name);
    }
    return input_tensor;
}

ov::Tensor get_ov_input_tensor_static_decode(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                             const std::string & param_name) {
    // NPU decoding stage
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(param_name);
    const auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    if (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        assert(ggml_tensor->ne[0] == 1);
        ov::Shape input_shape = {1, 1, 1, 1};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        if (ggml_tensor->type == GGML_TYPE_I32) {
            *input_tensor.data<int32_t>() = *((int32_t *) ggml_tensor->data);
        } else if (ggml_tensor->type == GGML_TYPE_I64) {
            *input_tensor.data<int64_t>() = *((int64_t *) ggml_tensor->data);
        } else {
            throw std::runtime_error("Unexpected tensor type for " + param_name);
        }
        return input_tensor;
    }

    if (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape input_shape = {1, 1, 1, 1};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        int32_t inp_out_id = *((int32_t *) ggml_tensor->data);
        assert(ggml_tensor->ne[0] == 1);
        assert(inp_out_id == 0);
        *input_tensor.data<int32_t>() = inp_out_id;
        return input_tensor;
    }

    if (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t context_size = ggml_decoder->get_ctx_size();
        if (ggml_tensor->type == GGML_TYPE_F16) {
            auto padded_data =
                pad_input<ggml_fp16_t>(ggml_tensor, 1, context_size, GGML_FP32_TO_FP16(-INFINITY));
            ov::Tensor input_tensor(ov::element::f16, {1, 1, 1, context_size});
            std::memcpy(input_tensor.data(), padded_data.data(), padded_data.size() * sizeof(ggml_fp16_t));
            return input_tensor;
        }
        std::vector<float> padded_data = pad_input<float>(ggml_tensor, 1, context_size, -INFINITY);
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_size});
        auto * data_ptr = input_tensor.data<float>();
        std::copy(padded_data.begin(), padded_data.begin() + context_size, data_ptr);
        return input_tensor;
    }

    return get_ov_input_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_input_tensor_static_prefill(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                              const std::string & param_name,
                                              int chunk_index) {
    // NPU prompt processing stage
    const size_t input_len = ggml_decoder->get_input_len();
    const size_t chunk_size = ggml_decoder->m_prefill_chunk_size;
    const size_t chunk_valid_size = std::min(chunk_size, input_len - chunk_index * chunk_size);
    const size_t chunk_pad_size = chunk_size - chunk_valid_size;

    if (param_name == "chunk_valid_len") {
        ov::Tensor input_tensor(ov::element::i64, ov::Shape{1});
        *input_tensor.data<int64_t>() = static_cast<int64_t>(chunk_valid_size);
        return input_tensor;
    }
    if (chunk_index > 0 && param_name == "cache_rs_reset_len") {
        ov::Tensor input_tensor(ov::element::i64, ov::Shape{1});
        *input_tensor.data<int64_t>() = 0;
        return input_tensor;
    }
    if (ggml_decoder->get_model_extra_inputs().count(param_name)) {
        return get_ov_input_tensor(ggml_decoder, param_name);
    }

    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(param_name);
    const auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    if (GgmlOvDecoder::is_inp_pos(ggml_tensor, op) && GgmlOvDecoder::get_inp_pos_n_planes(op) > 1) {
        const int n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
        const size_t element_size = ggml_type_size(ggml_tensor->type);
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), {1, 1, 1, static_cast<size_t>(n_planes) * chunk_size});
        for (int plane = 0; plane < n_planes; ++plane) {
            const char * src = static_cast<const char *>(ggml_tensor->data) +
                               (plane * input_len + chunk_index * chunk_size) * element_size;
            char * dst = static_cast<char *>(input_tensor.data()) + plane * chunk_size * element_size;
            std::memcpy(dst, src, chunk_valid_size * element_size);
            if (chunk_pad_size > 0) {
                if (ggml_tensor->type == GGML_TYPE_I32) {
                    const int32_t last = reinterpret_cast<const int32_t *>(src)[chunk_valid_size - 1];
                    std::fill(reinterpret_cast<int32_t *>(dst) + chunk_valid_size,
                              reinterpret_cast<int32_t *>(dst) + chunk_size,
                              last + 1);
                } else if (ggml_tensor->type == GGML_TYPE_I64) {
                    const int64_t last = reinterpret_cast<const int64_t *>(src)[chunk_valid_size - 1];
                    std::fill(reinterpret_cast<int64_t *>(dst) + chunk_valid_size,
                              reinterpret_cast<int64_t *>(dst) + chunk_size,
                              last + 1);
                } else {
                    throw std::runtime_error("Unexpected tensor type for " + param_name);
                }
            }
        }
        return input_tensor;
    }

    if (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape input_shape = {1, 1, 1, chunk_size};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_size = ggml_type_size(ggml_tensor->type);
        void * input_data = (char *) ggml_tensor->data + chunk_index * chunk_size * element_size;
        std::memcpy(input_tensor.data(), input_data, chunk_valid_size * element_size);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        if (chunk_pad_size > 0) {
            if (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t last_value =
                    *((int32_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int32_t * output_data = input_tensor.data<int32_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_value + 1);
            } else if (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t last_value =
                    *((int64_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int64_t * output_data = input_tensor.data<int64_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_value + 1);
            } else {
                throw std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        return input_tensor;
    }

    if (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape input_shape = {1, 1, 1, output_len};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        if (ggml_tensor->ne[0] == 0) {
            *input_tensor.data<int32_t>() = 0;
        } else {
            auto * data_addr = input_tensor.data<int32_t>();
            for (size_t i = 0; i < output_len; i++) {
                data_addr[i] = ((int32_t *) ggml_tensor->data)[i] % chunk_size;
            }
        }
        return input_tensor;
    }

    if (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t cols = ggml_tensor->ne[0];
        size_t rows = ggml_tensor->ne[1];
        size_t chunk_valid_rows = std::min(chunk_size, rows - chunk_index * chunk_size);
        size_t context_size = ggml_decoder->get_ctx_size();
        if (ggml_tensor->type == GGML_TYPE_F16) {
            const auto * ggml_data = static_cast<const ggml_fp16_t *>(ggml_tensor->data) +
                                     chunk_index * chunk_size * cols;
            auto padded_data = pad_input<ggml_fp16_t>(ggml_data,
                                                       chunk_valid_rows,
                                                       cols,
                                                       chunk_size,
                                                       context_size,
                                                       GGML_FP32_TO_FP16(-INFINITY));
            set_zero_diagonal(padded_data, chunk_size, context_size, GGML_FP32_TO_FP16(0.0f));
            ov::Tensor input_tensor(ov::element::f16, {1, 1, chunk_size, context_size});
            std::memcpy(input_tensor.data(), padded_data.data(), padded_data.size() * sizeof(ggml_fp16_t));
            return input_tensor;
        }
        const auto * ggml_data = static_cast<const float *>(ggml_tensor->data) + chunk_index * chunk_size * cols;
        std::vector<float> padded_data =
            pad_input<float>(ggml_data, chunk_valid_rows, cols, chunk_size, context_size, -INFINITY);
        set_zero_diagonal(padded_data, chunk_size, context_size);
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, context_size});
        auto * data_ptr = input_tensor.data<float>();
        std::copy(padded_data.begin(), padded_data.begin() + chunk_size * context_size, data_ptr);
        return input_tensor;
    }

    return get_ov_input_tensor(ggml_decoder, param_name);
}

size_t checksum(const void * data, size_t size) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    size_t sum = 0;
    for (size_t i = 0; i < size; ++i) {
        sum += (uint8_t) i;
        sum += bytes[i];
    }
    return sum;
}

void print_input_tensor_info(const std::string & name, const ov::Tensor & tensor) {
    std::cout << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.data()
              << std::endl;
    switch (tensor.get_element_type()) {
    case ov::element::f32: {
        if (name.find("self_kq_mask") == std::string::npos) {
            std::cout << *(tensor.data<float>()) << std::endl;
        } else {
            size_t rows = tensor.get_shape()[2];
            size_t cols = tensor.get_shape()[3];
            auto * data = tensor.data<float>();
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    float val = data[i * cols + j];
                    if (std::isinf(val) && val < 0) {
                        std::cout << std::setw(5) << "-inf";
                    } else {
                        std::cout << std::setw(5) << val;
                    }
                }
                std::cout << std::endl;
            }
        }

        break;
    }
    case ov::element::f16:
        std::cout << *(tensor.data<ov::float16>()) << std::endl;
        break;
    case ov::element::i32:
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int32_t>()[i] << " ";
        }
        std::cout << std::endl;
        break;
    case ov::element::i64:
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int64_t>()[i] << " ";
        }
        std::cout << std::endl;
        break;
    default:
        break;
    }
}

void print_output_tensor_info(const std::string & name, const ov::Tensor & tensor, const void * output_dst) {
    std::cout << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << std::endl;

    auto print_float_stats = [](const std::string & type_name, size_t size, auto get_value) {
        if (size == 0) {
            return;
        }

        float first = get_value(0);
        float min = first;
        float max = first;
        double sum = first;

        for (size_t i = 1; i < size; ++i) {
            float v = get_value(i);
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
            sum += v;
        }
        double mean = sum / size;

        std::cout << std::right << std::setw(6) << type_name << std::right << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << std::endl;
        std::cout << std::right << std::setw(6) << "" << std::right << std::setw(12) << first << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << mean << std::endl;
    };

    switch (tensor.get_element_type()) {
    case ov::element::f32: {
        const float * data = tensor.data<float>();
        size_t size = tensor.get_size();
        print_float_stats("[f32]", size, [data](size_t i) { return data[i]; });
        break;
    }
    case ov::element::f16: {
        const ov::float16 * data = tensor.data<ov::float16>();
        size_t size = tensor.get_size();
        print_float_stats("[f16]", size, [data](size_t i) { return static_cast<float>(data[i]); });
        break;
    }
    default:
        break;
    }
}

// Debug seam: dump the FULL flat contents of an output tensor to a file, one float per line, when
// its name matches (substring) $GGML_OPENVINO_DUMP_TENSOR. Unlike print_output_tensor_info (which
// only prints First/Min/Max/Mean -- all permutation-invariant, so blind to head-scramble/layout
// bugs) this preserves element order for an element-wise cross-backend diff vs the ggml-CPU ref.
void maybe_dump_full_tensor(const std::string & name, const ov::Tensor & tensor) {
    const char * want = getenv("GGML_OPENVINO_DUMP_TENSOR");
    if (!want || name.find(want) == std::string::npos) {
        return;
    }
    static std::map<std::string, int> counters;
    std::string base = name;
    for (char & c : base) {
        if (c == '/' || c == ' ') {
            c = '_';
        }
    }
    std::string fname = "/tmp/ov_dump_" + base + "_" + std::to_string(counters[base]++) + ".txt";
    std::ofstream ofs(fname);
    ofs << std::setprecision(9);
    if (tensor.get_element_type() == ov::element::f32) {
        const float * data = tensor.data<float>();
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            ofs << data[i] << "\n";
        }
    } else if (tensor.get_element_type() == ov::element::f16) {
        const ov::float16 * data = tensor.data<ov::float16>();
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            ofs << static_cast<float>(data[i]) << "\n";
        }
    }
    std::cout << "[DUMP_TENSOR] wrote " << tensor.get_size() << " elems to " << fname << std::endl;
}

const ggml_tensor * find_inp_pos_tensor(ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            auto * src = op->src[j];
            if (src == nullptr) {
                break;
            }
            if (GgmlOvDecoder::is_inp_pos(src, op)) {
                return src;
            }
        }
    }
    return nullptr;
}

const ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    const auto * inp_pos = find_inp_pos_tensor(cgraph);
    if (inp_pos == nullptr) {
        GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
        throw std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
    }
    return inp_pos;
}

int64_t get_inp_pos_n_tokens(ggml_cgraph * cgraph, const ggml_tensor * inp_pos) {
    int n_planes = 1;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (op->src[j] == inp_pos) {
                n_planes = GgmlOvDecoder::get_inp_pos_n_planes(op);
                break;
            }
        }
    }
    return inp_pos->ne[0] / n_planes;
}

bool get_is_prefill(ggml_cgraph * cgraph, const ggml_tensor * inp_pos) {
    return get_inp_pos_n_tokens(cgraph, inp_pos) > 1;
}

#pragma GCC diagnostic pop
