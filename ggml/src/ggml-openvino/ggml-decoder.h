#pragma once

#include "ggml-backend-impl.h"  // ggml_backend_buffer::usage (recurrent-state cache detection)
#include "ggml-backend.h"       // GGML_BACKEND_BUFFER_USAGE_ANY
#include "ggml-quants.h"
#include "ggml.h"
#include "openvino/frontend/gguf/decoder.hpp"

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <openvino/core/partial_shape.hpp>
#include <optional>
#include <set>
#include <vector>

struct ModelParams {
    int ctx = -1;
    int ctx_swa = -1;
    int ctx_per_seq = -1;
    int ctx_per_seq_swa = -1;
    int n_seq = 1;
    int n_heads = -1;
    int n_heads_kv = -1;
    int head_size = -1;
    int state_size = -1;  // recurrent-state row width for SSM/DeltaNet models (qwen3-next); -1 = not recurrent
    // The first ROPE op's op_params, used to build the shared sin/cos table. Zero-initialized so a
    // graph with no ROPE op (an embedding-only graph, or the small-subgraph path whose bare decoder
    // never runs compute_llm_params) yields n_dims == 0, which the frontend reads as "model uses no
    // RoPE" -- rather than stack garbage, which produces an absurd sin/cos width (a float bit
    // pattern read as an int) and a broadcast failure inside the first ROPE.
    int32_t rope_params[15] = {};
    // Set when the graph's ROPE ops carry divergent op_params (e.g. gemma4's SWA vs global layers
    // use different n_dims / freq_base). The frontend maps this to RopeConfig::per_op so each ROPE
    // op builds its own sin/cos instead of sharing a single precomputed table.
    bool mixed_rope_params = false;
    bool has_zero_sized_tensor = false;
    uint64_t graph_shape_signature = 0;
    std::vector<int> swa_layers;
    // The sliding-window attention mask tensor, when this graph is an iSWA model (gemma3/gemma4).
    // llama.cpp builds both masks through one shared build_attn_inp_kq_mask helper, so both are
    // named "attn_inp_kq_mask" and the name cannot tell them apart -- they are distinct tensors
    // though, so identity is the discriminator. nullptr = no SWA layers in this graph.
    // swa_mask_src is the pre-staging graph input when the mask reaches attention through a CPY.
    const ggml_tensor * swa_mask = nullptr;
    const ggml_tensor * swa_mask_src = nullptr;

    bool is_swa_mask(const ggml_tensor * mask) const {
        return mask != nullptr && (mask == swa_mask || mask == swa_mask_src);
    }

    std::vector<std::string> kv_names;
    size_t kv_buffer_ctx_id = 0;

    bool same_rope_params(const ModelParams & other) const {
        return memcmp(rope_params, other.rope_params, sizeof(int32_t) * 15) == 0;
    }

    bool can_reuse_dynamically(const ModelParams & other) const {
        return same_rope_params(other) && has_zero_sized_tensor == other.has_zero_sized_tensor &&
               graph_shape_signature == other.graph_shape_signature;
    }

    bool can_reuse_statically(const ModelParams & other) const { return same_rope_params(other) && ctx == other.ctx; }

    bool kv_buffer_changed(const ModelParams & other) const { return kv_buffer_ctx_id != other.kv_buffer_ctx_id; }
};

struct ComputeParams {
    int n_seq_active = 1;
    int seq_active_start = 0;
    int attention_size = -1;
    int attention_size_swa = -1;
    int input_len = -1;
    int token_len_per_seq = -1;
    int past_kv_len = -1;
    int output_len = 1;

    // Recurrent-state (SSM/DeltaNet, qwen3-next) machinery. -1 means "not present in this graph".
    //
    // cache_rs_reset_idx/_len: an in-place SCALE-by-0 on a slot block [idx, idx+len) of the
    // recurrent-state cache clears it when a sequence starts fresh (the graph omits the SCALE
    // otherwise). We surface idx/len as graph inputs so one compiled model handles both cases.
    int cache_rs_reset_idx = -1;
    int cache_rs_reset_len = -1;

    int s_copy_active_slot_len = -1;

    struct RsWriteback {
        int slot_begin = 0;
        int src_begin = -1;
    };

    std::map<std::string, RsWriteback> rs_writebacks;
};

class GgmlOvDecoder : public ov::frontend::gguf::GgufDecoder {
public:
    struct NodeInfo {
        ggml_tensor * node;
        std::string node_name;
        std::string node_op_type;
        std::map<std::string, ggml_tensor *> node_inputs;
        std::vector<std::string> node_inputs_names;
        ggml_tensor * node_output;
        std::string node_output_name;
        int node_op_case = 0;
        void * data_addr;
    };
    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool is_prefill = false,
                  int prefill_chunk_size = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights);

    // Per-node accessors refer to the node this decoder is bound to (m_node_idx, set by
    // visit_subgraph). The model-scoped decoder has m_node_idx == -1 and only answers the
    // model-level queries (get_model_inputs, get_model_output_names, ...).
    virtual ov::Any get_attribute(const std::string & name) const override;

    virtual size_t get_input_size() const override;

    virtual std::vector<std::string> get_input_names() const override;

    virtual ov::PartialShape get_output_shape() const override;

    virtual ov::PartialShape get_input_shape(const std::string & name) const override;

    virtual int64_t get_input_view_element_offset(const std::string & name) const override;

    virtual std::vector<std::string> get_output_names() const override;

    virtual const std::string & get_op_type() const override;

    virtual const std::string & get_op_name() const override;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgufDecoder>)> node_visitor) const override;

    ggml_tensor * get_input_ggml_tensor(const std::string & name) const { return m_inputs.at(name); }

    // Returns all model-scope input nodes (primary Parameters + auxiliary constants/parameters).
    // Callers distinguish Parameters from auxiliary nodes via dynamic_pointer_cast.
    virtual const std::map<std::string, std::shared_ptr<ov::Node>> & get_model_inputs() const override {
        return m_all_model_inputs;
    }

    // Backend-internal: extra auxiliary input nodes (attention_size, n_seq_active, etc.) kept
    // separately so the runtime can access their initial values via get_model_extra_input_values().
    const std::map<std::string, std::shared_ptr<ov::Node>> & get_model_extra_inputs() const {
        return m_model_extra_inputs;
    }

    virtual const std::map<std::string, std::shared_ptr<ov::Tensor>> & get_model_extra_input_values() const {
        return m_model_extra_input_values;
    }

    const std::map<std::string, std::shared_ptr<ov::Node>> & get_model_weights() const {
        return m_model_weights;
    }

    virtual std::vector<std::string> get_model_output_names() const override {
        return m_model_output_names;
    }

    const std::map<std::string, ggml_tensor *> & get_model_outputs() const { return m_model_outputs; }

    virtual int get_ctx_size() const { return m_model_params.ctx; }

    // Decoder-internal helper (used by compute_op_case to classify KV-cache permutes); not part
    // of the GgufDecoder frontend interface.
    bool is_swa_layer(int layer) const {
        return std::find(m_model_params.swa_layers.begin(), m_model_params.swa_layers.end(), layer) !=
               m_model_params.swa_layers.end();
    }

    int get_input_len() const { return m_compute_params.input_len; }

    // Typed RoPE config, exposed to the frontend via get_attribute("rope_config") (model and
    // node scope). RopeConfig::n_dims == 0 means no RoPE; per_op is currently always false.
    ov::frontend::gguf::RopeConfig get_rope_config() const;

    // KV-cache Parameter/Result name pairs, used by the backend's stateful runtime bookkeeping.
    // Not part of the GgufDecoder frontend interface.
    std::map<std::string, std::string> get_kv_param_res_names() const;

    // Names of the KV caches belonging to sliding-window layers. Empty unless this graph is iSWA.
    // The stateful lowering leaves these stateless -- see LlamaCppToStateful's constructor.
    std::set<std::string> get_swa_kv_names() const;


    ov::PartialShape get_graph_input_shape(const ggml_tensor * op, const ggml_tensor * input) const;

    static void dump_cgraph(const ggml_cgraph * cgraph, std::string & filename);

    static std::shared_ptr<ov::Node> create_weight_node(ggml_tensor * tensor, bool naive = false);

    const ggml_tensor * get_tensor_used_op(const ggml_tensor * tensor) const;

    static std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() const { return m_model_params; }

    ComputeParams get_compute_params() const { return m_compute_params; }

    void set_model_params(const ModelParams & model_params) { m_model_params = model_params; }

    void set_compute_params(const ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    int m_prefill_chunk_size = 0;

    static ov::Shape get_shape(const ggml_tensor * tensor);
    static std::vector<size_t> get_stride(const ggml_tensor * tensor);
    static ov::element::Type get_ov_type(const ggml_tensor * tensor);
    static std::string compute_op_type(const ggml_tensor * node);
    void add_extra_inputs();

    void update_io(ggml_cgraph * cgraph);

    inline static bool is_inp_tok(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    inline static bool is_inp_pos(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    inline static bool is_inp_emb(const ggml_tensor * tensor, const ggml_tensor * op) {
        return tensor->op == GGML_OP_GET_ROWS && op->op == GGML_OP_RMS_NORM;
    }

    inline static bool is_inp_mask(const ggml_tensor * tensor, const ggml_tensor * op) {
        // Attention mask fed to: a CPY (mask staging), FLASH_ATTN_EXT src[3] (flash path), or
        // SOFT_MAX src[1] (the "-fa off" non-flash path, where attention is softmax(QK+mask)).
        return op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]) ||
               (op->op == GGML_OP_SOFT_MAX && tensor == op->src[1]);
    }

    inline static bool is_rope_freqs_weight(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    // Also matches the recurrent-state caches (cache_r conv-state, cache_s GDN-state) in SSM/
    // DeltaNet models: those live in a USAGE_ANY buffer and are written in place, not via SET_ROWS.
    inline static bool is_kvcache(const ggml_tensor * tensor, const ggml_tensor * op) {
        return (tensor != nullptr && tensor->buffer != nullptr &&
                tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) ||
               (op != nullptr && op->op == GGML_OP_SET_ROWS && op->src[2] == tensor);
    }

    inline static bool is_kv_idx(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_SET_ROWS && op->src[1] == tensor;
    }

    inline static bool is_output_idx(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE;
    }

    // The state-permutation index list (inp->s_copy in llama-graph.cpp) used by SSM/DeltaNet models
    // to reorder recurrent-state cache slots: it is the GET_ROWS index (src[1]) whose gathered data
    // (src[0]) is a recurrent-state cache (USAGE_ANY buffer).
    inline static bool is_inp_s_copy(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0] != nullptr &&
               op->src[0]->buffer != nullptr && op->src[0]->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY;
    }

    // Not static: telling the two attention masks of an iSWA model apart needs the whole-graph
    // classification in m_model_params (see ModelParams::swa_mask).
    std::string get_graph_input_ov_name(const ggml_tensor * tensor, const ggml_tensor * op) const {
        if (is_inp_tok(tensor, op)) {
            return "inp_tokens";
        }
        if (is_inp_pos(tensor, op)) {
            return "inp_pos";
        }
        if (is_inp_emb(tensor, op)) {
            return "embd";
        }
        if (is_output_idx(tensor, op)) {
            return "inp_out_ids";
        }
        if (is_inp_mask(tensor, op)) {
            return m_model_params.is_swa_mask(tensor) ? "self_kq_mask_swa" : "self_kq_mask";
        }
        return tensor->name;
    }

private:
    void set_input_output();
    int compute_op_case(const ggml_tensor * node) const;
    bool node_is_used_as_src(const int node_idx);
    void compute_model_inputs();
    void compute_model_outputs();
    // Infer, per graph tensor, which single ggml dimension (0..3, or -1 for none) is the variable
    // token/sequence axis, propagating it stride-based through the graph. The result is folded into
    // the PartialShapes returned by get_output_shape / get_input_shape / get_graph_input_shape so the
    // converted model has a dynamic token axis (not baked to the prefill length) on the dynamic path.
    void compute_node_dynamic_dims();
    // Dynamic ggml-dim index (0..3) per tensor; absent or -1 means fully static.
    int dynamic_dim_of(const ggml_tensor * tensor) const;
    // Fallback token-axis inference for a graph-input tensor that the stride-based dynamic-dim pass
    // could not resolve (e.g. gemma4's host-computed per-layer embedding `inp_per_layer`, fed whole
    // and sliced per layer by select-and-drop VIEWs). Returns the ggml dim (0..3) that a consuming
    // select-and-drop VIEW keeps and that is not dim0 (the embedding width) -- i.e. the token axis --
    // or -1 if no such consumer exists.
    int token_axis_from_consumer_views(const ggml_tensor * input) const;
    // Last-resort token-axis recovery for an inter-subgraph boundary leaf produced on CPU (qwen3-next's
    // split graph feeds e.g. Qcur_full-* / Qcur_view-* into an OV subgraph). Such a tensor has no
    // in-subgraph producer and no select-and-drop consumer, so both dynamic_dim_of() and
    // token_axis_from_consumer_views() bottom out; without this it would bake the prefill token count
    // and the shorter decode call would fail to bind. Recover the token axis as the (unique) ggml dim
    // whose size equals the prefill token count; only meaningful during prefill (input_len > 1), where
    // that size is unambiguous. Returns the ggml dim (0..3) or -1.
    int token_axis_by_input_len(const ggml_tensor * input) const;
    // The subgraph's prefill token count, or -1 if not determinable (decode step / no dynamic tensor).
    // Prefers m_compute_params.input_len; falls back to ne[dyn] of any tensor whose dynamic axis
    // compute_node_dynamic_dims() resolved (all agree on the count) -- FFN-only splits never set
    // input_len. Used to recover / validate token axes for CPU-produced boundary leaves.
    int derived_token_len() const;

    void validate_cgraph() const;

    // Assign a stable, UNIQUE name per ggml tensor pointer. ggml connects nodes by pointer and can
    // give several distinct tensors the same name (e.g. the 8 per-expert "ffn_moe_weighted (view)"
    // slices in an MoE layer). The frontend keys its TensorMap by name, so colliding names make
    // consumers read the wrong producer. This disambiguates by appending a suffix when a base name
    // is already taken by a different pointer; the same pointer always maps to the same name, so
    // producers and consumers agree.
    std::string unique_tensor_name(const ggml_tensor * tensor, const std::string & base);
    std::unordered_map<const ggml_tensor *, std::string> m_tensor_unique_name;
    std::set<std::string> m_used_names;

    ggml_cgraph * m_cgraph = nullptr;
    // Index of the node this decoder is bound to (set by visit_subgraph); -1 for the
    // model-scoped decoder, which only answers model-level queries.
    int m_node_idx = -1;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, std::shared_ptr<ov::Node>> m_model_inputs;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::Node>> m_all_model_inputs;  // union of the above two
    std::map<std::string, std::shared_ptr<ov::Tensor>> m_model_extra_input_values;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_weights;
    std::set<std::string> m_weight_names;  // GGML_OP_NONE leaves surfaced as weight nodes
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::vector<std::string> m_model_output_names;
    std::map<const ggml_tensor *, int> m_node_dynamic_dims;  // ggml-dim index of the dynamic axis
    std::vector<NodeInfo> m_node_info_list;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

void print_tensor_address_map(const ggml_cgraph * cgraph);

int extract_layer_from_name(const std::string & name);
