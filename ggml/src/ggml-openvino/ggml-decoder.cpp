#include "ggml-decoder.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-openvino-extra.h"
#include "ggml-openvino.h"
#include "ggml-quants.h"

#include <ggml-impl.h>
#include <ggml.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <openvino/core/dimension.hpp>
#include <openvino/core/except.hpp>
#include <openvino/core/node.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/runtime/tensor.hpp>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
// An in-place op writes its result back into its view_src tensor rather than producing a fresh
// buffer: SET_ROWS (KV-cache writeback), CPY into a recurrent-state cache view (conv/GDN state),
// and SCALE of a recurrent-state cache view (the reset-by-0). For these the decoder routes the
// node's output to the view_src's name so downstream consumers read the updated tensor.
bool is_inplace_op(const ggml_tensor * node) {
    if (node == nullptr) {
        return false;
    }
    return node->op == GGML_OP_SET_ROWS || (node->op == GGML_OP_CPY && node->view_src != nullptr) ||
           (node->op == GGML_OP_SCALE && node->view_src != nullptr);
}

bool is_conv_states_all_tensor(const ggml_tensor * tensor) {
    return tensor != nullptr && strncmp(tensor->name, "conv_states_all", strlen("conv_states_all")) == 0;
}

bool is_conv_state_writeback(const ggml_tensor * node) {
    return node->op == GGML_OP_CPY && node->view_src != nullptr && GgmlOvDecoder::is_kvcache(node->view_src, nullptr) &&
           node->src[0] != nullptr && node->src[0]->op == GGML_OP_VIEW && node->src[0]->src[0] != nullptr &&
           node->src[0]->src[0]->op == GGML_OP_CONCAT && node->src[1] != nullptr && node->src[1]->op == GGML_OP_VIEW &&
           node->src[1]->view_src == node->view_src;
}

std::string recurrent_writeback_name(const ggml_cgraph * cgraph, const ggml_tensor * tensor) {
    const size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    if (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        return std::string(tensor->name) + "#" + std::to_string(hash_pos);
    }
    return tensor->name;
}

// A VIEW's byte offset (op_params[0..1] as size_t).
size_t view_byte_offset(const ggml_tensor * node) {
    size_t byte_offset = 0;
    std::memcpy(&byte_offset, node->op_params, sizeof(size_t));
    return byte_offset;
}

// Detect the qwen3-next "feature-window + reshape" VIEW: a contiguous 2D source [F, T] (token count
// T on ggml dim1) reinterpreted so the non-token node dims pack a contiguous feature sub-window
// [off, off+feat_len) of F (e.g. q/k/v_conv split the mixed [4096, tok] conv output into per-head
// tensors). Returns {off_elem, feat_len} on match (a slice of ggml dim0 == OV innermost axis,
// followed by a reshape to the view's own layout), or {-1,-1} otherwise. This is a strided
// reinterpretation the plain ne-comparison shrink/select detection cannot see.
std::pair<int64_t, int64_t> detect_view_feature_window(const ggml_tensor * node) {
    const ggml_tensor * src = node->src[0];
    if (src == nullptr || src->op == GGML_OP_VIEW) {
        return {-1, -1};
    }
    if (src->ne[2] != 1 || src->ne[3] != 1 || !ggml_is_contiguous(src)) {
        return {-1, -1};  // src must be a contiguous 2D [F, T]
    }
    const size_t elem = ggml_type_size(src->type);
    if (elem == 0 || src->nb[0] != elem) {
        return {-1, -1};
    }
    const size_t token_stride = src->nb[1];  // bytes per token in src
    // The token axis is the unique node dim whose stride equals the src token stride and whose
    // extent equals T; every other node dim must index below the token stride (i.e. into F).
    int token_dim = -1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (node->nb[i] == token_stride && node->ne[i] == src->ne[1]) {
            token_dim = i;
            break;
        }
    }
    if (token_dim < 0) {
        return {-1, -1};
    }
    int64_t feat_len = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (i == token_dim) {
            continue;
        }
        if (node->ne[i] > 1 && node->nb[i] >= token_stride) {
            return {-1, -1};
        }
        feat_len *= node->ne[i];
    }
    if (feat_len <= 0 || feat_len > src->ne[0]) {
        return {-1, -1};
    }
    // The window must be DENSELY packed on ggml dim0: the feature dims (non-token, non-singleton),
    // sorted by ascending stride, must tile contiguously starting at elem (running stride == elem *
    // product-of-smaller-extents). An interleaved split (qwen3.5 Q/gate: head stride 512 but head
    // extent 256, leaving a 256-elem gate gap) fails this and must NOT be flattened into a contiguous
    // [off, off+feat_len) slice -- that reinterprets the gap as feature data and scrambles the heads.
    // detect_view_interleaved_window handles those via a stride-preserving reshape instead.
    {
        struct FeatDim {
            int64_t ne;
            size_t nb;
        };

        std::vector<FeatDim> feats;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (i == token_dim || node->ne[i] <= 1) {
                continue;
            }
            feats.push_back({node->ne[i], node->nb[i]});
        }
        std::sort(feats.begin(), feats.end(), [](const FeatDim & a, const FeatDim & b) { return a.nb < b.nb; });
        size_t running = elem;
        for (const auto & f : feats) {
            if (f.nb != running) {
                return {-1, -1};  // gap between feature dims -> not a dense contiguous window
            }
            running *= static_cast<size_t>(f.ne);
        }
    }
    const size_t byte_offset = view_byte_offset(node);
    if (byte_offset % elem != 0) {
        return {-1, -1};
    }
    const int64_t off_elem = static_cast<int64_t>(byte_offset / elem);
    if (off_elem < 0 || off_elem + feat_len > src->ne[0]) {
        return {-1, -1};
    }
    return {off_elem, feat_len};
}

// Detect the qwen3.5 "interleaved feature window" VIEW: a contiguous 2D source [F, T] (token count
// T on ggml dim1) split into per-head sub-blocks that are INTERLEAVED with another tensor along the
// feature axis, so the per-head stride is larger than the per-head extent. E.g. the joint Q/gate
// projection Qcur_full is [(head_dim*2)*n_head, T] laid out per token as
// [Q_h0(head_dim), gate_h0(head_dim), Q_h1, gate_h1, ...]; the Q view is ne=[head_dim,n_head,T] with
// head stride head_dim*2 (not head_dim), and the gate view is the same with byte_offset = head_dim.
//
// detect_view_feature_window rejects this (it requires the non-token dims to pack a *contiguous*
// window), and the plain op_case-3 slice would treat [off, off+head_dim*n_head) as contiguous and
// scramble the heads. This detector instead recovers the regular {group_count, group_stride,
// inner_off, inner_len} tiling so translate_view can reshape src to expose the stride, slice the
// inner sub-window, and reshape to the view's own per-head layout. Returns valid=false unless the
// view is a clean, uniformly-strided window (so a genuine dense feature-window still takes the
// simpler op_case-3 path).
struct InterleavedWindow {
    bool valid = false;
    int64_t token_dim = -1;    // node dim carrying T
    int64_t group_count = 1;   // number of interleaved groups packed on ggml dim0 (== n_head here)
    int64_t group_stride = 0;  // element stride between consecutive groups on ggml dim0
    int64_t inner_off = 0;     // element offset of the wanted sub-block within a group
    int64_t inner_len = 0;     // element length of the wanted sub-block within a group
};

InterleavedWindow detect_view_interleaved_window(const ggml_tensor * node) {
    InterleavedWindow w;
    const ggml_tensor * src = node->src[0];
    if (src == nullptr || src->op == GGML_OP_VIEW) {
        return w;
    }
    if (src->ne[2] != 1 || src->ne[3] != 1 || !ggml_is_contiguous(src)) {
        return w;  // src must be a contiguous 2D [F, T]
    }
    const size_t elem = ggml_type_size(src->type);
    if (elem == 0 || src->nb[0] != elem) {
        return w;
    }
    const size_t token_stride = src->nb[1];  // bytes per token in src
    // Token axis: the unique node dim whose stride == src token stride and extent == T.
    int token_dim = -1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (node->nb[i] == token_stride && node->ne[i] == src->ne[1]) {
            token_dim = i;
            break;
        }
    }
    if (token_dim < 0) {
        return w;
    }
    // Among the non-token dims, exactly one may carry the group stride (a dim whose byte stride is a
    // multiple of elem and > its own extent*elem, i.e. leaves a gap). The innermost dim (ggml dim0)
    // is the per-group sub-block. Require: node->nb[0] == elem (dense within the sub-block); a single
    // "group" dim g with node->nb[g] % elem == 0 and stride/elem > inner_len; all other non-token
    // dims singleton.
    if (node->nb[0] != elem) {
        return w;
    }
    const int64_t inner_len = node->ne[0];  // sub-block extent on ggml dim0 (e.g. head_dim)
    int group_dim = -1;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (i == token_dim) {
            continue;
        }
        if (node->ne[i] <= 1) {
            continue;  // singleton -> ignore
        }
        if (group_dim != -1) {
            return w;  // more than one non-trivial group dim: not the simple interleaved split
        }
        group_dim = i;
    }
    if (group_dim < 0) {
        return w;  // no repeated group -> a plain contiguous window, let feature-window handle it
    }
    if (node->nb[group_dim] % elem != 0) {
        return w;
    }
    const int64_t group_stride = static_cast<int64_t>(node->nb[group_dim] / elem);
    const int64_t group_count = node->ne[group_dim];
    // The interleave is real only when the group stride exceeds the sub-block extent (a gap between
    // groups). A stride equal to inner_len is a *dense* window (feature-window path handles it).
    if (group_stride <= inner_len) {
        return w;
    }
    const size_t byte_offset = view_byte_offset(node);
    if (byte_offset % elem != 0) {
        return w;
    }
    const int64_t off_elem = static_cast<int64_t>(byte_offset / elem);
    // off_elem selects the sub-block within the first group; it must stay inside one group.
    const int64_t inner_off = off_elem % group_stride;
    if (off_elem / group_stride != 0 || inner_off + inner_len > group_stride) {
        return w;
    }
    // The whole tiling must fit within src dim0.
    if (group_stride * group_count > src->ne[0]) {
        return w;
    }
    w.valid = true;
    w.token_dim = token_dim;
    w.group_count = group_count;
    w.group_stride = group_stride;
    w.inner_off = inner_off;
    w.inner_len = inner_len;
    return w;
}

bool is_flat_kv_view(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->op != GGML_OP_VIEW || tensor->view_src == nullptr) {
        return false;
    }
    const ggml_tensor * base = tensor->view_src;
    return base->ne[1] == 1 && base->ne[2] == 1 && base->ne[3] == 1;
}
}  // namespace

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             bool is_static,
                             bool is_stateful,
                             bool is_prefill,
                             int prefill_chunk_size) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(false),
    m_prefill_chunk_size(prefill_chunk_size),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    if (auto * env = getenv("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS"); env && std::string(env) != "0") {
#ifdef _WIN32
        _putenv_s("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS", "");
#else
        unsetenv("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS");
#endif
        print_tensor_address_map(cgraph);
    }

    validate_cgraph();

    set_input_output();
    compute_node_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    for (int node_n = 0; node_n < (int) m_node_info_list.size(); node_n++) {
        if (m_weight_names.count(m_node_info_list[node_n].node_name)) {
            continue;  // weight nodes already have their op type / case set
        }
        m_node_info_list[node_n].node_op_case = compute_op_case(m_node_info_list[node_n].node);
        m_node_info_list[node_n].node_op_type = compute_op_type(m_node_info_list[node_n].node);
    }

    add_extra_inputs();
}

void GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_node_info_list.clear();
    set_input_output();
    compute_model_inputs();
    compute_model_outputs();
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = true;
    set_input_output();
    compute_model_inputs();
    compute_model_outputs();
    m_all_model_inputs = m_model_inputs;  // no extra inputs in naive mode
    for (int node_n = 0; node_n < (int) m_node_info_list.size(); node_n++) {
        if (m_weight_names.count(m_node_info_list[node_n].node_name)) {
            continue;  // weight nodes already have their op type / case set
        }
        m_node_info_list[node_n].node_op_case = compute_op_case(m_node_info_list[node_n].node);
        m_node_info_list[node_n].node_op_type = compute_op_type(m_node_info_list[node_n].node);
    }
}

std::string GgmlOvDecoder::unique_tensor_name(const ggml_tensor * tensor, const std::string & base) {
    auto it = m_tensor_unique_name.find(tensor);
    if (it != m_tensor_unique_name.end()) {
        return it->second;  // same pointer -> same name (producer/consumer agree)
    }
    std::string name = base;
    if (m_used_names.count(name)) {
        // Base name already taken by a DIFFERENT tensor pointer: disambiguate. Use the pointer so
        // the suffix is stable across the run (ggml tensor addresses are fixed within a cgraph).
        char buf[32];
        snprintf(buf, sizeof(buf), "#%p", (const void *) tensor);
        name = base + buf;
    }
    m_used_names.insert(name);
    m_tensor_unique_name[tensor] = name;
    return name;
}

void GgmlOvDecoder::set_input_output() {
    // First, surface each unique quantized/weight tensor as a node. A weight is a ggml leaf
    // (GGML_OP_NONE) and keeps that genuine op type; the frontend recognizes it as a weight by
    // the presence of the "data" attribute (get_attribute), dequantizes/requantizes from the raw
    // bytes, and the decoder only provides them. These come first so a weight is visited before
    // its consumer.
    m_tensor_unique_name.clear();
    m_used_names.clear();
    std::set<const ggml_tensor *> injected_external_views;
    {
        std::set<std::string> & seen_weights = m_weight_names;
        seen_weights.clear();
        for (int node_n = 0; node_n < m_cgraph->n_nodes; node_n++) {
            auto * node = m_cgraph->nodes[node_n];
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                auto * src = node->src[i];
                if (src == nullptr || src->view_src) {
                    continue;
                }
                std::string src_name(src->name);
                if (is_rope_freqs_weight(src, node)) {
                    src_name = "rope_freqs.weight";
                }
                ggml_backend_buffer * buffer = src->buffer;
                const bool is_weight =
                    is_rope_freqs_weight(src, node) ||
                    (buffer && (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type)));
                if (!is_weight || seen_weights.count(src_name)) {
                    continue;
                }
                seen_weights.insert(src_name);
                m_used_names.insert(src_name);         // reserve canonical weight name
                m_tensor_unique_name[src] = src_name;  // weights keep their canonical name
                NodeInfo wi;
                wi.node = src;
                wi.node_name = src_name;
                wi.node_op_type = "GGML_OP_NONE";  // genuine ggml leaf op type
                wi.node_output = src;
                wi.node_output_name = src_name;
                wi.node_op_case = 0;
                wi.data_addr = src->data;
                m_node_info_list.push_back(wi);
            }
        }
    }

    // Pre-pass: VIEW-of-an-external-input that is only referenced as a src (never a standalone
    // cgraph node). gemma4's per-layer embedding does this: a host-computed CONT
    // `inp_per_layer (permuted) (cont)` [emb, tok, n_layers] is fed whole, and each transformer
    // layer consumes a select-and-drop VIEW picking that layer's slice. Because those VIEW tensors
    // are not cgraph nodes, they are never translated and leak in as their own graph-input
    // Parameters (shaped like a single-layer slice) while the backend binds the whole CONT buffer to
    // them -> a shape mismatch. Inject each such VIEW as a translated node (so it is intermediate,
    // not an input) that slices the shared CONT; compute_model_inputs then surfaces only the CONT.
    {
        std::set<const ggml_tensor *> cgraph_nodes;
        for (int node_n = 0; node_n < m_cgraph->n_nodes; node_n++) {
            cgraph_nodes.insert(m_cgraph->nodes[node_n]);
        }
        for (int node_n = 0; node_n < m_cgraph->n_nodes; node_n++) {
            auto * node = m_cgraph->nodes[node_n];
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                auto * src = node->src[i];
                if (src == nullptr || src->op != GGML_OP_VIEW) {
                    continue;
                }
                if (cgraph_nodes.count(src) || injected_external_views.count(src)) {
                    continue;  // already a real node / already injected
                }
                const ggml_tensor * vsrc = src->view_src;
                if (vsrc == nullptr) {
                    continue;
                }
                if (cgraph_nodes.count(vsrc) || m_weight_names.count(std::string(vsrc->name))) {
                    continue;  // view_src is produced locally / is a weight -> normal handling
                }
                // Cache views consumed by stateful operations have dedicated handling. Ordinary
                // backend inputs also use USAGE_ANY, so the buffer usage alone is not sufficient.
                const bool dedicated_cache_view =
                    vsrc->buffer && vsrc->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY &&
                    (node->op == GGML_OP_PERMUTE || node->op == GGML_OP_SET_ROWS || node->op == GGML_OP_CPY ||
                     node->op == GGML_OP_SCALE || (node->op == GGML_OP_GET_ROWS && i == 0) ||
                     (node->op == GGML_OP_FLASH_ATTN_EXT && is_flat_kv_view(src)));
                if (dedicated_cache_view) {
                    continue;
                }
                injected_external_views.insert(src);

                const std::string vsrc_name = unique_tensor_name(vsrc, std::string(vsrc->name));
                const std::string view_name = unique_tensor_name(src, std::string(src->name));
                NodeInfo vi;
                vi.node = const_cast<ggml_tensor *>(src);
                vi.node_name = std::string(src->name);
                vi.node_output = const_cast<ggml_tensor *>(src);
                vi.node_output_name = view_name;
                vi.node_op_case = 0;  // set by the shared compute_op_case pass below
                vi.data_addr = src->data;
                vi.node_inputs[vsrc_name] = const_cast<ggml_tensor *>(vsrc);
                vi.node_inputs_names.push_back(vsrc_name);
                m_node_info_list.push_back(vi);
            }
        }
    }

    for (int node_n = 0; node_n < m_cgraph->n_nodes; node_n++) {
        auto node = m_cgraph->nodes[node_n];

        NodeInfo current_node_info;
        auto node_name = std::string(node->name);
        auto * node_output = node;
        // Unique output name per tensor pointer (disambiguates ggml name collisions, e.g. the 8
        // per-expert "ffn_moe_weighted (view)" slices that would otherwise overwrite each other in
        // the frontend's name-keyed TensorMap).
        auto node_output_name = unique_tensor_name(node, node_name);
        if (is_inplace_op(node)) {
            // In-place ops (SET_ROWS KV-cache writeback, CPY/SCALE into a recurrent-state cache
            // view) update the tensor in place. For later ov ops that use the view_src, we route
            // this node's output to the view_src name in the tensor_map (translate_session.cpp) so
            // they read the updated tensor.
            node_output = node->view_src;
            node_output_name = unique_tensor_name(node->view_src, std::string(node->view_src->name));
        }

        current_node_info.node = node;
        current_node_info.node_name = node_name;
        current_node_info.node_output = node_output;
        current_node_info.node_output_name = node_output_name;
        current_node_info.node_op_case = 0;
        current_node_info.data_addr = node->data;

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = node->src[i];
            if (src == nullptr) {
                continue;
            }
            std::string src_name;
            const bool recurrent_cpy_source =
                i == 0 && node->op == GGML_OP_CPY && src->op == GGML_OP_VIEW && src->src[0] != nullptr &&
                (is_conv_state_writeback(node) || src->src[0]->op == GGML_OP_GATED_DELTA_NET);
            const bool recurrent_cpy_destination = i == 1 && node->op == GGML_OP_CPY && src->op == GGML_OP_VIEW &&
                                                   src->view_src != nullptr && is_kvcache(src->view_src, nullptr);
            const bool recurrent_scale_source =
                i == 0 && node->op == GGML_OP_SCALE && node->view_src != nullptr && is_kvcache(node->view_src, nullptr);
            if (recurrent_cpy_source) {
                src_name = unique_tensor_name(src->src[0], std::string(src->src[0]->name));
            } else if (recurrent_cpy_destination) {
                src_name = unique_tensor_name(src->view_src, std::string(src->view_src->name));
            } else if (recurrent_scale_source) {
                src_name = unique_tensor_name(node->view_src, std::string(node->view_src->name));
            } else if (node->op == GGML_OP_FLASH_ATTN_EXT && is_flat_kv_view(src)) {
                // Whisper exposes K/V as direct 3-D views of a flat cache allocation. Pass the
                // allocation itself to the frontend and retain the VIEW tensor in node_inputs so
                // typed attributes can describe its offset and logical head geometry.
                src_name = unique_tensor_name(src->view_src, std::string(src->view_src->name));
            } else if (injected_external_views.count(src)) {
                src_name = unique_tensor_name(src, std::string(src->name));
            } else if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                src_name = get_graph_input_ov_name(src, node);
            } else if (m_weight_names.count(std::string(src->name)) || is_rope_freqs_weight(src, node)) {
                // Weights keep their canonical (shared) name.
                src_name = is_rope_freqs_weight(src, node) ? "rope_freqs.weight" : std::string(src->name);
            } else {
                // Regular producer output: use the same unique name the producer registered (keyed
                // by the src pointer), so this consumer reads the correct producer.
                src_name = unique_tensor_name(src, std::string(src->name));
            }
            current_node_info.node_inputs[src_name] = src;
            current_node_info.node_inputs_names.push_back(src_name);
        }

        m_node_info_list.push_back(current_node_info);
    }
}

int GgmlOvDecoder::compute_op_case(const ggml_tensor * node) const {
    int op_case = 0;
    switch (node->op) {
    case GGML_OP_RESHAPE: {
        auto * src = node->src[0];
        if (src->ne[0] == node->ne[0] && src->ne[1] == node->ne[1] && src->ne[2] == node->ne[2] &&
            src->ne[3] == node->ne[3]) {
            // Identity reshape (ggml src and node have the same ne, e.g. qwen3-next's
            // linear_attn_qkv_mixed after the mul_mat): a no-op in ggml. Emitting a static reshape here
            // would bake in the compile-time token count and fail at a differing runtime count (0-token
            // decode). Pass the input through unchanged so the dynamic token axis is preserved.
            op_case = 8;
        } else if (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == node->ne[0] &&
                   src->src[0]->ne[1] == node->ne[1]) {
            op_case = 4;
        } else if (node->ne[0] * node->ne[1] == src->ne[0]) {
            op_case = 1;
        } else if (src->ne[0] * src->ne[1] == node->ne[0]) {
            op_case = 2;
            if (src->ne[2] * src->ne[3] == node->ne[1]) {
                op_case = 5;
            }
        } else if (src->ne[0] * src->ne[1] == node->ne[1]) {
            op_case = 3;
        } else if (src->ne[1] * src->ne[2] == node->ne[1]) {
            op_case = 6;
        } else {
            // General fully-static reshape (no genuine variable token/sequence axis to preserve),
            // e.g. qwen3-next's recurrent-state predelta reshape [262144] -> [16,128,128]. A dynamic
            // dim marked on a size-1 axis contributes factor 1 and is not a real variable count, so
            // treat it as static. Reshape straight to the static output shape.
            const int dn = dynamic_dim_of(node);
            const int ds = dynamic_dim_of(src);
            const bool node_static = (dn == -1) || (node->ne[dn] == 1);
            const bool src_static = (ds == -1) || (src->ne[ds] == 1);
            if (node_static && src_static) {
                op_case = 7;
            }
        }
        break;
    }
    case GGML_OP_CONT: {
        if (node->src[0]->op == GGML_OP_PERMUTE) {
            op_case = 1;
        } else if (node->src[0]->op == GGML_OP_TRANSPOSE) {
            op_case = 2;
        } else if (node->src[0]->op == GGML_OP_VIEW) {
            op_case = 3;
        }
        break;
    }
    case GGML_OP_POOL_2D: {
        const auto mode = static_cast<ggml_op_pool>(ggml_get_op_params_i32(node, 0));
        if (mode == GGML_OP_POOL_MAX) {
            op_case = 1;
        } else if (mode == GGML_OP_POOL_AVG) {
            op_case = 2;
        }
        break;
    }
    case GGML_OP_PERMUTE: {
        if (node->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } else if (node->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(node->view_src->name);
            int layer = extract_layer_from_name(src_name);
            if (!is_swa_layer(layer)) {
                op_case = 2;
            } else {
                op_case = 3;
            }
        } else {
            // rope'ed query tensor
            op_case = 4;
        }
        break;
    }
    case GGML_OP_MUL_MAT: {
        if (node->src[0]->op == GGML_OP_CONT && node->src[0]->src[0]->op == GGML_OP_TRANSPOSE) {
            op_case = 2;
        } else if (node->src[0]->op == GGML_OP_VIEW && node->src[1]->op == GGML_OP_VIEW) {
            op_case = 3;
        }
        break;
    }
    case GGML_OP_GET_ROWS: {
        if (node->src[0] != nullptr && node->src[0]->buffer != nullptr &&
            node->src[0]->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY &&
            strncmp(node->src[0]->name, "cache_", 6) == 0) {
            op_case = ggml_nelements(node) == 0 ? 3 : 4;
        } else if (node->src[1]->op == GGML_OP_VIEW) {
            op_case = 2;
        }
        break;
    }
    case GGML_OP_CPY: {
        if (node->src[0]->op == GGML_OP_VIEW) {
            if (node->src[0]->src[0] != nullptr && node->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 1;
            } else if (is_conv_state_writeback(node)) {
                op_case = 2;
            } else if (is_conv_states_all_tensor(node->view_src) && node->src[1] != nullptr &&
                       node->src[1]->op == GGML_OP_VIEW && node->src[1]->view_src == node->view_src) {
                op_case = 4;
            }
        } else if (node->src[0]->op == GGML_OP_GET_ROWS && node->src[1] != nullptr &&
                   node->src[1]->op == GGML_OP_VIEW && node->src[1]->view_src != nullptr &&
                   is_kvcache(node->src[1]->view_src, nullptr)) {
            // Empty recurrent-state compaction is a no-op. Keep it explicit because the selected
            // row count may become dynamic in the OpenVINO graph and cannot then be detected from
            // PartialShape alone by the CPY translator.
            op_case = ggml_nelements(node->src[0]) == 0 ? 5 : 3;
        }
        break;
    }
    case GGML_OP_SCALE:
        if (node->view_src != nullptr && is_kvcache(node->view_src, nullptr)) {
            op_case = 1;
        }
        break;
    case GGML_OP_ROPE: {
        const int mode = node->op_params[2];
        switch (mode) {
        case GGML_ROPE_TYPE_NEOX: {
            op_case = 0x00010000;
            break;
        }
        case GGML_ROPE_TYPE_IMROPE: {
            op_case = 0x00020000;
            break;
        }
        default:
            op_case = 0x00000000;
            break;
        }
        if (node->src[0]->op == GGML_OP_VIEW) {
            op_case = (op_case | 0x00000002);
        }
        break;
    }
    case GGML_OP_VIEW: {
        if (node->src[0]->op == GGML_OP_GATED_DELTA_NET) {
            // qwen3-next GDN packs its result as [attn_rows(T) | state_rows(S_v)] stacked on the row
            // axis. attn_output-0 (byte_offset 0) reads the first T rows; new_state-0 (byte_offset>0)
            // reads the last S_v rows. translate_view slices the packed output along that axis.
            op_case = 4;
            break;
        }
        if (node->src[0]->op == GGML_OP_VIEW) {
            auto * src = node->src[0];
            if (ggml_nelements(node) != ggml_nelements(src)) {
                throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 2;
        }
        {
            auto * src = node->src[0];
            // op_case 3 drives translate_view's single-axis Slice(+Reshape), for two MoE patterns:
            //   * pure shrink of exactly one dim, contiguous (ffn_moe_topk: 64->8), and
            //   * offset-selected sub-block keeping the innermost dim (per-expert ffn_moe_weighted:
            //     [2048,8,1]->[2048,1,1], offset picks the expert).
            // It must NOT capture the KV-cache reshape-views (cache_k/v_l* (view)), which change the
            // innermost dim and grow rank -- those keep their existing handling (op_case 0/backend).
            if (src != nullptr && ggml_nelements(node) != ggml_nelements(src) && src->op != GGML_OP_VIEW) {
                size_t byte_offset = 0;
                std::memcpy(&byte_offset, node->op_params, sizeof(size_t));
                int diff_count = 0;
                int diff_dim = -1;
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    if (node->ne[i] != src->ne[i]) {
                        diff_count++;
                        diff_dim = i;
                    }
                }
                bool strides_preserved = true;
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    if (node->nb[i] != src->nb[i]) {
                        strides_preserved = false;
                        break;
                    }
                }
                // Shrink of exactly one dim keeping the parent strides. Two sub-cases:
                //   * offset 0 (ffn_moe_topk: 64->8), and
                //   * offset > 0 selecting a contiguous sub-range along that dim (qwen3-next
                //     conv_state_last: last d_conv-1 columns of the conv window). The offset must be a
                //     clean multiple of the sliced dim's stride. view_slice computes {axis,start,len}.
                const bool pure_shrink = (diff_count == 1) && strides_preserved && diff_dim >= 0 &&
                                         src->nb[diff_dim] > 0 && (byte_offset % src->nb[diff_dim] == 0);
                bool select_and_drop = false;
                for (int d = 0; d < GGML_MAX_DIMS && !select_and_drop; ++d) {
                    if (src->ne[d] <= 1) {
                        continue;
                    }
                    int64_t squeezed[GGML_MAX_DIMS];
                    int p = 0;
                    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                        if (i != d) {
                            squeezed[p++] = src->ne[i];
                        }
                    }
                    squeezed[p] = 1;
                    bool match = true;
                    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                        if (node->ne[i] != squeezed[i]) {
                            match = false;
                            break;
                        }
                    }
                    select_and_drop = match;
                }
                // qwen3-next feature-window + reshape (q/k/v_conv split the mixed [F, tok] conv
                // output into per-head tensors). Not a plain shrink/select -- detected via strides.
                // It reuses op_case 3 (restore -> Slice -> Reshape-with-dynamic-token); view_slice
                // and view_reshape return the window slice / target layout for it.
                const bool feature_window = detect_view_feature_window(node).first >= 0;
                const bool interleaved_window = detect_view_interleaved_window(node).valid;
                if (pure_shrink || select_and_drop || feature_window) {
                    op_case = 3;
                } else if (interleaved_window) {
                    // qwen3.5 interleaved Q/gate split: a strided (non-dense) feature window that the
                    // contiguous op_case-3 slice would scramble. translate_view reshapes to expose the
                    // per-group stride, slices the inner sub-window, and keeps the token axis dynamic.
                    op_case = 5;
                }
            }
        }
        break;
    }
    case GGML_OP_FLASH_ATTN_EXT:
        if (is_flat_kv_view(node->src[1]) && is_flat_kv_view(node->src[2])) {
            op_case = node->src[3] != nullptr ? 1 : 2;
        }
        break;
    default:
        break;
    }
    return op_case;
}

int extract_layer_from_name(const std::string & name) {
    size_t pos1 = name.find("_l");
    // Callers only reach here for KV-cache tensors, whose names always contain "_l<layer>".
    // Use a hard check (not assert) so a violated assumption fails loudly even in NDEBUG builds
    // instead of underflowing pos1 and feeding garbage to substr/stoi.
    if (pos1 == std::string::npos) {
        throw std::runtime_error("extract_layer_from_name: no \"_l<layer>\" in tensor name: " + name);
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    if (pos2 == std::string::npos) {
        pos2 = name.length();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    int layer = std::stoi(layer_str);
    return layer;
}

namespace {
// Walk a FLASH_ATTN_EXT's K input back to the K-cache leaf it views:
//   src[1] -> [CPY] -> PERMUTE -> VIEW -> cache_k_l<N>
// Returns nullptr when the chain does not match, so callers that only need a best-effort
// classification can skip the node instead of failing.
const ggml_tensor * attn_kvcache_view(const ggml_tensor * attn) {
    const ggml_tensor * perm = attn->src[1];
    if (perm != nullptr && perm->op == GGML_OP_CPY) {
        perm = perm->src[0];
    }
    if (perm == nullptr || perm->op != GGML_OP_PERMUTE || perm->src[0] == nullptr || perm->src[0]->op != GGML_OP_VIEW ||
        perm->src[0]->src[0] == nullptr) {
        return nullptr;
    }
    return perm->src[0];
}

const ggml_tensor * attn_kvcache(const ggml_tensor * attn) {
    const ggml_tensor * view = attn_kvcache_view(attn);
    return view == nullptr ? nullptr : view->src[0];
}
}  // namespace

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, bool is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    bool seen_rope = false;

    uint64_t shape_signature = 1469598103934665603ULL;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        shape_signature ^= static_cast<uint64_t>(node->op);
        shape_signature *= 1099511628211ULL;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            shape_signature ^= static_cast<uint64_t>(node->ne[d]);
            shape_signature *= 1099511628211ULL;
        }
        if (ggml_nelements(node) == 0) {
            model_params.has_zero_sized_tensor = true;
        }
    }
    model_params.graph_shape_signature = shape_signature;

    // Sliding-window-mask pre-scan over the WHOLE graph. It must run before the main loop, which
    // consults model_params.swa_mask to classify each attention node.
    //
    // An iSWA model (gemma3/gemma4) feeds its SWA layers a different mask tensor than its global
    // layers, but llama.cpp builds both through the same build_attn_inp_kq_mask helper, so both are
    // named "attn_inp_kq_mask" and the name cannot tell them apart. Discriminate structurally
    // instead: the two flavors read K caches of different lengths (n_swa rows vs n_ctx rows) and the
    // shorter cache is the sliding-window one. Fall back to the masks' own widths when the K cache
    // is not reachable (the "-fa off" softmax path), where the SWA mask is likewise the narrower.
    // A single-flavor model leaves swa_mask null and takes the global path throughout.
    {
        struct MaskInfo {
            const ggml_tensor * mask;
            const ggml_tensor * cache_k;  // K cache this mask's attention reads; nullptr = unknown
        };

        std::vector<MaskInfo> masks;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const auto * node = cgraph->nodes[i];
            const ggml_tensor * mask = nullptr;
            const ggml_tensor * cache_k = nullptr;
            if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                mask = node->src[3];
                cache_k = attn_kvcache(node);
            } else if (node->op == GGML_OP_SOFT_MAX) {
                mask = node->src[1];
            }
            if (mask == nullptr) {
                continue;
            }
            auto it = std::find_if(masks.begin(), masks.end(), [mask](const MaskInfo & m) { return m.mask == mask; });
            if (it == masks.end()) {
                masks.push_back({mask, cache_k});
            } else if (it->cache_k == nullptr) {
                it->cache_k = cache_k;
            }
        }
        if (masks.size() == 2) {
            const ggml_tensor * a = masks[0].mask;
            const ggml_tensor * b = masks[1].mask;
            const ggml_tensor * ka = masks[0].cache_k;
            const ggml_tensor * kb = masks[1].cache_k;
            // llama.cpp allocates the sliding-window cache n_swa rows and the global one n_ctx rows,
            // so the shorter cache is the sliding-window one. Only the row *count* may differ: the
            // stream count must agree (same batch of sequences), while the row *width* may not (a
            // model can use different KV head counts per layer flavor -- gemma4's SWA rows are 512
            // wide against its global 1024).
            if (ka != nullptr && kb != nullptr) {
                if (ka->ne[2] == kb->ne[2] && ka->ne[1] != kb->ne[1]) {
                    model_params.swa_mask = ka->ne[1] < kb->ne[1] ? a : b;
                }
            } else if (a->ne[1] == b->ne[1] && a->ne[3] == b->ne[3] && a->ne[0] != b->ne[0]) {
                // No cache reachable (the "-fa off" softmax path): the masks themselves differ only
                // in width, and the sliding-window one is the narrower.
                model_params.swa_mask = a->ne[0] < b->ne[0] ? a : b;
            }
        }
        // A mask staged through a CPY reaches attention as the CPY's output, while the graph *input*
        // (what get_graph_input_ov_name is asked about) is its source. Record both identities.
        if (model_params.swa_mask != nullptr && model_params.swa_mask->op == GGML_OP_CPY) {
            model_params.swa_mask_src = model_params.swa_mask->src[0];
        }
    }

    // Every FLASH_ATTN_EXT is visited, not just the first: an iSWA model carries two attention
    // flavors, each with its own ctx_per_seq / attention_size, and its full layer list has to be
    // collected in swa_layers. Fields that are the same for every layer (head geometry, token
    // counts) are simply written repeatedly with the same value.
    bool seen_attn = false;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        auto * node = cgraph->nodes[i];
        std::string name = std::string(node->name);
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            if (is_flat_kv_view(node->src[1]) && is_flat_kv_view(node->src[2])) {
                // Whisper keeps each K/V cache in one flat allocation and passes a logical
                // [head_size, length, heads] VIEW directly to attention. Decoder self-attention
                // has a runtime mask; encoder/cross-attention has a separately tracked fixed fill
                // length and no mask.
                compute_params.input_len = node->src[0]->ne[1];
                compute_params.token_len_per_seq = node->src[0]->ne[1];
                if (node->src[3] != nullptr) {
                    compute_params.attention_size = node->src[1]->ne[1];
                } else {
                    compute_params.attention_size_static = node->src[1]->ne[1];
                }
                continue;
            }
            auto * cache_k_view = attn_kvcache_view(node);
            auto * mask = node->src[3];
            if (cache_k_view == nullptr || mask == nullptr) {
                continue;  // not the KV-cache attention pattern this scan describes
            }
            seen_attn = true;

            model_params.n_heads = node->src[0]->ne[2];
            model_params.n_heads_kv = node->src[1]->ne[2];
            model_params.head_size = node->src[0]->ne[0];
            compute_params.input_len = node->src[0]->ne[1];

            auto * cache_k = cache_k_view->src[0];
            int layer = extract_layer_from_name(cache_k->name);
            const bool is_swa = model_params.is_swa_mask(mask);

            model_params.kv_buffer_ctx_id = ggml_backend_openvino_buffer_get_ctx_id(cache_k->buffer);
            if (is_swa) {
                model_params.swa_layers.push_back(layer);
                model_params.ctx_per_seq_swa = cache_k->ne[1];
                compute_params.attention_size_swa = mask->ne[0];
            } else {
                model_params.ctx_per_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
                compute_params.attention_size = mask->ne[0];
            }

            compute_params.n_seq_active = mask->ne[3];
            auto seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            size_t offset;
            memcpy(&offset, cache_k_view->op_params, sizeof(size_t));
            compute_params.seq_active_start = offset / seq_size;
            compute_params.token_len_per_seq = node->ne[2];
        }
        if (node->op == GGML_OP_ROPE) {
            // The first ROPE op's params drive the shared sin/cos table. gemma4-style models
            // interleave SWA layers (small n_dims / freq_base_swa) with global layers (large n_dims /
            // freq_base); if any two ROPE ops disagree, one shared table cannot serve both (an
            // SWA-width table mismatches a global layer's rope, e.g. rope_sin[128] vs Split[256]), so
            // mixed_rope_params tells the frontend to build sin/cos per op instead.
            if (!seen_rope) {
                memcpy(model_params.rope_params, node->op_params, sizeof(int32_t) * 15);
                seen_rope = true;
            } else if (memcmp(model_params.rope_params, node->op_params, sizeof(int32_t) * 15) != 0) {
                model_params.mixed_rope_params = true;
            }
            // Fallback token count for the "-fa off" (softmax) path, where the FLASH_ATTN_EXT
            // branch that normally sets input_len does not run: inp_pos (ROPE src[1]) has one
            // element per token, so its ne[0] is the prefill token length.
            if (compute_params.input_len == -1 && node->src[1] != nullptr) {
                compute_params.input_len = node->src[1]->ne[0];
            }
        }
    }
    // Recurrent-state (SSM/DeltaNet, qwen3-next) detection: a second whole-graph pass, since the
    // FLASH_ATTN_EXT loop above breaks at layer 0 and these ops live in the linear-attention layers.
    for (int i = 0; i < cgraph->n_nodes; i++) {
        auto * node = cgraph->nodes[i];
        if (node->op == GGML_OP_GATED_DELTA_NET) {
            // Recurrent-state row width, used by the frontend to slice the packed [attn|new_state].
            model_params.state_size = node->src[0]->ne[0];
            if (compute_params.input_len == -1) {
                compute_params.input_len = node->src[0]->ne[2];
                compute_params.token_len_per_seq = node->src[0]->ne[2];
            }
        }
        // In-place SCALE-by-0 that clears a slot block [idx, idx+len) of the recurrent-state cache on
        // sequence reset. When absent (normal decode) the slot is preserved; we emit idx/len as graph
        // inputs so one compiled model covers both. view_src is the cache; src[0] is the sliced view.
        if (node->op == GGML_OP_SCALE && node->view_src != nullptr && is_kvcache(node->view_src, nullptr) &&
            node->src[0] != nullptr && node->view_src->ne[0] != 0) {
            compute_params.cache_rs_reset_len = (int) (ggml_nelements(node) / node->view_src->ne[0]);
            compute_params.cache_rs_reset_idx = (int) (node->src[0]->view_offs / node->view_src->ne[0]);
        }
        if (node->op == GGML_OP_CPY && node->view_src != nullptr && is_kvcache(node->view_src, nullptr) &&
            node->src[0] != nullptr && node->src[1] != nullptr && node->src[1]->op == GGML_OP_VIEW &&
            node->src[1]->view_src == node->view_src) {
            const bool is_conv = is_conv_state_writeback(node);
            const bool is_gdn = node->src[0]->op == GGML_OP_VIEW && node->src[0]->src[0] != nullptr &&
                                node->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            const bool is_extra = node->src[0]->op == GGML_OP_GET_ROWS;

            const ggml_tensor * dest_view = node->src[1];
            const ggml_tensor * cache = node->view_src;
            const size_t row_bytes = cache->ne[0] * ggml_type_size(cache->type);
            if (row_bytes > 0 && (is_conv || is_gdn || is_extra)) {
                ComputeParams::RsWriteback writeback;
                writeback.slot_begin = (int) (dest_view->view_offs / row_bytes);
                if (is_conv) {
                    writeback.src_begin = (int) (node->src[0]->view_offs / node->src[0]->view_src->nb[0]);
                }
                compute_params.rs_writebacks[recurrent_writeback_name(cgraph, node)] = writeback;
            }
            if (is_conv || is_gdn) {
                compute_params.s_copy_active_slot_len = (int) dest_view->ne[1];
            }
        }
    }

    // NPU compiles one static shape, so attention spans the whole preallocated cache rather than the
    // current ubatch's mask width. Applied after the loop: it needs the final ctx_per_seq of BOTH
    // attention flavors, which an iSWA graph only has once every layer has been visited.
    if (is_static && seen_attn) {
        compute_params.attention_size = model_params.ctx_per_seq;
        compute_params.attention_size_swa = model_params.ctx_per_seq_swa;
        compute_params.token_len_per_seq = 1;
    }

    auto * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    // for NPU, output_len is always 1 except for llama-perplexity
    if (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_per_seq * model_params.n_seq;
    model_params.ctx_swa = model_params.ctx_per_seq_swa * model_params.n_seq;
    return {model_params, compute_params};
}

void GgmlOvDecoder::validate_cgraph() const {
    if (m_model_params.n_seq > 1 && m_is_static == true) {
        throw std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_input_shape(const ggml_tensor * op, const ggml_tensor * input) const {
    if (m_naive) {
        return input != nullptr ? ov::PartialShape{get_shape(input)} : ov::PartialShape{get_shape(op)};
    }
    auto name = std::string(input->name);
    ov::PartialShape input_shape;

    if (is_inp_tok(input, op) || is_inp_pos(input, op)) {
        // tokens or positions
        int len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        input_shape = ov::PartialShape{1, 1, 1, len};

    } else if (is_output_idx(input, op)) {
        // output index
        input_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } else if (is_inp_mask(input, op)) {
        // mask
        if (m_is_static) {
            input_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_size : 1, m_model_params.ctx};
        } else {
            input_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } else if (is_kvcache(input, op)) {
        // kvcache: always the stateless layout [1, 1, seq, n_heads_kv * head_size]. Conversion to
        // the OpenVINO stateful (ReadValue/Assign) form is done by the LlamaCppToStateful pass,
        // which only rewires the Parameter/Result into state vars without changing the layout.
        input_shape = ov::PartialShape{get_shape(input)};
        const bool recurrent_cache = strncmp(input->name, "cache_r_l", strlen("cache_r_l")) == 0 ||
                                     strncmp(input->name, "cache_s_l", strlen("cache_s_l")) == 0;
        if (!m_is_static && !recurrent_cache) {
            // do not fix ctx size to make llama-bench work across test params
            input_shape[2] = -1;
        }

    } else if (is_kv_idx(input, op)) {
        // kv update index
        int len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        input_shape = ov::PartialShape{1, 1, 1, len};

    } else {
        input_shape = ov::PartialShape{get_shape(input)};
        // Generic graph-input leaf (e.g. MoE expert-routing index tensors). On the dynamic path,
        // mark its inferred token/sequence axis dynamic so the compiled model is not specialized to
        // the prefill token length (which would break cache reuse on the shorter decode call).
        if (!m_is_static) {
            int d = dynamic_dim_of(input);
            if (d == -1) {
                d = token_axis_from_consumer_views(input);
            }
            // Validate against the subgraph's token count: the dynamic token axis must have that
            // size. The ported stride-based inference can mis-resolve onto a same-shaped static axis
            // (qwen3-next's MoE routing chain resolves ffn_moe_weights_norm-* onto the expert axis of
            // size 10 instead of the token axis of size 5). If the picked axis's size does not match
            // the token count, fall back to picking the axis by size.
            const int tok_len = derived_token_len();
            if (tok_len > 1 && (d == -1 || input->ne[d] != tok_len)) {
                d = token_axis_by_input_len(input);
            }
            if (d != -1) {
                // dynamic_dim_of is in ggml dim order; OV shape is reversed.
                input_shape[GGML_MAX_DIMS - 1 - d] = ov::Dimension::dynamic();
            }
        }
    }
    return input_shape;
}

void GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    auto create_1d_input = [this](const std::string & name, int64_t value, bool force_parameter = false) {
        if (m_is_static && !force_parameter) {
            auto constant =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{value});
            constant->set_friendly_name(name);
            m_model_extra_inputs[name] = constant;
        } else {
            auto param_node = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
            param_node->set_friendly_name(name);
            param_node->output(0).get_tensor().set_names({name});
            m_model_extra_inputs[name] = param_node;

            auto tensor = std::make_shared<ov::Tensor>(ov::element::i64, ov::Shape{1});
            *tensor->data<int64_t>() = value;
            m_model_extra_input_values[name] = tensor;
        }
    };

    // Only surface attention_size / token_len_per_seq when they were actually derived (!= -1).
    // On the "-fa off" softmax path there is no FLASH_ATTN_EXT node, so attention_size stays -1;
    // creating it anyway would feed -1 into permute.cpp's Slice (has_input branch) instead of
    // letting the translator fall back to the statically-known output size.
    if (m_compute_params.attention_size != -1) {
        create_1d_input("attention_size", m_compute_params.attention_size);
    }
    if (m_compute_params.attention_size_static != -1) {
        create_1d_input("attention_size_static", m_compute_params.attention_size_static);
    }
    if (m_compute_params.attention_size_swa != -1) {
        create_1d_input("attention_size_swa", m_compute_params.attention_size_swa);
    }
    create_1d_input("n_seq_active", m_compute_params.n_seq_active);
    create_1d_input("seq_active_start", m_compute_params.seq_active_start);
    create_1d_input("seq_active_end", m_compute_params.seq_active_start + m_compute_params.n_seq_active);
    if (m_compute_params.token_len_per_seq != -1) {
        create_1d_input("token_len_per_seq", m_compute_params.token_len_per_seq);
    }
    // create_1d_input("token_len", m_token_len_per_seq * m_n_seq_active);

    // Recurrent-state (qwen3-next) extra inputs: slot-reset block and active-slot block. Only
    // surfaced when the graph actually carries recurrent state (values != -1), so non-recurrent
    // models are unaffected.
    if (m_compute_params.cache_rs_reset_idx != -1) {
        create_1d_input("cache_rs_reset_idx", m_compute_params.cache_rs_reset_idx, m_is_static);
        create_1d_input("cache_rs_reset_len", m_compute_params.cache_rs_reset_len, m_is_static);
    }
    if (m_compute_params.s_copy_active_slot_len != -1) {
        create_1d_input("s_copy_active_slot_len", m_compute_params.s_copy_active_slot_len);
        if (m_is_static) {
            create_1d_input("chunk_valid_len", m_compute_params.input_len, true);
        }
    }
    for (const auto & [node_name, writeback] : m_compute_params.rs_writebacks) {
        create_1d_input("rs_slot_begin_" + node_name, writeback.slot_begin);
        if (writeback.src_begin != -1) {
            create_1d_input("rs_src_begin_" + node_name, writeback.src_begin);
        }
    }

    // Build the unified map returned by get_model_inputs() (primary + extra).
    m_all_model_inputs = m_model_inputs;
    m_all_model_inputs.insert(m_model_extra_inputs.begin(), m_model_extra_inputs.end());
}

bool GgmlOvDecoder::node_is_used_as_src(const int node_idx) {
    ggml_tensor * node = m_cgraph->nodes[node_idx];
    for (int i = node_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_node = m_cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (other_node->src[j] == node) {
                return true;
            }
        }
    }
    return false;
}

void GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * node = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        if (node->op == GGML_OP_NONE && node_is_used_as_src(i)) {
            std::string node_name(node->name);
            // Weights are surfaced as their own GGML_OP_NONE nodes (tracked in m_weight_names),
            // not model inputs, even though m_model_weights is empty in the node-based weight path.
            if (m_model_weights.find(node_name) == m_model_weights.end() &&
                m_weight_names.find(node_name) == m_weight_names.end()) {
                m_inputs[node_name] = node;
                auto param_node =
                    std::make_shared<ov::op::v0::Parameter>(get_ov_type(node), get_graph_input_shape(node, nullptr));
                param_node->set_friendly_name(node_name);
                param_node->output(0).get_tensor().set_names({node_name});
                m_model_inputs[node_name] = param_node;
            }
            continue;
        }
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = node->src[i];
            if (src == nullptr) {
                continue;
            }
            ggml_tensor * model_input = src;
            if (node->op == GGML_OP_FLASH_ATTN_EXT && is_flat_kv_view(src)) {
                model_input = src->view_src;
            } else if (i == 1 && node->op == GGML_OP_CPY && src->op == GGML_OP_VIEW && src->view_src != nullptr &&
                       is_kvcache(src->view_src, nullptr)) {
                // Recurrent CPY consumers are wired to the full cache allocation in
                // set_input_output(); expose that same allocation as the model Parameter.
                model_input = src->view_src;
            }
            std::string src_name = std::string(model_input->name);
            if (model_input->flags & GGML_TENSOR_FLAG_INPUT) {
                src_name = get_graph_input_ov_name(model_input, node);
            }
            if (m_model_weights.find(src_name) != m_model_weights.end() ||
                m_weight_names.find(src_name) != m_weight_names.end()) {
                continue;
            }

            bool is_intermediate_node = false;
            for (const auto & node_info : m_node_info_list) {
                if (node_info.node == src) {
                    is_intermediate_node = true;
                    break;
                }
            }
            if (is_intermediate_node) {
                continue;
            }
            if (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                continue;
            }
            m_inputs[src_name] = model_input;

            ggml_backend_buffer * buffer = model_input->buffer;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                if (auto it = std::find(m_model_params.kv_names.begin(), m_model_params.kv_names.end(), src_name);
                    it == m_model_params.kv_names.end()) {
                    m_model_params.kv_names.push_back(src_name);
                }
            }
            ov::PartialShape param_shape = get_graph_input_shape(node, model_input);
            auto param_node = std::make_shared<ov::op::v0::Parameter>(get_ov_type(model_input), param_shape);
            param_node->set_friendly_name(src_name);
            param_node->output(0).get_tensor().set_names({src_name});
            m_model_inputs[src_name] = param_node;
        }
    }

    // Injected VIEW-of-external-input nodes (see set_input_output) reference a view_src that is not a
    // direct src of any cgraph node, so the loop above never registered it. Surface each unique
    // view_src as the real input Parameter (full shape, token axis kept dynamic); the injected VIEW
    // slices it per layer. Consumers read the sliced result, not this Parameter.
    std::set<const ggml_tensor *> local_nodes;
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        local_nodes.insert(m_cgraph->nodes[i]);
    }
    for (const auto & ni : m_node_info_list) {
        if (ni.node == nullptr || ni.node->op != GGML_OP_VIEW || ni.node->view_src == nullptr ||
            ni.node_inputs_names.empty() || local_nodes.count(ni.node)) {
            continue;  // only the injected (non-cgraph-node) views
        }
        const ggml_tensor * vsrc = ni.node->view_src;
        const std::string & vsrc_name = ni.node_inputs_names[0];
        if (m_model_inputs.count(vsrc_name)) {
            continue;
        }
        m_inputs[vsrc_name] = const_cast<ggml_tensor *>(vsrc);
        ov::PartialShape vsrc_shape{get_shape(vsrc)};
        if (!m_is_static) {
            int d = token_axis_from_consumer_views(vsrc);
            // Inter-subgraph boundary leaf (Qcur_full-* / Qcur_view-*): recover / validate the token
            // axis by the prefill token count so the Parameter is not specialized to the prefill length
            // (and a mis-resolved consumer-view axis of the wrong size is corrected).
            const int tok_len = derived_token_len();
            if (tok_len > 1 && (d == -1 || vsrc->ne[d] != tok_len)) {
                d = token_axis_by_input_len(vsrc);
            }
            if (d != -1) {
                vsrc_shape[GGML_MAX_DIMS - 1 - d] = ov::Dimension::dynamic();
            }
        }
        auto param_node = std::make_shared<ov::op::v0::Parameter>(get_ov_type(vsrc), vsrc_shape);
        param_node->set_friendly_name(vsrc_name);
        param_node->output(0).get_tensor().set_names({vsrc_name});
        m_model_inputs[vsrc_name] = param_node;
    }
}

void GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_names.clear();
    // Multiple in-place writebacks to the same recurrent-state cache (e.g. several disjoint
    // rollback windows written back within one cgraph) all redirect to the same view_src, so they
    // resolve to the same output name below. Track which names already got a Result so each name
    // is only listed once: translate_session.cpp's tensor_map is name-keyed too, so a name listed
    // more than once would just create redundant Results for the identical (last-write) Output
    // anyway, and OpenVINO auto-disambiguates duplicate Result friendly names, desyncing them from
    // the names this decoder reports through get_model_output_names()/get_model_outputs().
    std::set<std::string> seen_output_names;
    for (int node_n = 0; node_n < m_cgraph->n_nodes; node_n++) {
        auto * cur_node = m_cgraph->nodes[node_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        if (cur_node->op == GGML_OP_NONE) {
            continue;
        }
        auto cur_node_use_count = m_cgraph->use_counts[ggml_hash_find(&m_cgraph->visited_hash_set, cur_node)];
        if (cur_node_use_count == 0) {
            // The output of an in-place op is the view_src tensor, which is updated in place. We use
            // the view_src name as the output name so it matches later ops that use the view_src.
            if (is_inplace_op(cur_node)) {
                cur_node = cur_node->view_src;
            }
        } else {
            int input_use_count = 0;
            for (int i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * node = m_cgraph->nodes[i];
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != NULL && node->src[j] == cur_node) {
                        input_use_count++;
                    }
                }
            }
            if (input_use_count == cur_node_use_count) {
                cur_node = nullptr;
            }
        }
        if (cur_node != nullptr) {
            // Use the same unique name the producer registered so a model output matches its node.
            auto un = m_tensor_unique_name.find(cur_node);
            std::string node_output_name =
                (un != m_tensor_unique_name.end()) ? un->second : std::string(cur_node->name);
            m_model_outputs[node_output_name] = cur_node;
            if (seen_output_names.insert(node_output_name).second) {
                m_model_output_names.push_back(node_output_name);
            }
        }
    }
}

int GgmlOvDecoder::dynamic_dim_of(const ggml_tensor * tensor) const {
    auto it = m_node_dynamic_dims.find(tensor);
    return it == m_node_dynamic_dims.end() ? -1 : it->second;
}

int GgmlOvDecoder::token_axis_from_consumer_views(const ggml_tensor * input) const {
    // Look for a VIEW whose view_src is `input` and which is a select-and-drop of exactly one axis
    // (a per-layer pick). The token axis is then the surviving axis of size>1 other than dim0 (the
    // embedding width). This recovers the dynamic token axis for host-fed tensors like gemma4's
    // per-layer embedding, where the stride-based inference bottoms out at the host CONT.
    for (int node_n = 0; node_n < m_cgraph->n_nodes; node_n++) {
        const ggml_tensor * v = m_cgraph->nodes[node_n];
        if (v->op != GGML_OP_VIEW || v->view_src != input) {
            continue;
        }
        // Identify the single dropped (>1 -> 1) axis.
        int dropped = -1;
        bool is_select_and_drop = false;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (input->ne[d] > 1 && v->ne[d] == 1) {
                int64_t squeezed[GGML_MAX_DIMS];
                int p = 0;
                for (int e = 0; e < GGML_MAX_DIMS; ++e) {
                    if (e != d) {
                        squeezed[p++] = input->ne[e];
                    }
                }
                squeezed[p] = 1;
                bool match = true;
                for (int e = 0; e < GGML_MAX_DIMS; ++e) {
                    if (v->ne[e] != squeezed[e]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    dropped = d;
                    is_select_and_drop = true;
                    break;
                }
            }
        }
        if (!is_select_and_drop) {
            continue;
        }
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {  // skip dim0 (embedding width -- always static)
            if (d != dropped && input->ne[d] > 1) {
                return d;
            }
        }
    }
    return -1;
}

int GgmlOvDecoder::derived_token_len() const {
    int input_len = m_compute_params.input_len;
    if (input_len <= 1) {
        // FFN-only subgraphs (no FLASH_ATTN_EXT / ROPE) never populate input_len, but their other
        // tensors do get a resolved dynamic axis via compute_node_dynamic_dims(); ne[dyn] on any of
        // those is the subgraph's prefill token count. Recover it so a CPU-produced boundary leaf
        // (e.g. ffn_moe_down-*) can be marked dynamic too. All resolved tensors agree on the count.
        for (const auto & kv : m_node_dynamic_dims) {
            if (kv.second != -1 && kv.first != nullptr) {
                const int64_t n = kv.first->ne[kv.second];
                if (n > 1) {
                    input_len = static_cast<int>(n);
                    break;
                }
            }
        }
    }
    return input_len > 1 ? input_len : -1;
}

int GgmlOvDecoder::token_axis_by_input_len(const ggml_tensor * input) const {
    const int input_len = derived_token_len();
    if (input_len <= 1) {
        return -1;  // decode step: token count 1 is ambiguous with static singleton dims
    }
    int match = -1;
    for (int e = 0; e < GGML_MAX_DIMS; ++e) {
        if (input->ne[e] == input_len) {
            if (match != -1) {  // ambiguous -- more than one axis of that size
                return -1;
            }
            match = e;
        }
    }
    return match;
}

void GgmlOvDecoder::compute_node_dynamic_dims() {
    // Per-tensor inference (in ggml dim order 0..3) of the single variable token/sequence axis,
    // propagated stride-based from the token/position/output-index graph inputs through the ops.
    // Ported from the upstream ggml-openvino backend; -1 means the tensor is fully static. The
    // result is folded into the PartialShapes the frontend consumes (dynamic_dim_of()).

    // Pre-seed inter-subgraph boundary leaves (qwen3-next's split graph). A leaf produced on CPU in
    // another subgraph (e.g. Qcur_full-* fed into a full-attention split) has no in-subgraph producer,
    // so the stride-based propagation below never reaches it and its downstream CONT/RESHAPE ops bake
    // the prefill token count into a static reshape target -- which then fails to bind on the shorter
    // decode call. Seed such a leaf's token axis (the unique ggml dim whose size equals the prefill
    // token count) so propagation flows through it. input_len comes from FLASH_ATTN_EXT / ROPE, which
    // these full-attention splits carry; FFN-only splits (input_len == -1) have no such leaf.
    if (m_compute_params.input_len > 1) {
        std::set<const ggml_tensor *> produced;
        for (int i = 0; i < m_cgraph->n_nodes; i++) {
            produced.insert(m_cgraph->nodes[i]);
        }
        for (int i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * node = m_cgraph->nodes[i];
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = node->src[j];
                if (src == nullptr || produced.count(src) || m_node_dynamic_dims.count(src)) {
                    continue;
                }
                int match = -1;
                for (int e = 0; e < GGML_MAX_DIMS; ++e) {
                    if (src->ne[e] == m_compute_params.input_len) {
                        if (match != -1) {  // ambiguous
                            match = -1;
                            break;
                        }
                        match = e;
                    }
                }
                if (match != -1) {
                    m_node_dynamic_dims[src] = match;
                }
            }
        }
    }

    auto visit_node = [&](auto && self, ggml_tensor * node) -> void {
        if (!node) {
            return;
        }
        if (node->op == GGML_OP_CPY) {
            m_node_dynamic_dims[node] = -1;
        }
        if (m_node_dynamic_dims.count(node)) {
            return;
        }
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = node->src[i];
            if (src == nullptr) {
                continue;
            }
            if (is_inp_tok(src, node) || is_inp_pos(src, node) || is_output_idx(src, node)) {
                m_node_dynamic_dims[src] = 0;
                continue;
            }
            if (node->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && !m_is_stateful) {
                m_node_dynamic_dims[src] = 1;
                continue;
            }
            self(self, src);
        }
        switch (node->op) {
        case GGML_OP_NONE:
            m_node_dynamic_dims[node] = -1;
            break;
        case GGML_OP_GET_ROWS:
            m_node_dynamic_dims[node] = -1;
            if (m_node_dynamic_dims[node->src[1]] != -1) {
                auto dynamic_dim_idx = m_node_dynamic_dims[node->src[1]];
                if (dynamic_dim_idx == 0) {
                    m_node_dynamic_dims[node] = 1;
                } else {
                    auto dynamic_dim_stride = node->src[1]->nb[dynamic_dim_idx] / ggml_type_size(node->src[1]->type) *
                                              ggml_type_size(node->src[0]->type);
                    for (int i = 0; i < GGML_MAX_DIMS; i++) {
                        if (dynamic_dim_stride == node->src[0]->nb[i]) {
                            m_node_dynamic_dims[node] = i;
                            break;
                        }
                    }
                }
            }
            break;
        case GGML_OP_MUL:
        case GGML_OP_MUL_MAT:
            m_node_dynamic_dims[node] = -1;
            if (m_node_dynamic_dims[node->src[0]] != -1) {
                m_node_dynamic_dims[node] = m_node_dynamic_dims[node->src[0]];
            }
            if (m_node_dynamic_dims[node->src[1]] != -1) {
                m_node_dynamic_dims[node] = m_node_dynamic_dims[node->src[1]];
            }
            break;
        case GGML_OP_PERMUTE:
            m_node_dynamic_dims[node] = -1;
            if (m_node_dynamic_dims[node->src[0]] != -1) {
                auto dynamic_dim_idx = m_node_dynamic_dims[node->src[0]];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    if (node->op_params[i] == dynamic_dim_idx) {
                        m_node_dynamic_dims[node] = i;
                        break;
                    }
                }
            }
            break;
        case GGML_OP_VIEW: {
            m_node_dynamic_dims[node] = -1;
            if (m_node_dynamic_dims[node->src[0]] != -1) {
                if (node->src[0]->op == GGML_OP_NONE) {
                    m_node_dynamic_dims[node] = m_node_dynamic_dims[node->src[0]];
                    break;
                }
                auto dynamic_dim_idx = m_node_dynamic_dims[node->src[0]];
                auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                auto dynamic_dim_stride =
                    node->src[0]->nb[dynamic_dim_idx] / ggml_type_size(node->src[0]->type) * ggml_type_size(node->type);
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    if (node->nb[i] == dynamic_dim_stride) {
                        m_node_dynamic_dims[node] = i;
                        break;
                    }
                }
                if (m_node_dynamic_dims[node] != -1 && dynamic_dim_value != node->ne[m_node_dynamic_dims[node]]) {
                    m_node_dynamic_dims[node] = -1;
                }
            }
            break;
        }
        case GGML_OP_TRANSPOSE: {
            m_node_dynamic_dims[node] = -1;
            if (m_node_dynamic_dims[node->src[0]] != -1) {
                // GGML_OP_TRANSPOSE swaps ggml dimensions 0 and 1. Propagate the runtime axis by
                // that permutation instead of matching captured extents/strides: an empty token
                // dimension makes adjacent strides equal, and its captured extent (0) need not
                // match the non-empty shape used to build the reusable OpenVINO model.
                const int dynamic_dim_idx = m_node_dynamic_dims[node->src[0]];
                m_node_dynamic_dims[node] = dynamic_dim_idx == 0 ? 1 : dynamic_dim_idx == 1 ? 0 : dynamic_dim_idx;
            }
            break;
        }
        case GGML_OP_RESHAPE: {
            m_node_dynamic_dims[node] = -1;
            if (m_node_dynamic_dims[node->src[0]] != -1) {
                const auto dynamic_dim_idx = m_node_dynamic_dims[node->src[0]];
                auto dynamic_dim_stride = node->src[0]->nb[dynamic_dim_idx];
                int matched_dim_count = 0;
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    if (node->nb[i] == dynamic_dim_stride &&
                        (node->ne[i] == node->src[0]->ne[dynamic_dim_idx] || node->ne[i] == 0 ||
                         node->src[0]->ne[dynamic_dim_idx] == 0)) {
                        m_node_dynamic_dims[node] = i;
                        matched_dim_count++;
                    }
                }
                if (matched_dim_count != 1) {
                    m_node_dynamic_dims[node] = -1;
                }
            }
            break;
        }
        case GGML_OP_FLASH_ATTN_EXT: {
            constexpr int q_to_out[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_node_dynamic_dims[node] = -1;
            if (m_node_dynamic_dims[node->src[0]] != -1) {
                m_node_dynamic_dims[node] = q_to_out[m_node_dynamic_dims[node->src[0]]];
            }
            break;
        }
        case GGML_OP_CONT:
            m_node_dynamic_dims[node] = -1;
            if (m_node_dynamic_dims[node->src[0]] != -1) {
                auto dynamic_dim_idx = m_node_dynamic_dims[node->src[0]];
                if (ggml_are_same_shape(node, node->src[0])) {
                    m_node_dynamic_dims[node] = dynamic_dim_idx;
                } else {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_size(node->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (node->src[0]->ne[0] / ggml_blck_size(node->src[0]->type));
                    for (int i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * node->src[0]->ne[i - 1];
                    }
                    auto dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_size(node->src[0]->type) *
                                              ggml_type_size(node->type);
                    int matched_dim_count = 0;
                    for (int i = 0; i < GGML_MAX_DIMS; i++) {
                        if (node->nb[i] == dynamic_dim_stride && node->ne[i] == node->src[0]->ne[dynamic_dim_idx]) {
                            m_node_dynamic_dims[node] = i;
                            matched_dim_count++;
                        }
                    }
                    if (matched_dim_count != 1) {
                        // No output dim exactly matches the token axis: ggml_cont MERGED the token dim
                        // into a larger contiguous output dim (qwen3-next's a/b: src [tok,8,2,1] ->
                        // [8*tok,2,1,1], token folded into output dim 0). The token no longer has its
                        // own axis, but the output dim into which it merged still VARIES with the token
                        // count and must be dynamic. Locate it as the output dim whose contiguous
                        // logical-stride range [nb[i], nb[i+1]) contains the token's stride.
                        m_node_dynamic_dims[node] = -1;
                        for (int i = 0; i < GGML_MAX_DIMS; i++) {
                            const size_t lo = node->nb[i];
                            const size_t hi = (i + 1 < GGML_MAX_DIMS) ? node->nb[i + 1] : SIZE_MAX;
                            if (dynamic_dim_stride >= lo && dynamic_dim_stride < hi) {
                                m_node_dynamic_dims[node] = i;
                                break;
                            }
                        }
                    }
                }
            }
            break;
        case GGML_OP_CONCAT:
            // The token axis is the one CONCAT does NOT grow; find the first dim that differs.
            m_node_dynamic_dims[node] = -1;
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                if (node->src[0]->ne[d] != node->ne[d]) {
                    m_node_dynamic_dims[node] = d;
                    break;
                }
            }
            break;
        case GGML_OP_SSM_CONV:
        case GGML_OP_GATED_DELTA_NET:
            // Recurrent linear-attention ops: token axis is ggml dim 1 (the sequence).
            m_node_dynamic_dims[node] = 1;
            break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_DIV:
            // Elementwise binary: the token axis is whichever operand carries it. Prefer src[0], but
            // fall back to src[1] -- a residual add can have the (static) recurrent/GDN branch as src[0]
            // and the dynamic-token residual as src[1]; taking only src[0] drops the token axis and
            // desyncs later layers (qwen3-next: a-0 dynamic but a-1 static). Both operands share the
            // same token axis when both are dynamic, so src[0] wins by default.
            m_node_dynamic_dims[node] = m_node_dynamic_dims[node->src[0]];
            if (m_node_dynamic_dims[node] == -1 && node->src[1] != nullptr) {
                m_node_dynamic_dims[node] = m_node_dynamic_dims[node->src[1]];
            }
            break;
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_NORM:
        case GGML_OP_GLU:
        case GGML_OP_ROPE:
        case GGML_OP_SCALE:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ARGSORT:
        case GGML_OP_ADD_ID:
        case GGML_OP_UNARY:
        case GGML_OP_CUMSUM:
        case GGML_OP_FILL:
        case GGML_OP_SET:
        case GGML_OP_DIAG:
        case GGML_OP_TRI:
        case GGML_OP_REPEAT:
        case GGML_OP_POOL_2D:
        case GGML_OP_ROLL:
            m_node_dynamic_dims[node] = m_node_dynamic_dims[node->src[0]];
            break;
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_SOLVE_TRI:
            m_node_dynamic_dims[node] = m_node_dynamic_dims[node->src[1]];
            break;
        case GGML_OP_CPY:
            m_node_dynamic_dims[node] = -1;
            break;
        case GGML_OP_SET_ROWS:
            // KV SET_ROWS returns the full cache. Its sequence extent is runtime-dependent in the
            // reusable dynamic model and becomes append-grown after LlamaCppToStateful. Preserve
            // ggml dim 1 through the following cache VIEW/RESHAPE instead of baking ctx_per_seq.
            m_node_dynamic_dims[node] = node->src[2] != nullptr && is_kvcache(node->src[2], nullptr) ? 1 : -1;
            break;
        default:
            // Explicitly mark unhandled ops static instead of leaving the node absent from the map.
            // A later op reads m_node_dynamic_dims[src] with operator[], which would otherwise
            // auto-insert 0 for a missing key -- falsely marking ggml dim0 dynamic and, after a
            // reshape, resolving the wrong axis (qwen3-next's DIV-fed ffn_moe_weights_norm picked the
            // expert axis instead of the token axis, baking the token count into the boundary output).
            m_node_dynamic_dims[node] = -1;
            break;
        }
    };
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        visit_node(visit_node, m_cgraph->nodes[i]);
    }
}

const ggml_tensor * GgmlOvDecoder::get_tensor_used_op(const ggml_tensor * tensor) const {
    if (tensor == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        const auto * node = m_cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] == tensor) {
                return node;
            }
        }
    }
    return nullptr;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() const {
    std::map<std::string, std::string> kv_param_res_names;
    const auto swa_names = get_swa_kv_names();
    for (const auto & name : m_model_params.kv_names) {
        // A sliding-window cache stays stateless, so it has no state to look up here.
        if (swa_names.count(name) == 0) {
            kv_param_res_names[name] = name;
        }
    }
    return kv_param_res_names;
}

std::set<std::string> GgmlOvDecoder::get_swa_kv_names() const {
    std::set<std::string> names;
    if (m_model_params.swa_layers.empty()) {
        return names;
    }
    for (const auto & name : m_model_params.kv_names) {
        if (is_swa_layer(extract_layer_from_name(name))) {
            names.insert(name);
        }
    }
    return names;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_node(ggml_tensor * tensor, bool naive) {
    const bool is_ov_buffer = ggml_backend_buffer_is_openvino(tensor->buffer);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    if (tensor->extra) {
        OPENVINO_ASSERT(is_ov_buffer, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(tensor->extra);

        if (extra_base->type == ggml_openvino_extra_base::Type::WEIGHT) {
            // F16/F32/BF16 weight with shared-memory constant
            auto * weight_extra = static_cast<ggml_openvino_weight_extra *>(tensor->extra);
            if (weight_extra->weight_node) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                return weight_extra->weight_node;
            }
        } else if (extra_base->type == ggml_openvino_extra_base::Type::QUANTIZED_WEIGHT) {
            // Quantized weight with pre-extracted data
            auto * quant_extra = static_cast<ggml_openvino_quantized_weight_extra *>(tensor->extra);
            if (quant_extra->weight_node) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                return quant_extra->weight_node;
            }
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    static const std::set<ggml_type> weight_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16,
                                                     GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
                                                     GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};
    if (weight_types.find(tensor->type) == weight_types.end()) {
        throw std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    OvWeight ov_weight;
    if (ggml_is_quantized(tensor->type)) {
        auto use_bias = naive;
        if (is_ov_buffer) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_size = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_size);
            memcpy(tmp.data(), tensor->data, raw_size);
            ov_weight = process_weight_tensor(tensor, tmp.data(), tensor->data, use_bias);
        } else {
            ov_weight = process_weight_tensor(tensor, tensor->data, nullptr, use_bias);
        }
    } else {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_weight = process_weight_tensor(tensor, tensor->data, tensor->data);
    }

    ov_weight.weight_node->set_friendly_name(tensor->name);
    if (!is_ov_buffer) {
        return ov_weight.weight_node;
    }

    ggml_openvino_extra_base * extra;
    if (ov_weight.is_quantized()) {
        extra = new ggml_openvino_quantized_weight_extra(std::move(ov_weight.weights), std::move(ov_weight.scales),
                                                         std::move(ov_weight.zp), ov_weight.weight_node);
    } else {
        extra = new ggml_openvino_weight_extra(std::move(ov_weight.weights), ov_weight.weight_node);
    }
    ggml_openvino_buffer_register_extra(tensor, extra);

    return ov_weight.weight_node;
}

void GgmlOvDecoder::dump_cgraph(const ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    file << "=== GRAPH ===\n";

    // clang-format off
    file << "n_nodes = " << cgraph->n_nodes << "\n";
    file << " " << std::setw(3) << "nodes"
                <<  std::setw(15) << "shape"
                << std::setw(20) << "op"
                << std::setw(20) << "name"
                << std::setw(3) << "    "
                << std::setw(62) << "stride"
                << std::setw(20) << "buffer_type"
                << "\n";
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        // Get buffer type name
        const char * buf_name = "none";
        ggml_backend_buffer_t buf = node->view_src ? node->view_src->buffer : node->buffer;
        if (buf) {
            buf_name = ggml_backend_buffer_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << node->ne[0] << ", "
             << std::setw(5) << node->ne[1] << ", "
             << std::setw(5) << node->ne[2] << ", "
             << std::setw(5) << node->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(node->op) << std::right << " "
             << std::left << std::setw(45) << node->name << std::right
             << std::setw(2) << "[ "
             << std::setw(0) << node->nb[0] << ", "
             << std::setw(5) << node->nb[1] << ", "
             << std::setw(5) << node->nb[2] << ", "
             << std::setw(5) << node->nb[3] << "] "
             << std::right << std::setw(15) << buf_name << std::right
             << "\n";

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (auto* src = node->src[i]) {
                // Get buffer type name for source
                const char * src_buf_name = "none";
                ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
                if (src_buf) {
                    src_buf_name = ggml_backend_buffer_name(src_buf);
                }

                file << std::setw(10) << " [ "
                << std::setw(5) << src->ne[0] << ", "
                << std::setw(5) << src->ne[1] << ", "
                << std::setw(5) << src->ne[2] << ", "
                << std::setw(5) << src->ne[3] << "] "
                << std::setw(12)
                << i << ": " << std::left << std::setw(12) << ggml_op_name(src->op) << std::right;
                file << std::left << std::setw(30) << src->name << std::right
                << std::setw(16) << "[ "
                << std::setw(0) << src->nb[0] << ", "
                << std::setw(5) << src->nb[1] << ", "
                << std::setw(5) << src->nb[2] << ", "
                << std::setw(5) << src->nb[3] << "] "
                << std::right << std::setw(15) << src_buf_name << std::right
                << "\n";
            }
        }
    }

    file << "n_leafs = " << cgraph->n_leafs << "\n";
    for (int i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * node = cgraph->leafs[i];

        // Get buffer type name for leaf
        const char * leaf_buf_name = "none";
        ggml_backend_buffer_t leaf_buf = node->view_src ? node->view_src->buffer : node->buffer;
        if (leaf_buf) {
            leaf_buf_name = ggml_backend_buffer_name(leaf_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << node->ne[0] << ", "
             << std::setw(5) << node->ne[1] << "] "
             << std::setw(8) << ggml_op_name(node->op) << " "
             << std::setw(16) << ggml_get_name(node)
             << std::setw(20) << leaf_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

void print_tensor_address_map(const ggml_cgraph * cgraph) {
    std::map<void *, std::vector<std::string>> address_map;
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        auto * node = cgraph->nodes[node_n];
        if (node->data) {
            auto it = address_map.find(node->data);
            if (it == address_map.end()) {
                address_map[node->data] = std::vector<std::string>();
            }
            address_map[node->data].push_back(node->name);
        }
    }
    for (const auto & pair : address_map) {
        std::cout << "Address: " << pair.first << std::endl;
        for (const auto & name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << std::endl << std::endl;
    }
}

ov::Shape GgmlOvDecoder::get_shape(const ggml_tensor * tensor) {
    std::vector<size_t> shape;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    return shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(const ggml_tensor * tensor) {
    std::vector<size_t> stride;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    return stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(const ggml_tensor * tensor) {
    switch (tensor->type) {
    case GGML_TYPE_F64:
        return ov::element::f64;
    case GGML_TYPE_F32:
        return ov::element::f32;
    case GGML_TYPE_F16:
        return ov::element::f16;
    case GGML_TYPE_BF16:
        return ov::element::bf16;
    case GGML_TYPE_I8:
        return ov::element::i8;
    case GGML_TYPE_I16:
        return ov::element::i16;
    case GGML_TYPE_I32:
        return ov::element::i32;
    case GGML_TYPE_I64:
        return ov::element::i64;
    default:
        return ov::element::dynamic;
    }
}

size_t GgmlOvDecoder::get_input_size() const {
    // Model scope: number of model inputs. Node scope: number of this node's inputs.
    return m_node_idx == -1 ? m_model_inputs.size() : m_node_info_list[m_node_idx].node_inputs_names.size();
}

std::vector<std::string> GgmlOvDecoder::get_input_names() const {
    return m_node_info_list[m_node_idx].node_inputs_names;
}

ov::PartialShape GgmlOvDecoder::get_output_shape() const {
    auto * ggml_tensor = m_node_info_list[m_node_idx].node_output;
    if (m_weight_names.count(m_node_info_list[m_node_idx].node_name)) {
        // MoE expert weights are 3D in ggml ([k, m, n_expert]); keep the expert dimension so
        // MUL_MAT_ID sees rank-4 [1, n_expert, m, k]. get_shape() reverses ne[] to exactly that.
        if (ggml_tensor->ne[2] > 1) {
            return ov::PartialShape(get_shape(ggml_tensor));
        }
        // Regular 2D weight: the logical [rows, cols] the frontend's dequant path expects.
        return ov::PartialShape({static_cast<int64_t>(ggml_tensor->ne[1]), static_cast<int64_t>(ggml_tensor->ne[0])});
    }
    return ov::PartialShape(get_shape(ggml_tensor));
}

ov::PartialShape GgmlOvDecoder::get_input_shape(const std::string & name) const {
    return ov::PartialShape(get_shape(m_node_info_list[m_node_idx].node_inputs.at(name)));
}

std::vector<std::string> GgmlOvDecoder::get_output_names() const {
    return {m_node_info_list[m_node_idx].node_output_name};
}

const std::string & GgmlOvDecoder::get_op_name() const {
    static const std::string unknown_name = "UNKNOWN_OP_NAME";
    return m_node_idx == -1 ? unknown_name : m_node_info_list[m_node_idx].node_name;
}

ov::frontend::gguf::RopeConfig GgmlOvDecoder::get_rope_config() const {
    // rope_params layout (see ggml_compute_forward_rope / set in compute_llm_params):
    //   [1]=n_dims, [4]=n_ctx_orig, [5]=freq_base, [6]=freq_scale, [7]=ext_factor,
    //   [8]=attn_factor, [9]=beta_fast, [10]=beta_slow (floats bit-stored in the int32 array).
    // rope_params holds floats bit-stored in the int32 array; read them back as floats. The array
    // is int32-aligned, so a reinterpret load is well-defined (same as ggml's own op-param reads).
    //
    // When this config is requested for a specific ROPE node (per-node decoder), read THAT node's
    // own op_params. This matters for gemma4-style mixed models where SWA and global layers have
    // different n_dims/freq_base: a shared first-rope config would make global layers (n_dims 512)
    // reuse the SWA sin/cos table (n_dims 256), producing a 256-vs-128 rope_sin broadcast mismatch.
    // The model-scoped decoder (m_node_idx == -1, used only when per_op is false) keeps the shared
    // params. This mirrors the vendored frontend's per-op get_output_op_params().
    const int32_t * rp = m_model_params.rope_params;
    if (m_node_idx != -1) {
        const ggml_tensor * node = m_node_info_list[m_node_idx].node;
        if (node != nullptr && node->op == GGML_OP_ROPE) {
            rp = node->op_params;
        }
    }
    const float * fp = reinterpret_cast<const float *>(rp);
    ov::frontend::gguf::RopeConfig cfg;
    cfg.n_dims = rp[1];
    cfg.n_ctx_orig = rp[4];
    cfg.freq_base = fp[5];
    cfg.freq_scale = fp[6];
    cfg.ext_factor = fp[7];
    cfg.attn_factor = fp[8];
    cfg.beta_fast = fp[9];
    cfg.beta_slow = fp[10];
    // gemma4-style mixed-layer models: each ROPE op has its own config, so the frontend skips the
    // shared sin/cos precompute and lets translate_rope build sin/cos per op.
    cfg.per_op = m_model_params.mixed_rope_params;
    // op_params[2] holds the rope mode; IMROPE (qwen3.5/qwen3vl) needs the imrope sin/cos layout in
    // the shared table too, not just per-op. rope_params (op_params[2]) is int, read directly.
    cfg.is_imrope = (rp[2] == GGML_ROPE_TYPE_IMROPE);
    return cfg;
}

int64_t GgmlOvDecoder::get_input_view_element_offset(const std::string & name) const {
    // A VIEW's start offset is stored in op_params[0..1] as a byte count.
    // Divide by nb[0] (bytes per element) to return an element count.
    const ggml_tensor * src = m_node_info_list[m_node_idx].node_inputs.at(name);
    size_t byte_offset = 0;
    memcpy(&byte_offset, src->op_params, sizeof(size_t));
    return static_cast<int64_t>(byte_offset / src->nb[0]);
}

ov::Any GgmlOvDecoder::get_attribute(const std::string & name) const {
    // Model-level attributes (available on both the model-scoped decoder and any node decoder).
    if (name == "rope_config") {
        return get_rope_config();
    }
    if (m_node_idx == -1) {
        return {};  // model-scoped decoder has no node-specific attributes
    }
    const auto & info = m_node_info_list[m_node_idx];

    // Weight nodes: expose raw bytes ("data") + ggml quant type name ("quant_type"). The "data"
    // attribute is what marks a GGML_OP_NONE leaf as a weight for the frontend; the frontend
    // dequantizes / requantizes from it, so the decoder never builds OV weight nodes.
    if (m_weight_names.count(info.node_name)) {
        const ggml_tensor * t = info.node_output;
        if (name == "data") {
            return ov::Tensor(ov::element::u8, ov::Shape{ggml_nbytes(t)}, t->data);
        }
        if (name == "quant_type") {
            return std::string(ggml_type_name(t->type));
        }
        return {};
    }

    // Scalar op parameters, read from the node's op_params via ggml's own accessors (same
    // int32/float layout the translators previously read by hand).
    if (name == "eps") {
        // GGML_OP_RMS_NORM / GGML_OP_NORM / GGML_OP_L2_NORM all store epsilon at op_params[0].
        return ggml_get_op_params_f32(info.node, 0);
    }
    if (name == "scale") {
        return ggml_get_op_params_f32(info.node, 0);
    }
    if (name == "clamp_min") {
        return ggml_get_op_params_f32(info.node, 0);
    }
    if (name == "clamp_max") {
        return ggml_get_op_params_f32(info.node, 1);
    }
    if (name == "concat_axis") {
        // GGML_OP_CONCAT stores the (ggml-order) concat dimension in op_params[0].
        return ggml_get_op_params_i32(info.node, 0);
    }
    if (name == "sort_order") {
        // GGML_OP_ARGSORT: map ggml's sort-order enum to a plain int (0 = ASC, 1 = DESC) so the
        // frontend translator needs no ggml headers.
        return static_cast<int>(ggml_get_op_params_i32(info.node, 0) == GGML_SORT_ORDER_DESC ? 1 : 0);
    }
    if (name == "tri_type") {
        // GGML_OP_TRI stores the ggml_tri_type enum in op_params[0]:
        //   0 = UPPER_DIAG (col >= row), 1 = UPPER (col > row),
        //   2 = LOWER_DIAG (col <= row), 3 = LOWER (col < row).
        return ggml_get_op_params_i32(info.node, 0);
    }
    if (name == "fill_value") {
        // GGML_OP_FILL stores the scalar fill constant as a float in op_params[0].
        return ggml_get_op_params_f32(info.node, 0);
    }
    if (name == "gdn_state_slots") {
        // GGML_OP_GATED_DELTA_NET stores K, the recurrent-state snapshot count, in op_params[0].
        return static_cast<int64_t>(ggml_get_op_params_i32(info.node, 0));
    }
    if (name == "gdn_packed_width") {
        const ggml_tensor * src = info.node->src[0];
        if (src != nullptr && src->op == GGML_OP_VIEW) {
            src = src->src[0];
        }
        if (src != nullptr && src->op == GGML_OP_GATED_DELTA_NET) {
            return static_cast<int64_t>(src->ne[0]);
        }
        return static_cast<int64_t>(-1);
    }
    if ((name == "flat_kv_shape_k" || name == "flat_kv_shape_v") && info.node->op == GGML_OP_FLASH_ATTN_EXT) {
        const int index = name == "flat_kv_shape_k" ? 1 : 2;
        const ggml_tensor * view = info.node->src[index];
        if (!is_flat_kv_view(view)) {
            return std::vector<int64_t>{};
        }
        return std::vector<int64_t>{view->ne[3], view->ne[2], view->ne[1], view->ne[0]};
    }
    if ((name == "flat_kv_offset_k" || name == "flat_kv_offset_v") && info.node->op == GGML_OP_FLASH_ATTN_EXT) {
        const int index = name == "flat_kv_offset_k" ? 1 : 2;
        const ggml_tensor * view = info.node->src[index];
        if (!is_flat_kv_view(view)) {
            return static_cast<int64_t>(0);
        }
        return static_cast<int64_t>(view->view_offs / view->nb[0]);
    }
    if (name == "rs_writeback_name") {
        return recurrent_writeback_name(m_cgraph, info.node);
    }
    if (name == "cpy_output_offset_elems") {
        const size_t elem_size = ggml_type_size(info.node->type);
        return static_cast<int64_t>(elem_size == 0 ? 0 : info.node->view_offs / elem_size);
    }
    if (name == "set_offset_elems") {
        // GGML_OP_SET writes src[1] into a contiguous region of src[0]'s flattened buffer. Its
        // op_params are { nb1, nb2, nb3, offset_bytes, inplace }; op_params[3] is the destination
        // byte offset. Return it in elements (offset_bytes / bytes-per-element) so the frontend never
        // touches raw ggml strides. The SET result shares src[0]'s type/layout, so nb[0] is the
        // element size.
        const ggml_tensor * node = info.node;
        const size_t offset_bytes = static_cast<size_t>(ggml_get_op_params_i32(node, 3));
        const size_t elem_size = node->nb[0];
        OPENVINO_ASSERT(elem_size != 0 && offset_bytes % elem_size == 0,
                        "GGML_OP_SET offset must be aligned to the destination element size");
        return static_cast<int64_t>(offset_bytes / elem_size);
    }
    if (name == "bias") {
        return ggml_get_op_params_f32(info.node, 1);
    }
    if (name == "max_bias") {
        return ggml_get_op_params_f32(info.node, 1);
    }
    if (name == "logit_softcap") {
        return ggml_get_op_params_f32(info.node, 2);
    }
    if (name == "swapped") {
        return ggml_get_op_params_i32(info.node, 1) != 0;
    }
    if (name == "glu_alpha") {
        // GGML_GLU_OP_SWIGLU_OAI (gpt-oss): alpha at op_params[2].
        return ggml_get_op_params_f32(info.node, 2);
    }
    if (name == "glu_limit") {
        // GGML_GLU_OP_SWIGLU_OAI and GGML_GLU_OP_SWIGLU_CLAMP: limit at op_params[3].
        return ggml_get_op_params_f32(info.node, 3);
    }
    if (name == "pool_params") {
        return std::vector<int32_t>{ggml_get_op_params_i32(info.node, 1), ggml_get_op_params_i32(info.node, 2),
                                    ggml_get_op_params_i32(info.node, 3), ggml_get_op_params_i32(info.node, 4),
                                    ggml_get_op_params_i32(info.node, 5), ggml_get_op_params_i32(info.node, 6)};
    }
    if (name == "roll_shifts") {
        return std::vector<int64_t>{ggml_get_op_params_i32(info.node, 3), ggml_get_op_params_i32(info.node, 2),
                                    ggml_get_op_params_i32(info.node, 1), ggml_get_op_params_i32(info.node, 0)};
    }
    if (name == "solve_tri_params") {
        return std::vector<int32_t>{1, 1, 0};
    }
    if (name == "dynamic_cache") {
        // Cache relayouts must retain the runtime sequence dimension for the reusable CPU/GPU
        // model. NPU's forced-static prefill/decode models intentionally keep their fixed extent.
        return !m_is_static;
    }
    if (name == "view_interleaved") {
        // GGML_OP_VIEW op_case 5 (qwen3.5 interleaved Q/gate split): return
        // {group_count, group_stride, inner_off, inner_len} in ggml-dim0 element units. translate_view
        // reshapes the contiguous src [F,T] to expose the per-group stride, slices the inner sub-window,
        // and reshapes to the view's own per-head layout (keeping the token axis dynamic). Empty on
        // no match (should not happen: op_case 5 is only set when detect_view_interleaved_window holds).
        auto iw = detect_view_interleaved_window(info.node);
        if (!iw.valid) {
            return std::vector<int64_t>{};
        }
        return std::vector<int64_t>{iw.group_count, iw.group_stride, iw.inner_off, iw.inner_len};
    }
    if (name == "gdn_view") {
        // GGML_OP_VIEW op_case 4: {part, s_v}. The GATED_DELTA_NET output packs T attention rows on
        // top of S_v recurrent-state rows. part=0 => attn_output (first T rows, byte_offset 0);
        // part=1 => new_state (last S_v rows, byte_offset > 0). s_v is the (static) state row count.
        const ggml_tensor * node = info.node;
        const ggml_tensor * gdn = node->src[0];
        size_t byte_offset = 0;
        std::memcpy(&byte_offset, node->op_params, sizeof(size_t));
        // The packed row axis is ggml dim1 of the GDN output ([S_v*H_v, T + S_v]); its tail S_v rows
        // are the state. new_state->ne is [S_v, S_v, H_v, 1] so S_v = gdn->ne[1] - T. Recover S_v from
        // the state view's own element count when part==1; for part==0 it equals gdn->ne[1] minus the
        // attn token rows. Use the GDN's second output view (new_state) shape: state rows = S_v.
        const int64_t packed_rows = gdn->ne[1];
        int64_t s_v = 0;
        const int64_t part = (byte_offset == 0) ? 0 : 1;
        const int64_t packed_width = gdn->ne[0];
        if (packed_width > 0) {
            const int64_t view_rows = static_cast<int64_t>(ggml_nelements(node)) / packed_width;
            s_v = part == 0 ? packed_rows - view_rows : view_rows;
        } else {
            s_v = packed_rows;
        }
        return std::vector<int64_t>{part, s_v};
    }
    if (name == "view_slice") {
        // GGML_OP_VIEW op_case 3: describe the view as a single-axis Slice of its base tensor,
        // returned as {ov_axis, start, len} in OV dim order (translate_view then Reshapes to the
        // view's own output shape). Two patterns:
        //   * pure shrink of one dim at offset 0 (ffn_moe_topk: 64->8), and
        //   * select-one-index-and-drop-a-dim (per-expert ffn_moe_weighted: [2048,8,tok] pick
        //     expert e -> [2048,tok]). This is a strided reinterpretation, so it can't be found by
        //     element-wise ne comparison; instead find the src dim d that, when sliced to 1 and
        //     squeezed, reproduces node->ne. The byte offset / nb[d] gives the selected index.
        // Empty vector => no sliceable axis (translate_view falls back to a plain Reshape).
        const ggml_tensor * node = info.node;
        const ggml_tensor * src = node->src[0];
        if (src == nullptr) {
            return std::vector<int64_t>{};
        }
        size_t byte_offset = 0;
        std::memcpy(&byte_offset, node->op_params, sizeof(size_t));

        // qwen3-next feature-window: slice a contiguous [off, off+len) sub-block of ggml dim0
        // (OV innermost axis, rank-1) out of the mixed [F, tok] source. view_reshape then rearranges
        // the sliced [len, tok] into the view's per-head layout.
        {
            auto win = detect_view_feature_window(node);
            if (win.first >= 0) {
                const int64_t ov_axis = GGML_MAX_DIMS - 1;  // ggml dim0
                int64_t start = win.first;
                const int64_t len = win.second;
                // conv_state_last is the last d_conv-1 columns of the conv window (src dim0), whose
                // length ncs = n_t + d_conv-1 is DYNAMIC (n_t differs prefill vs decode). A tail slice
                // (start + len == compile-time ncs) must be end-anchored, else the baked absolute
                // start over-reads by (prefill_ncs - decode_ncs) and OV clamps conv_state_last to
                // d_conv-2 columns -> the writeback reshape fails. Encode start = -len so
                // translate_view emits a negative-begin Slice that always takes the last len columns.
                if (start > 0 && start + len == node->src[0]->ne[0]) {
                    start = -len;
                }
                return std::vector<int64_t>{ov_axis, start, len};
            }
        }

        // Pure shrink: exactly one dim differs, same rank/order (topk 64->8). Distinguished from a
        // select-and-drop by strides: a pure shrink keeps the source strides verbatim (it is just a
        // contiguous prefix along one axis), whereas a select-and-drop recomputes them (a dim is
        // removed and the trailing dims shift into its place). This stride check is essential when
        // the shapes alone are ambiguous -- in the warmup graph n_tokens == n_expert_used, so a
        // per-expert view [F,used,tok] -> [F,tok] looks shape-identical to shrinking the token axis.
        int diff = 0, diff_dim = -1;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (node->ne[i] != src->ne[i]) {
                ++diff;
                diff_dim = i;
            }
        }
        bool strides_preserved = true;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (node->nb[i] != src->nb[i]) {
                strides_preserved = false;
                break;
            }
        }
        if (diff == 1 && strides_preserved && diff_dim >= 0 && src->nb[diff_dim] > 0 &&
            byte_offset % src->nb[diff_dim] == 0) {
            // Shrink one dim, keeping parent strides. The byte offset (a clean multiple of that dim's
            // stride) gives the start index: 0 for a contiguous prefix (topk 64->8), or > 0 for a
            // contiguous sub-range (conv_state_last: last d_conv-1 columns).
            int64_t ov_axis = (GGML_MAX_DIMS - 1) - diff_dim;
            int64_t start = static_cast<int64_t>(byte_offset / src->nb[diff_dim]);
            const int64_t len = node->ne[diff_dim];
            return std::vector<int64_t>{ov_axis, start, len};
        }

        // Select-and-drop: try each src dim d as the one collapsed to a single index.
        // Ambiguity trap: when two src dims share the same extent (e.g. the warmup graph has
        // n_tokens == n_expert_used == 2, so ffn_moe_weighted src is [F, n_used=2, tok=2, 1]),
        // the squeeze test alone matches BOTH the expert axis and the token axis. The byte offset
        // disambiguates: a per-expert view strides along the expert axis, so offset is an exact
        // multiple of that axis' stride (offset = expert_index * nb[expert_dim]) and lands inside
        // it, whereas it is NOT a clean multiple of the other candidate's stride. Prefer the
        // offset-consistent axis; on a further tie (offset 0) prefer the inner (smaller d) axis,
        // which is the expert-used dim in ggml's MoE layout.
        int best_d = -1;
        for (int d = GGML_MAX_DIMS - 1; d >= 0; --d) {
            if (src->ne[d] <= 1 || src->nb[d] <= 0) {
                continue;
            }
            // Build src->ne with dim d removed and a trailing 1 appended, then compare to node->ne.
            int64_t squeezed[GGML_MAX_DIMS];
            int p = 0;
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                if (i == d) {
                    continue;
                }
                squeezed[p++] = src->ne[i];
            }
            squeezed[p] = 1;
            bool match = true;
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                if (node->ne[i] != squeezed[i]) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                continue;
            }
            // Offset must index cleanly inside this axis to be the selected one.
            const bool offset_consistent =
                (byte_offset % src->nb[d] == 0) && (static_cast<int64_t>(byte_offset / src->nb[d]) < src->ne[d]);
            if (!offset_consistent) {
                continue;
            }
            // Ascending sweep would be simpler, but we go high->low and just keep the smallest
            // qualifying d so the expert-used (inner) axis wins the offset-0 tie.
            best_d = d;
        }
        if (best_d >= 0) {
            int64_t start = static_cast<int64_t>(byte_offset / src->nb[best_d]);
            int64_t ov_axis = (GGML_MAX_DIMS - 1) - best_d;
            return std::vector<int64_t>{ov_axis, start, 1};
        }
        return std::vector<int64_t>{};
    }
    if (name == "view_offset_bytes") {
        size_t byte_offset = 0;
        std::memcpy(&byte_offset, info.node->op_params, sizeof(size_t));
        return static_cast<int64_t>(byte_offset);
    }
    if (name == "view_input_strides") {
        std::vector<int64_t> strides(GGML_MAX_DIMS);
        const ggml_tensor * src = info.node->src[0];
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            strides[i] = static_cast<int64_t>(src->nb[i]);
        }
        return strides;
    }
    if (name == "view_output_strides") {
        std::vector<int64_t> strides(GGML_MAX_DIMS);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            strides[i] = static_cast<int64_t>(info.node->nb[i]);
        }
        return strides;
    }
    if (name == "view_reshape") {
        // GGML_OP_VIEW op_case 3 target shape (OV dim order = ggml ne[] reversed). Only needed for
        // the select-and-drop case, where the Slice keeps a singleton on the collapsed source axis
        // but the view's own layout drops it and shifts the remaining dims (e.g. select-expert:
        // sliced [1,tok,1,F] -> view [1,1,tok,F], token moves from OV axis1 to axis2). Pure-shrink
        // views need no reshape (the slice already has the right layout) -> return empty.
        //
        // The token axis stays dynamic: mark it -1 in OV order. The token axis is the source dim
        // carrying the dynamic (token) count; find it via dynamic_dim_of(src) mapped through the
        // drop, falling back to the output's own inferred dynamic dim.
        const ggml_tensor * node = info.node;
        const ggml_tensor * src = node->src[0];
        if (src == nullptr) {
            return std::vector<int64_t>{};
        }
        // GDN packed-output view (op_case 4): return the view's own ggml layout in OV order.
        // translate_view reshapes the sliced rows into it (placing -1 on the token axis for attn).
        if (src->op == GGML_OP_GATED_DELTA_NET) {
            std::vector<int64_t> tgt(GGML_MAX_DIMS);
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                tgt[i] = static_cast<int64_t>(node->ne[GGML_MAX_DIMS - 1 - i]);  // reverse to OV order
            }
            return tgt;
        }
        // qwen3-next feature-window: the Slice yields [len, tok] (OV [1,1,tok,len]); the view's own
        // layout packs the window into per-head dims (e.g. [128,8,tok,1] -> OV [1,tok,8,128]). Return
        // the static OV target; translate_view places the single -1 on the token axis.
        if (detect_view_feature_window(node).first >= 0) {
            std::vector<int64_t> tgt(GGML_MAX_DIMS);
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                tgt[i] = static_cast<int64_t>(node->ne[GGML_MAX_DIMS - 1 - i]);  // reverse to OV order
            }
            return tgt;
        }
        // qwen3.5 interleaved Q/gate window (op_case 5): same contract -- return the view's own OV
        // layout; translate_view slices the per-group sub-window then reshapes to this (token -> -1).
        if (detect_view_interleaved_window(node).valid) {
            std::vector<int64_t> tgt(GGML_MAX_DIMS);
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                tgt[i] = static_cast<int64_t>(node->ne[GGML_MAX_DIMS - 1 - i]);  // reverse to OV order
            }
            return tgt;
        }
        // Detect select-and-drop: some src dim d (>1) collapses to a single index and is squeezed.
        int drop = -1;
        for (int d = GGML_MAX_DIMS - 1; d >= 0; --d) {
            if (src->ne[d] <= 1) {
                continue;
            }
            int64_t squeezed[GGML_MAX_DIMS];
            int p = 0;
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                if (i != d) {
                    squeezed[p++] = src->ne[i];
                }
            }
            squeezed[p] = 1;
            bool match = true;
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                if (node->ne[i] != squeezed[i]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                drop = d;
                break;
            }
        }
        if (drop == -1) {
            // A generic zero-offset VIEW may only repack dimensions (Falcon Kcur:
            // [128,2,T,1] -> [256,T,1,1]). If it carries the runtime token/cache axis, provide an
            // explicit dynamic target instead of letting translate_view bake the capture-time T
            // into a static Reshape. This is also required when the captured extent itself is 0.
            const int dynamic_dim = dynamic_dim_of(node);
            if (dynamic_dim == -1 || node->ne[dynamic_dim] == 1) {
                return std::vector<int64_t>{};  // pure-shrink/static view: no reshape needed
            }
            std::vector<int64_t> tgt(GGML_MAX_DIMS);
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                tgt[i] = static_cast<int64_t>(node->ne[GGML_MAX_DIMS - 1 - i]);
            }
            tgt[GGML_MAX_DIMS - 1 - dynamic_dim] = -1;
            return tgt;
        }
        // Return the static output layout (OV order). translate_view places the single -1 on the
        // token axis itself, using the sliced result's dynamic axis (element-count conservation).
        std::vector<int64_t> tgt(GGML_MAX_DIMS);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            tgt[i] = static_cast<int64_t>(node->ne[GGML_MAX_DIMS - 1 - i]);  // reverse to OV order
        }
        return tgt;
    }
    if (name == "cont_reshape") {
        // GGML_OP_CONT op_case 3 reshape-to-output target (OV dim order = ggml ne[] reversed). ggml_cont
        // may merge/split dims relative to its VIEW source (qwen3-next a/b: [tok,8,2,1] -> [8*tok,2,1,1]).
        // The token axis stays dynamic: mark the inferred dynamic output axis -1 in OV order so a 0-token
        // decode step reshapes correctly. dynamic_dim_of() is in ggml dim order; reverse for OV.
        const ggml_tensor * node = info.node;
        std::vector<int64_t> tgt(GGML_MAX_DIMS);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            tgt[i] = static_cast<int64_t>(node->ne[GGML_MAX_DIMS - 1 - i]);
        }
        const int d = dynamic_dim_of(node);
        if (d != -1 && node->ne[d] != 1) {
            tgt[GGML_MAX_DIMS - 1 - d] = -1;
        }
        return tgt;
    }
    if (name == "im2col_params") {
        // GGML_OP_IM2COL: s0,s1,p0,p1,d0,d1 + is_2D flag in op_params[0..6].
        std::vector<int32_t> p(7);
        for (int i = 0; i < 7; ++i) {
            p[i] = ggml_get_op_params_i32(info.node, i);
        }
        return p;
    }
    if (name == "pad_params") {
        // GGML_OP_PAD: 8 per-edge extents in op_params[0..7].
        std::vector<int32_t> pads(8);
        for (int i = 0; i < 8; ++i) {
            pads[i] = ggml_get_op_params_i32(info.node, i);
        }
        return pads;
    }
    if (name == "pad_circular") {
        // GGML_OP_PAD: circular-mode flag in op_params[8].
        return ggml_get_op_params_i32(info.node, 8) != 0;
    }
    if (name == "perm") {
        if (info.node->op == GGML_OP_PERMUTE) {
            std::vector<int64_t> perm(GGML_MAX_DIMS);
            for (int ggml_in_axis = 0; ggml_in_axis < GGML_MAX_DIMS; ++ggml_in_axis) {
                const int ggml_out_axis = ggml_get_op_params_i32(info.node, ggml_in_axis);
                const int ov_out_axis = GGML_MAX_DIMS - 1 - ggml_out_axis;
                perm[ov_out_axis] = GGML_MAX_DIMS - 1 - ggml_in_axis;
            }
            return perm;
        }

        // GGML_OP_TRANSPOSE: permutation order derived from input/output strides, in OV dim order.
        // Rank input/output dims by descending stride and map matched ranks to build the perm.
        std::vector<size_t> in_stride = get_stride(info.node->src[0]);
        std::vector<size_t> out_stride = get_stride(info.node);
        std::vector<std::pair<size_t, int>> in_sd, out_sd;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            in_sd.push_back({in_stride[i], i});
            out_sd.push_back({out_stride[i], i});
        }
        std::sort(in_sd.rbegin(), in_sd.rend());
        std::sort(out_sd.rbegin(), out_sd.rend());
        std::vector<int64_t> perm(GGML_MAX_DIMS);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            perm[out_sd[i].second] = in_sd[i].second;
        }
        return perm;
    }
    if (name == "reshape_target") {
        // GGML_OP_RESHAPE target shape (OV dim order = ggml ne[] reversed) with the inferred dynamic
        // (token/sequence) axis marked -1. Used by op_cases that would otherwise bake the convert-time
        // token count into a static reshape and fail when the stateful model is reused at a different
        // count (qwen3-next q/k_conv_predelta: [128,2,8,T] -> [128,16,T,1], T dynamic).
        const ggml_tensor * node = info.node;
        std::vector<int64_t> tgt(GGML_MAX_DIMS);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            tgt[i] = static_cast<int64_t>(node->ne[GGML_MAX_DIMS - 1 - i]);
        }
        const int d = dynamic_dim_of(node);
        if (d != -1 && node->ne[d] != 1) {
            tgt[GGML_MAX_DIMS - 1 - d] = -1;
        }
        return tgt;
    }
    if (name == "op_case") {
        return info.node_op_case;
    }
    if (name == "output_type") {
        return get_ov_type(info.node);
    }
    if (name == "input_ggml_shape") {
        // ggml shape of input 0, needed by the VIEW op_case 3 translator to restore the
        // original layout before slicing (the OV node may have been reshaped already).
        return get_shape(info.node_inputs_names.empty() ? info.node : info.node_inputs.at(info.node_inputs_names[0]));
    }
    if (name == "is_swa") {
        // FLASH_ATTN_EXT: the mask input (src[3]) identifies the layer flavor -- see
        // ModelParams::swa_mask for why identity, not the name, is the discriminator.
        if (info.node->op == GGML_OP_FLASH_ATTN_EXT) {
            return m_model_params.is_swa_mask(info.node->src[3]);
        }
        return false;
    }
    if (name == "view_seq_offset") {
        // PERMUTE KV-cache case: byte offset of the VIEW input divided by nb[3] (bytes per
        // sequence) gives the first active sequence index.
        if (info.node->op == GGML_OP_PERMUTE && info.node->src[0] != nullptr && info.node->src[0]->op == GGML_OP_VIEW) {
            const ggml_tensor * view = info.node->src[0];
            size_t byte_offset = 0;
            memcpy(&byte_offset, view->op_params, sizeof(size_t));
            return static_cast<int64_t>(byte_offset / view->src[0]->nb[3]);
        }
        return static_cast<int64_t>(0);
    }
    return {};
}

void GgmlOvDecoder::visit_subgraph(std::function<void(std::shared_ptr<GgufDecoder>)> node_visitor) const {
    for (int node_idx = 0; node_idx < (int) m_node_info_list.size(); node_idx++) {
        // Hand the visitor a decoder bound to this node (per-node accessors use m_node_idx).
        auto node_decoder = std::make_shared<GgmlOvDecoder>(*this);
        node_decoder->m_node_idx = node_idx;
        node_visitor(node_decoder);
    }
}

std::string GgmlOvDecoder::compute_op_type(const ggml_tensor * node) {
    // Derive the op-type string straight from ggml's own name tables so new ops need no per-op
    // entry here: "GGML_OP_" + ggml_op_name(op) yields exactly the keys the frontend op_table uses
    // (e.g. GGML_OP_MUL_MAT, GGML_OP_CONCAT), and likewise for the unary/glu families. Weight
    // leaves keep their genuine "GGML_OP_NONE" type (set_input_output surfaces them separately).
    switch (node->op) {
    case GGML_OP_UNARY:
        return std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(node));
    case GGML_OP_GLU:
        return std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(node));
    default:
        return std::string("GGML_OP_") + ggml_op_name(node->op);
    }
}

const std::string & GgmlOvDecoder::get_op_type() const {
    static const std::string unknown_op = "UNKNOWN_GGML_OP";
    return m_node_idx == -1 ? unknown_op : m_node_info_list[m_node_idx].node_op_type;
}
