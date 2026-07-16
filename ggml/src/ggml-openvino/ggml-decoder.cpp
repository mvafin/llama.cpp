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
#include <execution>
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

void GgmlOvDecoder::set_input_output() {
    // First, surface each unique quantized/weight tensor as a node. A weight is a ggml leaf
    // (GGML_OP_NONE) and keeps that genuine op type; the frontend recognizes it as a weight by
    // the presence of the "data" attribute (get_attribute), dequantizes/requantizes from the raw
    // bytes, and the decoder only provides them. These come first so a weight is visited before
    // its consumer.
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
                    buffer && (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type));
                if (!is_weight || seen_weights.count(src_name)) {
                    continue;
                }
                seen_weights.insert(src_name);
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

    for (int node_n = 0; node_n < m_cgraph->n_nodes; node_n++) {
        auto node = m_cgraph->nodes[node_n];

        NodeInfo current_node_info;
        auto node_name = std::string(node->name);
        auto node_output_name = node_name;
        auto * node_output = node;
        if (node->op == GGML_OP_SET_ROWS) {
            // SET_ROWS updates the tensor in place. For later ov op that uses the
            // the view_src of SET_ROWS, we need to make sure they get the updated tensor
            // by putting the view_src name in the tensor_map in
            // <openvino>/src/frontends/ggml/src/translate_session.cpp
            node_output_name = std::string(node->view_src->name);
            node_output = node->view_src;
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
            auto src_name = std::string(src->name);
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                src_name = get_graph_input_ov_name(src, node);
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
        if (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == node->ne[0] && src->src[0]->ne[1] == node->ne[1]) {
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
        if (node->src[1]->op == GGML_OP_VIEW) {
            op_case = 2;
        }
        break;
    }
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
        if (node->src[0]->op == GGML_OP_VIEW) {
            auto * src = node->src[0];
            if (ggml_nelements(node) != ggml_nelements(src)) {
                throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 2;
        }
        {
            auto * src = node->src[0];
            if ((ggml_nelements(node) != ggml_nelements(src)) && m_naive) {
                // Compare each dimension of node and src, if only one dimension differs then op_case=3
                int diff_count = 0;
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    if (node->ne[i] != src->ne[i]) {
                        diff_count++;
                    }
                }
                if (diff_count == 1) {
                    op_case = 3;
                }
            }
        }
        break;
    }
    default:
        break;
    }
    return op_case;
}

int extract_layer_from_name(const std::string & name) {
    size_t pos1 = name.find("_l");
    assert(pos1 != std::string::npos);
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    if (pos2 == std::string::npos) {
        pos2 = name.length();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    int layer = std::stoi(layer_str);
    return layer;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, bool is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        auto * node = cgraph->nodes[i];
        std::string name = std::string(node->name);
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            model_params.n_heads = node->src[0]->ne[2];
            model_params.n_heads_kv = node->src[1]->ne[2];
            model_params.head_size = node->src[0]->ne[0];
            compute_params.input_len = node->src[0]->ne[1];

            auto * cache_k_perm = node->src[1];
            if (cache_k_perm->op == GGML_OP_CPY) {
                cache_k_perm = cache_k_perm->src[0];
            }
            assert(cache_k_perm->op == GGML_OP_PERMUTE);
            auto * cache_k_view = cache_k_perm->src[0];
            assert(cache_k_view->op == GGML_OP_VIEW);

            auto * cache_k = cache_k_view->src[0];
            int layer = extract_layer_from_name(cache_k->name);
            auto * mask = node->src[3];
            std::string mask_name(mask->name);

            model_params.kv_buffer_ctx_id = ggml_backend_openvino_buffer_get_ctx_id(cache_k->buffer);
            if (mask_name.find("swa") != std::string::npos) {
                model_params.swa_layers.push_back(layer);
                model_params.ctx_per_seq_swa = cache_k->ne[1];
            } else {
                model_params.ctx_per_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_active = mask->ne[3];
            auto seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            size_t offset;
            memcpy(&offset, cache_k_view->op_params, sizeof(size_t));
            compute_params.seq_active_start = offset / seq_size;
            compute_params.token_len_per_seq = node->ne[2];

            if (mask_name.find("swa") != std::string::npos) {
                compute_params.attention_size_swa = mask->ne[0];
            } else {
                compute_params.attention_size = mask->ne[0];
            }
            if (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
                compute_params.attention_size_swa = model_params.ctx_per_seq_swa;
                compute_params.token_len_per_seq = 1;
            }
            break;
        }
        if (node->op == GGML_OP_ROPE) {
            memcpy(model_params.rope_params, node->op_params, sizeof(int32_t) * 15);
        }
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
        return input!= nullptr ? ov::PartialShape{get_shape(input)} : ov::PartialShape{get_shape(op)};
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
        if (!m_is_static) {
            // do not fix ctx size to make llama-bench work across test params
            input_shape[2] = -1;
        }

    } else if (is_kv_idx(input, op)) {
        // kv update index
        int len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        input_shape = ov::PartialShape{1, 1, 1, len};

    } else {
        input_shape = ov::PartialShape{get_shape(input)};
    }
    return input_shape;
}

void GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    auto create_1d_input = [this](const std::string & name, int64_t value) {
        if (m_is_static) {
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

    create_1d_input("attention_size", m_compute_params.attention_size);
    if (m_compute_params.attention_size_swa != -1) {
        create_1d_input("attention_size_swa", m_compute_params.attention_size_swa);
    }
    create_1d_input("n_seq_active", m_compute_params.n_seq_active);
    create_1d_input("seq_active_start", m_compute_params.seq_active_start);
    create_1d_input("seq_active_end", m_compute_params.seq_active_start + m_compute_params.n_seq_active);
    create_1d_input("token_len_per_seq", m_compute_params.token_len_per_seq);
    // create_1d_input("token_len", m_token_len_per_seq * m_n_seq_active);

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
            std::string src_name = std::string(src->name);
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                src_name = get_graph_input_ov_name(src, node);
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

            m_inputs[src_name] = src;

            ggml_backend_buffer * buffer = src->buffer;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                if (auto it = std::find(m_model_params.kv_names.begin(), m_model_params.kv_names.end(), src_name);
                    it == m_model_params.kv_names.end()) {
                    m_model_params.kv_names.push_back(src_name);
                }
            }
            ov::PartialShape param_shape = get_graph_input_shape(node, src);
            auto param_node = std::make_shared<ov::op::v0::Parameter>(get_ov_type(src), param_shape);
            param_node->set_friendly_name(src_name);
            param_node->output(0).get_tensor().set_names({src_name});
            m_model_inputs[src_name] = param_node;
        }
    }
}

void GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_names.clear();
    for (int node_n = 0; node_n < m_cgraph->n_nodes; node_n++) {
        auto * cur_node = m_cgraph->nodes[node_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        if (cur_node->op == GGML_OP_NONE) {
            continue;
        }
        auto cur_node_use_count = m_cgraph->use_counts[ggml_hash_find(&m_cgraph->visited_hash_set, cur_node)];
        if (cur_node_use_count == 0) {
            // The output of SET_ROWS is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            if (cur_node != nullptr && cur_node->op == GGML_OP_SET_ROWS) {
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
            std::string node_output_name(cur_node->name);
            m_model_outputs[node_output_name] = cur_node;
            m_model_output_names.push_back(node_output_name);
        }
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
    for (const auto & name : m_model_params.kv_names) {
        kv_param_res_names[name] = name;
    }
    return kv_param_res_names;
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
    // A weight node's output shape is the logical 2D [rows, cols] the frontend expects.
    if (m_weight_names.count(m_node_info_list[m_node_idx].node_name)) {
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
    const int32_t * rp = m_model_params.rope_params;
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
        return ggml_get_op_params_f32(info.node, 0);
    }
    if (name == "scale") {
        return ggml_get_op_params_f32(info.node, 0);
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
    if (name == "op_case") {
        return info.node_op_case;
    }
    if (name == "output_type") {
        return get_ov_type(info.node);
    }
    if (name == "input_ggml_shape") {
        // ggml shape of input 0, needed by the VIEW op_case 3 translator to restore the
        // original layout before slicing (the OV node may have been reshaped already).
        return get_shape(info.node_inputs_names.empty() ? info.node
                                                        : info.node_inputs.at(info.node_inputs_names[0]));
    }
    if (name == "is_swa") {
        // FLASH_ATTN_EXT: mask input (src[3]) name contains "swa" for SWA layers.
        if (info.node->op == GGML_OP_FLASH_ATTN_EXT && info.node->src[3] != nullptr) {
            return std::string(info.node->src[3]->name).find("swa") != std::string::npos;
        }
        return false;
    }
    if (name == "view_seq_offset") {
        // PERMUTE KV-cache case: byte offset of the VIEW input divided by nb[3] (bytes per
        // sequence) gives the first active sequence index.
        if (info.node->op == GGML_OP_PERMUTE && info.node->src[0] != nullptr &&
            info.node->src[0]->op == GGML_OP_VIEW) {
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
    static const std::map<ggml_op, std::string> ops = {
        {GGML_OP_NONE,           "GGML_OP_NONE"          },
        {GGML_OP_ACC,            "GGML_OP_ACC"           },
        {GGML_OP_ADD,            "GGML_OP_ADD"           },
        {GGML_OP_ADD1,           "GGML_OP_ADD1"          },
        {GGML_OP_CONT,           "GGML_OP_CONT"          },
        {GGML_OP_DIV,            "GGML_OP_DIV"           },
        {GGML_OP_DUP,            "GGML_OP_DUP"           },
        {GGML_OP_GET_ROWS,       "GGML_OP_GET_ROWS"      },
        {GGML_OP_MUL,            "GGML_OP_MUL"           },
        {GGML_OP_MUL_MAT,        "GGML_OP_MUL_MAT"       },
        {GGML_OP_PERMUTE,        "GGML_OP_PERMUTE"       },
        {GGML_OP_RESHAPE,        "GGML_OP_RESHAPE"       },
        {GGML_OP_RMS_NORM,       "GGML_OP_RMS_NORM"      },
        {GGML_OP_ROPE,           "GGML_OP_ROPE"          },
        {GGML_OP_SCALE,          "GGML_OP_SCALE"         },
        {GGML_OP_SOFT_MAX,       "GGML_OP_SOFT_MAX"      },
        {GGML_OP_SUB,            "GGML_OP_SUB"           },
        {GGML_OP_TRANSPOSE,      "GGML_OP_TRANSPOSE"     },
        {GGML_OP_VIEW,           "GGML_OP_VIEW"          },
        {GGML_OP_SET_ROWS,       "GGML_OP_SET_ROWS"      },
        {GGML_OP_CPY,            "GGML_OP_CPY"           },
        {GGML_OP_FLASH_ATTN_EXT, "GGML_OP_FLASH_ATTN_EXT"},
    };
    static const std::map<ggml_unary_op, std::string> unary_ops = {
        {GGML_UNARY_OP_ABS,         "GGML_UNARY_OP_ABS"        },
        {GGML_UNARY_OP_SGN,         "GGML_UNARY_OP_SGN"        },
        {GGML_UNARY_OP_NEG,         "GGML_UNARY_OP_NEG"        },
        {GGML_UNARY_OP_STEP,        "GGML_UNARY_OP_STEP"       },
        {GGML_UNARY_OP_TANH,        "GGML_UNARY_OP_TANH"       },
        {GGML_UNARY_OP_ELU,         "GGML_UNARY_OP_ELU"        },
        {GGML_UNARY_OP_RELU,        "GGML_UNARY_OP_RELU"       },
        {GGML_UNARY_OP_SIGMOID,     "GGML_UNARY_OP_SIGMOID"    },
        {GGML_UNARY_OP_GELU,        "GGML_UNARY_OP_GELU"       },
        {GGML_UNARY_OP_GELU_QUICK,  "GGML_UNARY_OP_GELU_QUICK" },
        {GGML_UNARY_OP_SILU,        "GGML_UNARY_OP_SILU"       },
        {GGML_UNARY_OP_HARDSWISH,   "GGML_UNARY_OP_HARDSWISH"  },
        {GGML_UNARY_OP_HARDSIGMOID, "GGML_UNARY_OP_HARDSIGMOID"},
        {GGML_UNARY_OP_EXP,         "GGML_UNARY_OP_EXP"        },
        {GGML_UNARY_OP_COUNT,       "GGML_UNARY_OP_COUNT"      }
    };
    static const std::map<ggml_glu_op, std::string> glu_ops = {
        {GGML_GLU_OP_SWIGLU, "GGML_GLU_OP_SWIGLU"},
        {GGML_GLU_OP_GEGLU,  "GGML_GLU_OP_GEGLU" },
        {GGML_GLU_OP_REGLU,  "GGML_GLU_OP_REGLU" }
    };

    switch (node->op) {
    case GGML_OP_UNARY:
        return unary_ops.at(ggml_get_unary_op(node));
    case GGML_OP_GLU:
        return glu_ops.at(ggml_get_glu_op(node));
    default:
        return ops.at(node->op);
    }
    static const std::string unknown_op = "UNKNOWN_GGML_OP";
    return unknown_op;
}

const std::string & GgmlOvDecoder::get_op_type() const {
    static const std::string unknown_op = "UNKNOWN_GGML_OP";
    return m_node_idx == -1 ? unknown_op : m_node_info_list[m_node_idx].node_op_type;
}
