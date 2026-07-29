# Performance validation: ov-frontend-swap-llamacpp-sync @ 6a6442a0c

llama-bench -fa 1 -p 128 -n 64 -r 2, device CPU, GGML_OPENVINO_STATEFUL_EXECUTION=1.
Reference column is the ggml CPU backend (`--device none`) on the same host and build.

| Model                | pp128 OV        | pp128 CPU ref   | pp ratio | tg64 OV      | tg64 CPU ref  | tg ratio |
|----------------------|-----------------|-----------------|----------|--------------|---------------|----------|
| Llama-3.2-1B Q4_K_M  | 1036.56 ±19.36  |  521.55 ± 2.34  |  1.99x   | 50.45 ±1.69  |  74.44 ±0.03  |  0.68x   |
| gemma3-1B Q4_K_M     | 1416.90 ±51.67  |  421.16 ± 3.60  |  3.36x   | 47.82 ±1.12  |  76.00 ±0.08  |  0.63x   |
| gemma4-E4B Q4_K_M    |  249.94 ± 4.89  |  113.94 ± 0.47  |  2.19x   | 13.89 ±0.27  |  18.43 ±0.06  |  0.75x   |
| Qwen3-0.9B-A0.6B MoE |   15.56 ± 0.02  |  528.80 ± 5.26  |  0.03x   |  8.45 ±0.04  | 100.33 ±0.13  |  0.08x   |
| Qwen3.5-4B Q4_K_M    |  243.56 ± 2.42  |  104.99 ± 0.92  |  2.32x   | 12.56 ±0.08  |  19.76 ±0.05  |  0.64x   |

Four of the five follow the established shape: prefill 2-3.4x faster than the ggml CPU
backend, decode 0.63-0.75x of it.

## Known issue: MoE expert weights are dequantized before the expert Gather

Qwen3-MoE is ~34x slower than the CPU reference on prefill and ~12x on decode. It is not
graph fragmentation (the graph compiles as one subgraph) but the shape of the expert-weight
subgraph, visible in a dumped IR:

    Const FP16 -> Subtract -> Multiply -> Reshape -> Convert FP32 [6144,1024]
      -> Reshape [1, n_expert, 3072, 1024] -> Gather(ids) -> MatMul

The whole expert tensor is decompressed to fp32 and only then indexed, so every matmul
materializes all experts to select n_used=2 of them. Counted over the IR: 84 such Gathers
(28 layers x 3 matmuls), 2.11 GB of fp32 expert weights per forward pass against a 525 MiB
model -- which accounts for the measured ~118 ms/token. The eager `Convert FP32` also
defeats OpenVINO's compressed-weight fusion, inflating the serialized IR to 616 MB.

The fix is to gather on the *quantized* constant and decompress only the selected experts
(or keep the experts as a compressed constant so the plugin fuses the decompression), which
turns per-token traffic from all-experts-fp32 into n_used-experts-quantized. This is a
frontend weight-path change (`src/quant/weights.cpp` + `op/mul_mat_id.cpp`), not a backend
one, and is independent of the correctness fixes validated here.
