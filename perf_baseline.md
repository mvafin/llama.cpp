# Baseline: llama.cpp OpenVINO backend, IN-TREE ggml frontend (build 0b7154066)
# llama-bench -fa 1 -p 128 -n 64 -r 2

| Model                        | Device | pp128 t/s      | tg64 t/s     |
|------------------------------|--------|----------------|--------------|
| Llama-3.2-3B-Instruct Q4_0   | CPU    | 340.25 ±1.88   | 17.94 ±0.06  |
| Llama-3.2-3B-Instruct Q4_0   | GPU    |  92.71 ±0.05   | 17.54 ±0.22  |
| TinyLlama-1.1B-Chat Q4_K_M   | CPU    | 922.09 ±27.47  | 45.14 ±0.16  |
| TinyLlama-1.1B-Chat Q4_K_M   | GPU    | 273.84 ±0.41   | 44.33 ±8.31  |
| Qwen2.5-0.5B-Instruct Q8_0   | CPU    | 2519.57 ±9.56  | 67.18 ±0.51  |
| Qwen2.5-0.5B-Instruct Q8_0   | GPU    | 767.05 ±6.62   | 37.77 ±2.64  |
| Maincoder-1B Q4_K_M          | CPU    | 941.94 ±18.22  | 37.40 ±0.15  |
| Maincoder-1B Q4_K_M          | GPU    | 318.89 ±0.49   | 40.20 ±6.17  |

# Correctness oracle (llama-3.2, "The capital of France is", greedy): "Paris."
