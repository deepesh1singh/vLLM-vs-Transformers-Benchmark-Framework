# LLM Inference Performance Lab

A reproducible framework for benchmarking, profiling, and optimizing Large Language Model (LLM) inference on NVIDIA GPUs.

## Overview

This project studies the performance characteristics of decoder-only LLM inference using Hugging Face Transformers, PyTorch, and vLLM.

The objective is not to build another chatbot, but to understand how different inference strategies affect latency, throughput, GPU memory consumption, power usage, and energy efficiency through controlled and reproducible experiments.

The main research question is:

> How do batching strategy, request concurrency, prompt-length heterogeneity, and runtime configuration affect LLM inference performance on NVIDIA GPUs?

The experiments cover both offline static batching and online continuous batching, including token-level streaming, stability analysis, hardware telemetry, and controlled backend comparisons.

---

## Key findings

- Token-level streaming instrumentation added approximately 1.3% latency overhead and 0.14% GPU memory overhead.
- Static batching scaled efficiently until the GPU reached a practical saturation point.
- Mixed-length static batches introduced substantial input padding and degraded throughput.
- Length bucketing reduced padding from 57.8% to 5.3% and improved throughput by approximately 12%.
- Transformers static batching was faster for an ideal homogeneous offline workload.
- vLLM delivered 2.73× the output throughput of Transformers for a heterogeneous workload.
- Eager vLLM execution produced the highest median throughput, while CUDA graphs offered better TTFT, power consumption, and stability.

---

## Features

- Hugging Face Transformers inference baseline
- Non-streaming and token-level streaming inference
- TTFT, inter-token latency, and decode-throughput measurement
- Static batching and batch-size scaling
- Mixed-length prompt analysis
- Length-bucketed batching
- vLLM OpenAI-compatible server
- Asynchronous streaming benchmark client
- Concurrent workload generation
- Repeated benchmark execution and stability classification
- Latency and throughput percentile aggregation
- NVIDIA GPU utilization, memory, power, clock, and temperature telemetry
- GPU energy estimation
- Homogeneous and heterogeneous backend comparisons
- JSON result generation
- Reproducible benchmark scripts

---

## Repository structure

```text
.
├── data/
│   └── heterogeneous_prompts.json
├── results/
├── scripts/
│   ├── run_async_benchmark.py
│   ├── run_async_smoke_test.py
│   ├── run_static_batch.py
│   └── start_vllm_server.sh
├── src/
│   └── llm_perf_lab/
│       ├── baseline/
│       ├── client/
│       └── profiling/
├── tests/
├── pyproject.toml
└── README.md
```

---

## Tested configuration

The reported results were collected using:

```text
GPU:             NVIDIA GeForce RTX 4080
GPU memory:      16 GB
Operating mode:  WSL
Model:           Qwen/Qwen2.5-3B-Instruct
Precision:       float16
Output length:   128 tokens
PyTorch:         2.13.0+cu130
vLLM:            0.26.0
```

Transformers and vLLM are installed in separate virtual environments because their dependency requirements may differ:

```text
.venv        Transformers benchmarks, clients, tests, and analysis
.venv-vllm   vLLM server
```

---

## Installation

### Transformers environment

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

### vLLM environment

```bash
python -m venv .venv-vllm
source .venv-vllm/bin/activate

pip install vllm
```

The exact installed package versions should be recorded alongside benchmark results when reproducing the experiments.

---

## Running the benchmarks

### Start the vLLM server

```bash
source .venv-vllm/bin/activate
./scripts/start_vllm_server.sh
```

### Run the asynchronous vLLM benchmark

```bash
source .venv/bin/activate

python scripts/run_async_benchmark.py \
  --prompt "Explain continuous batching and its performance benefits." \
  --concurrency 128 \
  --requests-per-level 384 \
  --repetitions 5 \
  --warmup-requests 384 \
  --warmup-concurrency 128 \
  --max-tokens 128 \
  --profile \
  --telemetry-interval-seconds 0.2 \
  --output results/vllm_concurrency_128.json
```

### Run the sustained Transformers benchmark

Stop the vLLM server before running Transformers so that both backends have exclusive access to the GPU.

```bash
source .venv/bin/activate

python scripts/run_static_batch.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --prompt "Explain continuous batching and its performance benefits." \
  --batch-sizes 32,64,128 \
  --requests-per-repetition 384 \
  --warmup-repetitions 1 \
  --repetitions 5 \
  --max-new-tokens 128 \
  --dtype float16 \
  --profile \
  --telemetry-interval-seconds 0.2 \
  --output results/transformers_static_sustained.json
```

### Run the heterogeneous workload

```bash
python scripts/run_static_batch.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --prompts-file data/heterogeneous_prompts.json \
  --batch-sizes 128 \
  --requests-per-repetition 384 \
  --warmup-repetitions 1 \
  --repetitions 5 \
  --max-new-tokens 128 \
  --dtype float16 \
  --profile \
  --output results/transformers_heterogeneous.json
```

The same prompt file can be supplied to the asynchronous vLLM benchmark:

```bash
python scripts/run_async_benchmark.py \
  --prompts-file data/heterogeneous_prompts.json \
  --concurrency 128 \
  --requests-per-level 384 \
  --repetitions 5 \
  --warmup-requests 384 \
  --warmup-concurrency 128 \
  --max-tokens 128 \
  --profile \
  --output results/vllm_heterogeneous.json
```

---

## Methodology

Unless otherwise stated, the controlled backend comparisons used:

```text
Model:                    Qwen/Qwen2.5-3B-Instruct
Precision:                float16
Requests per repetition:  384
Measured repetitions:     5
Maximum output tokens:    128
Sampling:                 disabled
Temperature:              0
GPU telemetry interval:   0.2 seconds
```

Warm-up requests were executed before measurement to reduce model-loading, kernel-compilation, cache-initialization, and CUDA-graph capture effects.

A throughput coefficient of variation was used to classify repeated measurements:

```text
CV <= 5%       stable
5% < CV <= 10% noisy
CV > 10%       unstable
```

Only executions with a 100% request success rate were accepted for final comparisons.

Static batch size and online concurrency are not identical concepts. They are compared as operating configurations rather than treated as directly equivalent scheduling mechanisms.

---

## Results

### Token-level streaming instrumentation

Token-level instrumentation added approximately 1.3% latency overhead and 0.14% GPU memory overhead, while enabling measurement of:

- Time to first token
- Inter-token latency
- Decode throughput
- Post-generation overhead

This demonstrated that detailed streaming metrics can be collected with minimal impact on end-to-end performance.

### Static batch-size scaling

Static batching scaled efficiently up to approximately batch size 256 on the tested configuration.

Beyond that point, throughput gains diminished sharply while latency and GPU memory usage continued to increase. This indicated that the GPU reached a practical saturation point before exhausting all available memory.

### Mixed-length static batching

Mixed-length static batching produced approximately 58–59% input padding.

Compared with the homogeneous-prompt workload:

- Throughput decreased by 13.9% at batch size 32.
- Throughput decreased by 61.0% at batch size 256.
- Latency increased by 16.1% at batch size 32.
- Latency increased by 156.3% at batch size 256.

Static batching therefore became increasingly inefficient as requests with different input lengths were grouped into larger batches.

### Length bucketing

Length bucketing reduced the input padding ratio from 57.8% to 5.3% when processing the same 256 requests in static micro-batches of 16.

This optimization:

- Reduced total latency by 10.7%.
- Increased request throughput by approximately 12.0%.
- Increased output-token throughput by approximately 12.0%.

The result demonstrates that grouping requests with similar prompt lengths can substantially improve static batching efficiency.

### vLLM concurrency scaling and saturation

Hardware profiling showed that sustained output throughput increased from approximately 1,975 tokens/s at concurrency 32 to 5,577 tokens/s at concurrency 128.

Increasing concurrency from 128 to 192:

- Improved throughput by only 1.9%.
- Increased median latency by 42.1%.
- Increased median TTFT by 59.6%.

GPU power plateaued near 220 W, confirming a practical saturation point around concurrency 128.

CPU utilization remained below 7%, graphics clocks remained stable, and the maximum GPU temperature was 62 °C. The workload was therefore neither CPU-bound nor thermally constrained.

Concurrency 64 provided a more balanced operating point, while concurrency 128 maximized aggregate throughput.

---

## Transformers vs. vLLM

### Homogeneous workload

Under a homogeneous sustained workload, Transformers static batching scaled efficiently from batch size 32 to 128.

At batch size 128, Transformers achieved:

```text
Output throughput:  5,575 tokens/s
Request throughput: 43.56 requests/s
Throughput CV:       0.53%
GPU power:           197.9 W
Energy:              35.06 J/1,000 output tokens
```

The stabilized vLLM result at concurrency 128 achieved:

```text
Output throughput:  4,820 tokens/s
Request throughput: 37.65 requests/s
Median latency:      3.10 seconds
Median TTFT:         549 ms
GPU power:           196.8 W
Energy:              40.3 J/1,000 output tokens
```

For this ideal static workload, Transformers delivered approximately 15.7% higher output throughput and consumed approximately 13% less GPU energy per 1,000 output tokens.

This scenario strongly favored static batching:

- Every request used the same prompt.
- Every request generated the same number of tokens.
- Complete batches were available in advance.
- No dynamic request scheduling was required.
- No batch-wide padding was introduced.

vLLM provides scheduling, request isolation, streaming, and continuous-batching capabilities whose advantages are more relevant to dynamic online workloads.

### Heterogeneous workload

The heterogeneous workload contained eight prompts with substantially different input lengths.

Each backend processed 384 requests per repetition, generated 128 output tokens per request, and was evaluated over five measured repetitions.

#### Transformers static batching

```text
Batch size:              128
Output throughput:       1,616 tokens/s
Request throughput:      12.63 requests/s
Workload completion:     30.41 seconds
Median batch latency:    10.17 seconds
Input padding ratio:     49.1%
GPU power:               252.6 W
Energy:                  156.5 J/1,000 output tokens
```

#### vLLM continuous batching

```text
Concurrency:             128
Output throughput:       4,405 tokens/s
Request throughput:      34.41 requests/s
Workload completion:     11.16 seconds
Median request latency:  3.37 seconds
Median TTFT:             662 ms
GPU power:               202.5 W
Energy:                  approximately 45.4 J/1,000 output tokens
```

Compared with Transformers, vLLM:

- Delivered 2.73× the output throughput.
- Reduced workload completion time by 63.3%.
- Used approximately 19.8% less GPU power.
- Reduced energy consumption per 1,000 output tokens by approximately 71%.

The result demonstrates that static batching can be exceptionally efficient for homogeneous workloads, but its performance degrades sharply when input lengths vary because every sequence is padded to the longest prompt in the batch.

Continuous batching avoids this batch-wide padding penalty and schedules requests independently, providing substantially better throughput and energy efficiency for heterogeneous online workloads.

### Comparison limitation

The vLLM chat endpoint applied the model's chat template, resulting in 27.8% more measured input tokens than the raw-text Transformers workload.

Despite processing more input tokens, vLLM retained a substantial performance advantage.

The input-token difference means that the comparison is not perfectly identical at the tokenized-input level and must be considered when interpreting the absolute results.

---

## Runtime configuration optimization

A controlled ablation compared vLLM CUDA graph execution with eager execution using the same heterogeneous workload, concurrency level, warm-up policy, and generation settings.

| Metric | CUDA graphs | Eager |
|---|---:|---:|
| Output throughput | 4,405 tok/s | 4,788 tok/s |
| Request throughput | 34.41 req/s | 37.40 req/s |
| Median latency | 3.37 s | 3.15 s |
| Median TTFT | 662 ms | 739 ms |
| Median GPU power | 202.5 W | 220.6 W |
| Energy per 1,000 output tokens | 45.4 J | 45.8 J |
| Throughput CV | 4.50% | 5.05% |

Eager execution:

- Increased median output throughput by approximately 8.7%.
- Reduced median request latency by approximately 6.4%.
- Increased median TTFT by approximately 11.7%.
- Increased median GPU power by approximately 8.9%.
- Produced slightly greater throughput variability.

Energy consumption per 1,000 output tokens was nearly identical between the two configurations.

For this hardware and workload, eager execution provided the highest median throughput, while CUDA graphs offered a better balance of TTFT, power consumption, and performance stability.

CUDA graph execution was therefore retained as the recommended configuration for latency-sensitive online serving.

---

## Operational recommendations

### Homogeneous offline workloads

Use Transformers static batching when:

- Requests are known in advance.
- Prompt and generation lengths are similar.
- Complete batches can be formed without delay.
- Maximum offline throughput is the primary objective.

### Heterogeneous online workloads

Use vLLM continuous batching when:

- Requests arrive dynamically.
- Prompt lengths differ substantially.
- Responses are streamed independently.
- Request-level latency and scheduling flexibility matter.
- Avoiding batch-wide padding is important.

### vLLM operating points

```text
Concurrency 64:
Balanced latency and throughput

Concurrency 128:
Maximum practical throughput

Concurrency 192:
Oversaturated; limited throughput improvement with substantial latency cost
```

### Runtime mode

```text
CUDA graphs:
Recommended for better TTFT, power consumption, and stability

Eager mode:
Useful when maximum median throughput is the primary objective
```

---

## Roadmap

- [x] Baseline inference
- [x] Token-level streaming
- [x] Static batching
- [x] Mixed-length prompt analysis
- [x] Length-bucketed batching
- [x] vLLM server
- [x] Asynchronous streaming client
- [x] Concurrent benchmark execution
- [x] Metrics aggregation
- [x] Stability analysis
- [x] Hardware profiling
- [x] Concurrency scaling
- [x] Practical saturation analysis
- [x] Transformers vs. vLLM comparison
- [x] Heterogeneous workload comparison
- [x] Optimization study
- [x] Runtime configuration ablation
- [x] Operational recommendations

---

## Reproducibility notes

Performance results depend on:

- GPU model and power limits
- NVIDIA driver
- CUDA version
- PyTorch version
- Transformers version
- vLLM version
- Model revision
- Precision
- Server startup options
- CUDA graph or eager execution
- Maximum model length
- GPU memory utilization
- Prompt tokenization and chat templates
- Warm-up policy
- Workload composition

Absolute results from different server configurations should not be combined without recording these parameters.

---
