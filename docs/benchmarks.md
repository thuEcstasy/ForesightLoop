# Benchmark definitions

`results/decode-latency-gb200-b32.csv` contains recorded medians from the final paper measurement. A batch contains 32 independent sequences, each with a 2,048-token prompt and 2,048 output tokens. Timing includes the 2,047 cached decode steps after prefill produces the first output token, excluding prefill, model loading, warmup, and CUDA Graph capture. Three repetitions were synchronized and the median reported.

One GB200 GPU is used per model (tensor parallelism one). The optimized path uses FP16 linear and attention operands, FP32 residual states/reductions/logits, CUDA Graphs, the vLLM FA4 global attention backend, and custom Triton G-SWA kernels. All recurrent configurations use a 64-token local window.

The archived LT-8 source record is `lt8_attnres`: it includes in-trajectory Attention Residuals and remains serial across loops. The paper displays this record as LT-8. LT-4 uses ordinary recurrence. This provenance is retained so the compared implementations remain unambiguous.

The reference package is not the optimized benchmark path and is not a serving scheduler. Its runtime should not be compared directly to the archived CSV. Logical cache tables exclude weights, temporary workspaces, and allocator padding; they are not peak device-memory measurements.
