# Native ABCA draft — NOT APPROVED FOR GPU

Uses temporary Python SDPA API wrapper and module hooks; passes original args and kwargs unchanged. This changes Python dispatch and synchronizes CPU copies. It does not establish the actual CUDA kernel. Numerical transparency requires exact original-score bridging; no such real test has happened. Input/coverage binding, production scoring, persistence and comparisons remain unimplemented. The toy test does not establish real BF16 GPU behavior. Keep separate from the active PQR process and do not run on GPU.
