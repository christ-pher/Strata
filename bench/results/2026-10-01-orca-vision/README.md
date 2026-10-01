# Orca IQ4_XS GPU vision smoke test

2026-10-01: upstream v0.1.32 merged into the V100 fork, CUDA 12.8, sm70,
262144 context, 32768 resident KV cells, INT8 KV, GPU vision (1024 token limit).
The input was a 400x300 white PNG with a red rectangle at (100,60)-(300,240).
Prompt: “What color is the large shape in this image? Answer with one color word.”
Temperature 0, thinking disabled, max_tokens 32. Response: “Red”.
147 prompt tokens, 2 generated tokens. The engine reported 2156 ms prompt
processing and 163 ms generation. These exclude encoder time and model startup.
This verifies the image path, not overall vision quality or throughput.
The server ran on an isolated port and was stopped afterward; production remains
text-only, with vision assets prepared for optional activation.
