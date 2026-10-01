#include "strata/prefill/gemm.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <vector>
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
int main() {
    strata::prefill::Gemm gemm;
    std::string error;
    if (!gemm.init(nullptr, 0, error)) throw std::runtime_error(error);
    constexpr int t = 8192, k = 2560, iterations = 10;
    for (int n : {48, 128, 512, 2560}) {
        uint16_t *x = nullptr, *w = nullptr;
        float* y = nullptr;
        std::vector<uint16_t> hx(size_t(t) * k, 0x3e80), hw(size_t(n) * k, 0x3d80);
        check(cudaMalloc((void**)&x, hx.size() * 2));
        check(cudaMalloc((void**)&w, hw.size() * 2));
        check(cudaMalloc((void**)&y, size_t(t) * n * 4));
        check(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        check(cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));
        for (int i = 0; i < 3; ++i) gemm.bf16(x, w, y, t, n, k);
        check(cudaDeviceSynchronize());
        cudaEvent_t a, b;
        check(cudaEventCreate(&a)); check(cudaEventCreate(&b));
        check(cudaEventRecord(a));
        for (int i = 0; i < iterations; ++i) gemm.bf16(x, w, y, t, n, k);
        check(cudaEventRecord(b)); check(cudaEventSynchronize(b));
        float ms = 0; check(cudaEventElapsedTime(&ms, a, b));
        std::printf("T=%d N=%d K=%d mean_ms=%.6f\n", t, n, k, ms / iterations);
        cudaEventDestroy(a); cudaEventDestroy(b);
        cudaFree(x); cudaFree(w); cudaFree(y);
    }
}
