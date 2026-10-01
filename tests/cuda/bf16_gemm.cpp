#include "strata/prefill/gemm.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
float decode(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
struct Buffer {
    void* p{};
    explicit Buffer(size_t n) { check(cudaMalloc(&p, n)); }
    ~Buffer() { cudaFree(p); }
};
int main() {
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(nullptr, 0, err)) throw std::runtime_error(err);
    // Narrow SIMT, irregular dimensions, prompt/output tile boundaries, and nontrivial output stride.
    const int shapes[][3] = {{17,4,47}, {33,35,47}, {1031,35,47}, {3,4103,4097}};
    for (const auto& shape : shapes) for (float beta : {0.f, .5f, 1.f}) {
        const int t=shape[0], n=shape[1], k=shape[2], stride=n+4;
        std::vector<uint16_t> x(t*k), w(n*k);
        for (size_t i=0; i<x.size(); ++i) x[i] = uint16_t(0x3c00 + i%256 + (i%3 == 0 ? 0x8000 : 0));
        for (size_t i=0; i<w.size(); ++i) w[i] = uint16_t(0x3c80 + i%256 + (i%5 == 0 ? 0x8000 : 0));
        x[0]=0x4780; // 65536: finite BF16 outside FP16's range.
        std::vector<float> y(t*stride, -777.f);
        for (int row=0;row<t;++row) for (int col=0;col<n;++col) y[row*stride+col]=.125f;
        Buffer dx(x.size()*2),dw(w.size()*2),dy(y.size()*4);
        check(cudaMemcpy(dx.p,x.data(),x.size()*2,cudaMemcpyHostToDevice));
        check(cudaMemcpy(dw.p,w.data(),w.size()*2,cudaMemcpyHostToDevice));
        check(cudaMemcpy(dy.p,y.data(),y.size()*4,cudaMemcpyHostToDevice));
        gemm.bf16((uint16_t*)dx.p,(uint16_t*)dw.p,(float*)dy.p,t,n,k,stride,beta);
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(y.data(),dy.p,y.size()*4,cudaMemcpyDeviceToHost));
        for (int row=0;row<t;++row) for (int col=0;col<stride;++col) {
            if (col>=n) { if (y[row*stride+col]!=-777.f) throw std::runtime_error("padding overwritten"); continue; }
            double ref=beta*.125, bound=0;
            for (int j=0;j<k;++j) { double p=double(decode(x[row*k+j]))*decode(w[col*k+j]);ref+=p;bound+=std::abs(p); }
            if (!std::isfinite(y[row*stride+col]) || std::abs(y[row*stride+col]-ref)>1e-5*(1+bound))
                throw std::runtime_error("BF16 GEMM CPU-reference mismatch");
        }
        std::printf("PASS T=%d N=%d K=%d beta=%.1f\n",t,n,k,beta);
    }
}
