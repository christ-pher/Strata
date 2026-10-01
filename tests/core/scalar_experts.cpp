#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/artifact/dequant.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
int main() {
    namespace c = strata::kernels::cpu;
    constexpr int width=128, rows=7, tokens=3;
    std::vector<uint8_t> w(rows*36);
    for (int r=0;r<rows;++r) for (int b=0;b<2;++b) {
        w[r*36+b*18]=0;w[r*36+b*18+1]=0x3c; // fp16 scale 1
        for (int j=0;j<16;++j) w[r*36+b*18+2+j]=(uint8_t)(r*17+b*31+j*7);
    }
    c::ActQ acts[tokens]; const c::ActQ* ap[tokens];
    float values[tokens][width], output[tokens][rows]; float* op[tokens];
    for(int t=0;t<tokens;++t) {
        for(int i=0;i<width;++i) values[t][i]=(i%17-8)*(t+1)*.03125f;
        c::act_quant_any(values[t],width,acts[t]);ap[t]=&acts[t];op[t]=output[t];
        for(int k=0;k<width/32;++k) {
            int sum=0;
            for(int j=0;j<32;++j) sum+=acts[t].q[k*32+j];
            if(sum!=acts[t].sum[k] || acts[t].hx[k]!=acts[t].scale[k]*sum) throw std::runtime_error("activation sums");
        }
    }
    c::q2_rows_any(w.data(),36,2,ap,tokens,op,0,rows);
    for(int t=0;t<tokens;++t) for(int r=0;r<rows;++r) {
        double ref=0;
        for(int i=0;i<width;++i) {
            int code=(w[r*36+(i/64)*18+2+(i%64)/4]>>(2*(i%4)))&3;
            ref+=(code-1)*acts[t].q[i]*double(acts[t].scale[i/32]);
        }
        if(std::abs(ref-output[t][r])>1e-5*(1+std::abs(ref))) throw std::runtime_error("Q2 scalar reference mismatch");
    }
    std::printf("PASS CPU Q2 rows and activation sums (%s)\n",c::cpu_avx2_ok()?"SIMD":"scalar");
}
