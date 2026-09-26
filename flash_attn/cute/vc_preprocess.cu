// Native CUDA preprocessing for FP8 VC-Attention. Inputs are contiguous BHND or BSHD.
typedef long long int64_t;
typedef unsigned char uint8_t;
#define INFINITY __int_as_float(0x7f800000)

__device__ float read_input(const void *p, int64_t i, int dtype) {
    if (dtype == 0) return __uint_as_float(((unsigned int)((const unsigned short*)p)[i]) << 16);
    if (dtype == 1) {
        float value;
        asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(((const unsigned short*)p)[i]));
        return value;
    }
    return ((const float*)p)[i];
}

// Native uint2 BF16 quartet-load idiom: FlashInfer vec_dtypes.cuh,
// vec_t<nv_bfloat16, 4>::load, 0e4c173821a0aca29e9eb00c50c3bde8696e6dc6.
// Guard the actual address: contiguous storage-offset views need not align to 8B.
template<int D> __device__ __forceinline__ void read_contiguous(
    const void *p, int64_t i, int dtype, float *out) {
    if constexpr(D==128) {
        if(dtype==0) {
            const unsigned short *address=((const unsigned short*)p)+i;
            if((((unsigned long long)address)&7)==0) {
                uint2 bits=*((const uint2*)address);
                out[0]=__uint_as_float(bits.x<<16);
                out[1]=__uint_as_float(bits.x&0xffff0000u);
                out[2]=__uint_as_float(bits.y<<16);
                out[3]=__uint_as_float(bits.y&0xffff0000u);
                return;
            }
        }
    }
    #pragma unroll
    for(int j=0;j<D/32;++j)out[j]=read_input(p,i+j,dtype);
}

__device__ uint8_t to_fp8(float value) {
    unsigned short pair;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(pair) : "f"(value));
    return pair & 255;
}

__device__ int64_t input_index(int bh, int t, int c, int n, int h, int d, bool bshd) {
    return bshd ? (((int64_t)(bh / h) * n + t) * h + bh % h) * d + c
                : ((int64_t)bh * n + t) * d + c;
}

// One warp per token. The first five butterfly stages use shuffle-xor;
// the final one/two stages exchange the per-lane registers for D=64/128.
template<int d> __device__ void rotate_qk_impl(const void *q, const void *k, float *oq, float *ok,
                                     int dtype, int n, int h, bool bshd, int64_t tokens, float scale) {
    int lane = threadIdx.x % 32;
    int64_t token = (int64_t)blockIdx.x * 4 + threadIdx.x / 32;
    if (token >= tokens) return;
    int64_t source = input_index(token / n, token % n, 0, n, h, d, bshd);
    float qv[4], kv[4];
    constexpr int width = d / 32;
    #pragma unroll
    for (int j=0;j<width;++j) {
        qv[j] = read_input(q,source+lane+32*j,dtype);
        kv[j] = read_input(k,source+lane+32*j,dtype);
    }
    #pragma unroll
    for (int bit=1;bit<32;bit*=2) {
        #pragma unroll
        for (int j=0;j<width;++j) {
            float qp = __shfl_xor_sync(0xffffffff,qv[j],bit);
            float kp = __shfl_xor_sync(0xffffffff,kv[j],bit);
            qv[j] = lane & bit ? qp-qv[j] : qp+qv[j];
            kv[j] = lane & bit ? kp-kv[j] : kp+kv[j];
        }
    }
    #pragma unroll
    for (int bit=1;bit<width;bit*=2) {
        #pragma unroll
        for (int j=0;j<width;++j) {
            if (!(j&bit)) {
                float qa=qv[j],qb=qv[j|bit],ka=kv[j],kb=kv[j|bit];
                qv[j]=qa+qb;qv[j|bit]=qa-qb;
                kv[j]=ka+kb;kv[j|bit]=ka-kb;
            }
        }
    }
    #pragma unroll
    for (int j=0;j<width;++j) {
        oq[token*d+lane+32*j]=qv[j]*scale;
        ok[token*d+lane+32*j]=kv[j]*scale;
    }
}

extern "C" __global__ void rotate_qk(const void *q, const void *k, float *oq, float *ok,
                                     int d, int dtype, int n, int h, bool bshd, int64_t tokens) {
    float scale = rsqrtf((float)d);
    if (d == 64) rotate_qk_impl<64>(q, k, oq, ok, dtype, n, h, bshd, tokens, scale);
    else rotate_qk_impl<128>(q, k, oq, ok, dtype, n, h, bshd, tokens, scale);
}

// Each lane owns a channel; all reads of a token are coalesced.
extern "C" __global__ void block_stats(const void *q, const void *k, const void *v,
                           const int64_t *perm, float *stats,
                           int n, int d, int nb, int h, int qkdtype, int vdtype, bool smooth, bool qk_bshd, bool v_bshd) {
    int bh = blockIdx.x / nb, block = blockIdx.x % nb, c = threadIdx.x;
    if (c >= d) return;
    int start = block * 128, count = min(128, n - start);
    float sk = 0, sv = 0, vmax = 0, kmin = INFINITY, kmax = -INFINITY, qmax = 0;
    for (int r = 0; r < count; ++r) {
        int t = start + r;
        int64_t base = input_index(bh, t, c, n, h, d, qk_bshd);
        float kv = read_input(k, base, qkdtype);
        sk += kv; kmin = fminf(kmin, kv); kmax = fmaxf(kmax, kv);
        qmax = fmaxf(qmax, fabsf(read_input(q, base, qkdtype)));
        int64_t vt = perm ? perm[(int64_t)bh * n + t] : t;
        float vv = read_input(v, input_index(bh, vt, c, n, h, d, v_bshd), vdtype);
        if (smooth) sv += vv;
        else vmax = fmaxf(vmax, fabsf(vv));
    }
    float mean = smooth ? sv / count : 0;
    for (int r = 0; smooth && r < count; ++r) {
        int t = start + r;
        int64_t vt = perm ? perm[(int64_t)bh * n + t] : t;
        vmax = fmaxf(vmax, fabsf(read_input(v, input_index(bh, vt, c, n, h, d, v_bshd), vdtype) - mean));
    }
    int64_t s = ((int64_t)bh * nb + block) * 6 * d + c;
    stats[s] = sk; stats[s + d] = kmin; stats[s + 2*d] = kmax;
    stats[s + 3*d] = qmax; stats[s + 4*d] = vmax; stats[s + 5*d] = mean;
}

extern "C" __global__ void reduce_stats(const float *stats, void *means, float *kmean,
                            float *qs, float *ks, float *vs, int n, int d, int nb, bool means_fp32) {
    int bh=blockIdx.x, tid=threadIdx.x, c=tid%d, group=tid/d, groups=blockDim.x/d;
    __shared__ float parts[4][1024], qreduce[128], kreduce[128], scales[128];
    float lo=INFINITY, hi=-INFINITY, qm=0, vm=0, sk=0;
    for (int j=group;j<nb;j+=groups) {
        int64_t s=((int64_t)bh*nb+j)*6*d+c;
        lo=fminf(lo,stats[s+d]);hi=fmaxf(hi,stats[s+2*d]);
        qm=fmaxf(qm,stats[s+3*d]);vm=fmaxf(vm,stats[s+4*d]);
    }
    parts[0][tid]=lo;parts[1][tid]=hi;parts[2][tid]=qm;parts[3][tid]=vm;
    // Keep centering's original accumulation order to preserve FP8 decisions.
    if (group==0) for (int j=0;j<nb;++j) sk+=stats[((int64_t)bh*nb+j)*6*d+c];
    __syncthreads();
    if (group==0) {
        for (int g=1;g<groups;++g) {
            lo=fminf(lo,parts[0][g*d+c]);hi=fmaxf(hi,parts[1][g*d+c]);
            qm=fmaxf(qm,parts[2][g*d+c]);vm=fmaxf(vm,parts[3][g*d+c]);
        }
        float km=sk/n, scale=vm>0?vm/448.f:1.f;
        kmean[bh*d+c]=km;vs[bh*d+c]=scale;scales[c]=scale;
        qreduce[c]=qm;kreduce[c]=fmaxf(fabsf(lo-km),fabsf(hi-km));
    }
    __syncthreads();
    for (int j=group;j<nb;j+=groups) {
        int64_t index=((int64_t)bh*nb+j)*d+c;
        float normalized=stats[((int64_t)bh*nb+j)*6*d+5*d+c]/scales[c];
        if (means_fp32) ((float*)means)[index]=normalized;
        else {unsigned short stored;asm("cvt.rn.bf16.f32 %0, %1;":"=h"(stored):"f"(normalized));((unsigned short*)means)[index]=stored;}
    }
    for (int delta=d/2;delta;delta/=2) {
        if (tid<delta) {qreduce[tid]=fmaxf(qreduce[tid],qreduce[tid+delta]);kreduce[tid]=fmaxf(kreduce[tid],kreduce[tid+delta]);}
        __syncthreads();
    }
    if (tid==0) {qs[bh]=qreduce[0]>0?qreduce[0]/448.f:1.f;ks[bh]=kreduce[0]>0?kreduce[0]/448.f:1.f;}
}

__device__ unsigned int to_fp8_four(float x0,float x1,float x2,float x3) {
    unsigned int packed;
    asm("{.reg .b16 lo,hi; cvt.rn.satfinite.e4m3x2.f32 lo,%2,%1; cvt.rn.satfinite.e4m3x2.f32 hi,%4,%3; mov.b32 %0,{lo,hi};}":"=r"(packed):"f"(x0),"f"(x1),"f"(x2),"f"(x3));
    return packed;
}

template<int d> __device__ void quantize_impl(const void *q, const void *k, const void *v,
                         const int64_t *perm, const float *kmean,
                         const float *qs, const float *ks, const float *vs,
                         const float *stats, uint8_t *oq, uint8_t *ok, uint8_t *ov,
                         int n, int h, int nb, int b, int qkdtype, int vdtype, bool qk_bshd, bool v_bshd, int64_t total) {
    int64_t offset=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    int batch=blockIdx.z,head=blockIdx.y;
    if (b>65535 || h>65535) {
        int64_t blocks_per_head=((int64_t)n*(d/4)+blockDim.x-1)/blockDim.x;
        int bh=blockIdx.x/blocks_per_head;batch=bh/h;head=bh%h;
        offset=(blockIdx.x%blocks_per_head)*blockDim.x+threadIdx.x;
    }
    int t=offset/(d/4), c=4*(offset%(d/4));if(t>=n)return;
    int bh=batch*h+head;
    int64_t pt=perm?perm[(int64_t)bh*n+t]:t;
    int64_t dst=(((int64_t)batch*n+t)*h+head)*d+c;
    int64_t token_dst=(((int64_t)batch*n+pt)*h+head)*d+c;
    int64_t qi=qk_bshd?dst:((int64_t)bh*n+t)*d+c;
    int64_t ki=qk_bshd?token_dst:((int64_t)bh*n+pt)*d+c;
    int64_t vi=v_bshd?token_dst:((int64_t)bh*n+pt)*d+c;
    float qv[4],kv[4],vv[4];
    #pragma unroll
    for(int j=0;j<4;++j){
        float scale=vs[bh*d+c+j];
        float mean=stats[((int64_t)bh*nb+t/128)*6*d+5*d+c+j];
        qv[j]=read_input(q,qi+j,qkdtype)/qs[bh];
        kv[j]=(read_input(k,ki+j,qkdtype)-kmean[bh*d+c+j])/ks[bh];
        vv[j]=(read_input(v,vi+j,vdtype)-mean)/scale;
    }
    *(unsigned int*)(oq+dst)=to_fp8_four(qv[0],qv[1],qv[2],qv[3]);
    *(unsigned int*)(ok+dst)=to_fp8_four(kv[0],kv[1],kv[2],kv[3]);
    *(unsigned int*)(ov+dst)=to_fp8_four(vv[0],vv[1],vv[2],vv[3]);
}

extern "C" __global__ void quantize(const void *q, const void *k, const void *v,
                         const int64_t *perm, const float *kmean,
                         const float *qs, const float *ks, const float *vs,
                         const float *stats, uint8_t *oq, uint8_t *ok, uint8_t *ov,
                         int n, int h, int d, int nb, int b, int qkdtype, int vdtype, bool qk_bshd, bool v_bshd, int64_t total) {
    if (d == 64) quantize_impl<64>(q, k, v, perm, kmean, qs, ks, vs, stats, oq, ok, ov,
                                  n, h, nb, b, qkdtype, vdtype, qk_bshd, v_bshd, total);
    else quantize_impl<128>(q, k, v, perm, kmean, qs, ks, vs, stats, oq, ok, ov,
                            n, h, nb, b, qkdtype, vdtype, qk_bshd, v_bshd, total);
}

// Same logical butterfly-bit order as rotate_qk_impl; contiguous channels per lane.
template<int D, bool SignedFma=false> __device__ void rotate_contiguous(float (&q)[D/32],float (&k)[D/32],float scale) {
    constexpr int W=D/32;int lane=threadIdx.x%32;
    #pragma unroll
    for(int bit=1;bit<W;bit*=2){
        #pragma unroll
        for(int j=0;j<W;++j) if(!(j&bit)){
            float qa=q[j],qb=q[j|bit],ka=k[j],kb=k[j|bit];
            q[j]=qa+qb;q[j|bit]=qa-qb;k[j]=ka+kb;k[j|bit]=ka-kb;
        }
    }
    #pragma unroll
    for(int bit=1;bit<32;bit*=2){
        #pragma unroll
        for(int j=0;j<W;++j){
            float qp=__shfl_xor_sync(0xffffffff,q[j],bit),kp=__shfl_xor_sync(0xffffffff,k[j],bit);
            if constexpr(SignedFma) {
                float sign=lane&bit ? -1.0f : 1.0f;
                q[j]=__fmaf_rn(sign,q[j],qp);k[j]=__fmaf_rn(sign,k[j],kp);
            } else {
                q[j]=lane&bit?qp-q[j]:qp+q[j];k[j]=lane&bit?kp-k[j]:kp+k[j];
            }
        }
    }
    #pragma unroll
    for(int j=0;j<W;++j){q[j]*=scale;k[j]*=scale;}
}

// Reuse D128 slabs while retaining ascending global row order for K additions.
template<int D> __device__ void fused_stats_impl(const void *q,const void *k,const void *v,float *stats,
    int n,int nb,int h,int dtype,bool bshd,float scale){
    constexpr int W=D/32,ROWS=D==128?32:128;int bh=blockIdx.x/nb,block=blockIdx.x%nb;
    int lane=threadIdx.x%32,warp=threadIdx.x/32,start=block*128,count=min(128,n-start);
    extern __shared__ float sm[];float *keys=sm,*qm=sm+ROWS*D,*vm=qm+8*D;
    float qmax[W],vmax[W];
    #pragma unroll
    for(int j=0;j<W;++j){qmax[j]=0;vmax[j]=0;}
    int c=threadIdx.x;
    float sk=0,lo=INFINITY,hi=-INFINITY;
    int64_t row_base=input_index(bh,start,lane*W,n,h,D,bshd);
    int64_t row_stride=(bshd?(int64_t)h:1)*D;
    for(int slab=0;slab<128;slab+=ROWS){
        for(int local=warp;local<ROWS;local+=8){
            int row=slab+local;
            int64_t base=row_base+row*row_stride;
            float qr[W],kr[W],vr[W];
            if(row<count){
                read_contiguous<D>(q,base,dtype,qr);
                read_contiguous<D>(k,base,dtype,kr);
                read_contiguous<D>(v,base,dtype,vr);
            }else{
                #pragma unroll
                for(int j=0;j<W;++j){qr[j]=0;kr[j]=0;vr[j]=0;}
            }
            #pragma unroll
            for(int j=0;j<W;++j)vmax[j]=fmaxf(vmax[j],fabsf(vr[j]));
            rotate_contiguous<D>(qr,kr,scale);
            #pragma unroll
            for(int j=0;j<W;++j){keys[local*D+lane*W+j]=kr[j];qmax[j]=fmaxf(qmax[j],fabsf(qr[j]));}
        }
        if(slab+ROWS==128){
            #pragma unroll
            for(int j=0;j<W;++j){qm[warp*D+lane*W+j]=qmax[j];vm[warp*D+lane*W+j]=vmax[j];}
        }
        __syncthreads();
        if(c<D){
            for(int local=0;local<min(ROWS,count-slab);++local){
                float kv=keys[local*D+c];sk+=kv;lo=fminf(lo,kv);hi=fmaxf(hi,kv);
            }
        }
        if(slab+ROWS<128)__syncthreads();
    }
    if(c<D){
        float aq=qm[c],av=vm[c];
        #pragma unroll
        for(int w=1;w<8;++w){aq=fmaxf(aq,qm[w*D+c]);av=fmaxf(av,vm[w*D+c]);}
        int64_t s=((int64_t)bh*nb+block)*6*D+c;
        stats[s]=sk;stats[s+D]=lo;stats[s+2*D]=hi;stats[s+3*D]=aq;stats[s+4*D]=av;stats[s+5*D]=0;
    }
}
extern "C" __global__ __launch_bounds__(256, 6) void fused_stats(const void *q,const void *k,const void *v,
 const int64_t *perm,float *stats,int n,int d,int nb,int h,int qkdtype,int vdtype,bool smooth,bool qk_bshd,bool v_bshd){
    float scale=rsqrtf((float)d);
    if(d==64)fused_stats_impl<64>(q,k,v,stats,n,nb,h,qkdtype,qk_bshd,scale);
    else fused_stats_impl<128>(q,k,v,stats,n,nb,h,qkdtype,qk_bshd,scale);
}

__device__ unsigned short to_fp8_two(float x0,float x1){unsigned short out;asm("cvt.rn.satfinite.e4m3x2.f32 %0,%2,%1;":"=h"(out):"f"(x0),"f"(x1));return out;}
template<int D> __device__ void fused_quant_impl(const void *q,const void *k,const void *v,
 const float *kmean,const float *qs,const float *ks,const float *vs,uint8_t *oq,uint8_t *ok,uint8_t *ov,
 int n,int h,int dtype,bool bshd,float scale){
    constexpr int W=D/32;int lane=threadIdx.x%32,t=blockIdx.x*8+threadIdx.x/32;
    if(t>=n)return;
    int head=blockIdx.y,batch=blockIdx.z,bh=batch*h+head,c=lane*W;
    int64_t dst=(((int64_t)batch*n+t)*h+head)*D+c,src=bshd?dst:((int64_t)bh*n+t)*D+c;
    float qr[W],kr[W],vr[W];
    read_contiguous<D>(q,src,dtype,qr);
    read_contiguous<D>(k,src,dtype,kr);
    read_contiguous<D>(v,src,dtype,vr);
    rotate_contiguous<D,true>(qr,kr,scale);
    #pragma unroll
    for(int j=0;j<W;++j){qr[j]/=qs[bh];kr[j]=(kr[j]-kmean[bh*D+c+j])/ks[bh];vr[j]/=vs[bh*D+c+j];}
    if constexpr(D==128){
        *(unsigned int*)(oq+dst)=to_fp8_four(qr[0],qr[1],qr[2],qr[3]);
        *(unsigned int*)(ok+dst)=to_fp8_four(kr[0],kr[1],kr[2],kr[3]);
        *(unsigned int*)(ov+dst)=to_fp8_four(vr[0],vr[1],vr[2],vr[3]);
    }else{
        *(unsigned short*)(oq+dst)=to_fp8_two(qr[0],qr[1]);
        *(unsigned short*)(ok+dst)=to_fp8_two(kr[0],kr[1]);
        *(unsigned short*)(ov+dst)=to_fp8_two(vr[0],vr[1]);
    }
}
extern "C" __global__ void fused_quantize(const void *q,const void *k,const void *v,const int64_t *perm,
 const float *kmean,const float *qs,const float *ks,const float *vs,const float *stats,uint8_t *oq,uint8_t *ok,uint8_t *ov,
 int n,int h,int d,int nb,int b,int qkdtype,int vdtype,bool qk_bshd,bool v_bshd,int64_t total){
    float scale=rsqrtf((float)d);
    if(d==64)fused_quant_impl<64>(q,k,v,kmean,qs,ks,vs,oq,ok,ov,n,h,qkdtype,qk_bshd,scale);
    else fused_quant_impl<128>(q,k,v,kmean,qs,ks,vs,oq,ok,ov,n,h,qkdtype,qk_bshd,scale);
}
