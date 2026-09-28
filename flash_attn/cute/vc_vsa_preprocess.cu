#include "vc_preprocess.cu"

// A BF16 quartet loaded as one uint2 -> four floats (the bits of read_contiguous's aligned branch).
__device__ __forceinline__ void unpack_bf16x4(uint2 bits, float *out) {
    out[0]=__uint_as_float(bits.x<<16);
    out[1]=__uint_as_float(bits.x&0xffff0000u);
    out[2]=__uint_as_float(bits.y<<16);
    out[3]=__uint_as_float(bits.y&0xffff0000u);
}

// prepare_vsa accepts BF16 only; this is the BF16 code of the shared read_input / read_contiguous (vc_preprocess.cu).
constexpr int BF16=0;

__device__ int64_t vsa_index(const void *p, int i, bool wide) {
    return wide ? ((const int64_t*)p)[i] : ((const int*)p)[i];
}

template<int D> __device__ void vsa_stats_impl(
    const void *q, const void *k, const void *v, const void *source_map,
    const void *query_map, const void *sizes, float *stats,
    float *pq, float *pk, float *pv, int source_n, int nblocks, int h,
    int block_size, int query_n, int query_offset, int metadata_mask, float scale) {
    constexpr int W = D/32;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int block = blockIdx.x, head = blockIdx.y, batch = blockIdx.z, bh = batch*h+head;
    float a[8][W];
    #pragma unroll
    for (int j=0; j<W; ++j) {
        #pragma unroll
        for (int m=0; m<8; ++m) a[m][j] = 0;
        a[1][j] = INFINITY; a[2][j] = -INFINITY;
    }
    // One row's accumulation; the BF16 prefetch loop and the generic loop share it, so the per-warp order is identical.
    auto accumulate=[&](const float (&rawq)[W], const float (&rawk)[W], const float (&rawv)[W], bool query_valid) {
        float qr[W], kr[W];
        #pragma unroll
        for (int j=0; j<W; ++j) {
            float qv=rawq[j],kv=rawk[j],vv=rawv[j];
            a[5][j] += qv; a[6][j] += kv; a[7][j] += vv;
            a[4][j] = fmaxf(a[4][j], fabsf(vv));
            qr[j] = query_valid ? qv : 0; kr[j] = kv;
        }
        rotate_contiguous<D>(qr,kr,scale);
        #pragma unroll
        for (int j=0; j<W; ++j) {
            a[0][j] += kr[j];
            a[1][j] = fminf(a[1][j],kr[j]); a[2][j] = fmaxf(a[2][j],kr[j]);
            a[3][j] = fmaxf(a[3][j],fabsf(qr[j]));
        }
    };
    bool prefetch = false;
    if constexpr(D==128) {
    prefetch = ((((unsigned long long)q)|((unsigned long long)k)|((unsigned long long)v))&7)==0;
    if (prefetch) {
        // BF16 D128, 8B-aligned (every lane offset is a multiple of 4 elements): load row+8 while rotating row.
        auto fetch=[&](int row, uint2 &qb, uint2 &kb, uint2 &vb, bool &query_valid) {
            int padded = block*block_size+row;
            int64_t original = vsa_index(source_map,padded,metadata_mask&1), qi = vsa_index(query_map,padded,metadata_mask&2)-query_offset;
            query_valid = qi >= 0 && qi < query_n;
            qb = kb = vb = make_uint2(0,0);
            if (original >= 0 && original < source_n) {
                int64_t base = (((int64_t)batch*source_n+original)*h+head)*D+lane*W;
                qb=*((const uint2*)((const unsigned short*)q+base));
                kb=*((const uint2*)((const unsigned short*)k+base));
                vb=*((const uint2*)((const unsigned short*)v+base));
            }
        };
        uint2 qb, kb, vb; bool query_valid;
        fetch(warp,qb,kb,vb,query_valid);  // block_size >= 128 > warp
        for (int row=warp; row<block_size; row+=8) {
            uint2 nq=make_uint2(0,0), nk=nq, nv=nq; bool nvalid=false;
            if (row+8<block_size) fetch(row+8,nq,nk,nv,nvalid);
            float rawq[W], rawk[W], rawv[W];
            unpack_bf16x4(qb,rawq); unpack_bf16x4(kb,rawk); unpack_bf16x4(vb,rawv);
            accumulate(rawq,rawk,rawv,query_valid);
            qb=nq; kb=nk; vb=nv; query_valid=nvalid;
        }
    }
    }
    for (int row=warp; !prefetch && row<block_size; row+=8) {
        int padded = block*block_size+row;
        int64_t original = vsa_index(source_map,padded,metadata_mask&1), qi = vsa_index(query_map,padded,metadata_mask&2)-query_offset;
        bool valid = original >= 0 && original < source_n;
        bool query_valid = qi >= 0 && qi < query_n;
        int64_t base = (((int64_t)batch*source_n+original)*h+head)*D;
        if constexpr(D==128){
            float rawq[W], rawk[W], rawv[W];
            if(valid){
                read_contiguous<D>(q,base+lane*W,BF16,rawq);
                read_contiguous<D>(k,base+lane*W,BF16,rawk);
                read_contiguous<D>(v,base+lane*W,BF16,rawv);
            }else{
                #pragma unroll
                for(int j=0;j<W;++j){rawq[j]=0;rawk[j]=0;rawv[j]=0;}
            }
            accumulate(rawq,rawk,rawv,query_valid);
        }else{
            float qr[W], kr[W];
            // Preserve scalar load/consume order for short D64 inputs.
            #pragma unroll
            for (int j=0; j<W; ++j) {
                int c = lane*W+j;
                float qv = valid ? read_input(q,base+c,BF16) : 0;
                float kv = valid ? read_input(k,base+c,BF16) : 0;
                float vv = valid ? read_input(v,base+c,BF16) : 0;
                a[5][j] += qv; a[6][j] += kv; a[7][j] += vv;
                a[4][j] = fmaxf(a[4][j], fabsf(vv));
                qr[j] = query_valid ? qv : 0; kr[j] = kv;
            }
            rotate_contiguous<D>(qr,kr,scale);
            #pragma unroll
            for (int j=0; j<W; ++j) {
                a[0][j] += kr[j];
                a[1][j] = fminf(a[1][j],kr[j]); a[2][j] = fmaxf(a[2][j],kr[j]);
                a[3][j] = fmaxf(a[3][j],fabsf(qr[j]));
            }
        }
    }
    __shared__ float partial[8][8][D];
    #pragma unroll
    for (int m=0;m<8;++m) {
        #pragma unroll
        for (int j=0;j<W;++j) partial[m][warp][lane*W+j]=a[m][j];
    }
    __syncthreads();
    int c=threadIdx.x;
    if (c<D) {
        float sums[8];
        #pragma unroll
        for (int m=0;m<8;++m) sums[m]=partial[m][0][c];
        #pragma unroll
        for (int w=1;w<8;++w) {
            sums[0]+=partial[0][w][c];
            sums[1]=fminf(sums[1],partial[1][w][c]);
            sums[2]=fmaxf(sums[2],partial[2][w][c]);
            sums[3]=fmaxf(sums[3],partial[3][w][c]);
            sums[4]=fmaxf(sums[4],partial[4][w][c]);
            sums[5]+=partial[5][w][c]; sums[6]+=partial[6][w][c]; sums[7]+=partial[7][w][c];
        }
        int64_t s=((int64_t)bh*nblocks+block)*6*D+c;
        #pragma unroll
        for(int m=0;m<5;++m) stats[s+m*D]=sums[m];
        stats[s+5*D]=0;
        int64_t out=((int64_t)bh*nblocks+block)*D+c;
        int64_t size=vsa_index(sizes,block,metadata_mask&4);
        float divisor=(float)(size>0 ? size : 1);
        // Match Triton pooling division: tiny rounding changes can flip near-tied top-k routes.
        float inverse; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(divisor));
        pq[out]=sums[5]*inverse; pk[out]=sums[6]*inverse; pv[out]=sums[7]*inverse;
    }
}

extern "C" __global__ void __launch_bounds__(256,4) vsa_stats(
    const void *q,const void *k,const void *v,const void *source_map,
    const void *query_map,const void *sizes,float *stats,float *pq,float *pk,float *pv,
    int source_n,int nblocks,int h,int d,int block_size,int query_n,int query_offset,int metadata_mask) {
    float scale=rsqrtf((float)d);
    if(d==64) vsa_stats_impl<64>(q,k,v,source_map,query_map,sizes,stats,pq,pk,pv,source_n,nblocks,h,block_size,query_n,query_offset,metadata_mask,scale);
    else vsa_stats_impl<128>(q,k,v,source_map,query_map,sizes,stats,pq,pk,pv,source_n,nblocks,h,block_size,query_n,query_offset,metadata_mask,scale);
}

// Rotate, scale and store one token (the arithmetic both vsa_quantize paths share). The scale pointers address the (b,h)
// qs / ks and this lane's W channels of kmean / vs, in global memory or in registers.
template<int D> __device__ __forceinline__ void vsa_quantize_store(
    float (&qr)[D/32],float (&kr)[D/32],float (&vr)[D/32],const float *qscale,const float *kscale,const float *km,const float *vscale,
    float scale,uint8_t *oq,uint8_t *ok,uint8_t *ov,int64_t dst,int64_t qdst,bool query_valid) {
    constexpr int W=D/32;
    rotate_contiguous<D,true>(qr,kr,scale);
    #pragma unroll
    for(int j=0;j<W;++j) {qr[j]/=*qscale; kr[j]=(kr[j]-km[j])/ *kscale; vr[j]/=vscale[j];}
    if constexpr(D==128) {
        if(query_valid) *(unsigned int*)(oq+qdst)=to_fp8_four(qr[0],qr[1],qr[2],qr[3]);
        *(unsigned int*)(ok+dst)=to_fp8_four(kr[0],kr[1],kr[2],kr[3]);
        *(unsigned int*)(ov+dst)=to_fp8_four(vr[0],vr[1],vr[2],vr[3]);
    } else {
        if(query_valid) *(unsigned short*)(oq+qdst)=to_fp8_two(qr[0],qr[1]);
        *(unsigned short*)(ok+dst)=to_fp8_two(kr[0],kr[1]);
        *(unsigned short*)(ov+dst)=to_fp8_two(vr[0],vr[1]);
    }
}

template<int D> __device__ __forceinline__ void vsa_quantize_token(
    const void *q,const void *k,const void *v,const void *source_map,const void *query_map,
    const float *kmean,const float *qs,const float *ks,const float *vs,uint8_t *oq,uint8_t *ok,uint8_t *ov,
    int source_n,int padded_n,int query_n,int query_offset,int h,int metadata_mask,float scale,int token) {
    constexpr int W=D/32;
    int lane=threadIdx.x%32;
    int head=blockIdx.y,batch=blockIdx.z,bh=batch*h+head,c=lane*W;
    int64_t original=vsa_index(source_map,token,metadata_mask&1), qi=vsa_index(query_map,token,metadata_mask&2)-query_offset;
    bool valid=original>=0 && original<source_n;
    bool query_valid=qi>=0 && qi<query_n;
    int64_t src=(((int64_t)batch*source_n+original)*h+head)*D+c;
    float qr[W],kr[W],vr[W];
    if constexpr(D==128){
        if(valid){
            read_contiguous<D>(k,src,BF16,kr);
            read_contiguous<D>(v,src,BF16,vr);
            if(query_valid)read_contiguous<D>(q,src,BF16,qr);
            else{
                #pragma unroll
                for(int j=0;j<W;++j)qr[j]=0;
            }
        }else{
            #pragma unroll
            for(int j=0;j<W;++j){qr[j]=0;kr[j]=0;vr[j]=0;}
        }
    }else{
        #pragma unroll
        for(int j=0;j<W;++j) {
            qr[j]=valid && query_valid ? read_input(q,src+j,BF16) : 0;
            kr[j]=valid ? read_input(k,src+j,BF16) : 0;
            vr[j]=valid ? read_input(v,src+j,BF16) : 0;
        }
    }
    int64_t dst=(((int64_t)batch*padded_n+token)*h+head)*D+c;
    int64_t qdst=(((int64_t)batch*query_n+qi)*h+head)*D+c;
    vsa_quantize_store<D>(qr,kr,vr,qs+bh,ks+bh,kmean+bh*D+c,vs+bh*D+c,scale,oq,ok,ov,dst,qdst,query_valid);
}

template<int D> __device__ void vsa_quantize_impl(
    const void *q,const void *k,const void *v,const void *source_map,const void *query_map,
    const float *kmean,const float *qs,const float *ks,const float *vs,uint8_t *oq,uint8_t *ok,uint8_t *ov,
    int source_n,int padded_n,int query_n,int query_offset,int h,int metadata_mask,float scale) {
    // Grid-stride over tokens (grid.x is capped at 4 x SMs by the launcher); a warp owns every gridDim.x*4-th token.
    // padded_n is a multiple of the 128/256 block (prepare_vsa validates it), so first < padded_n implies first + warp <
    // padded_n: loop control on the CTA's first token is provably warp-uniform and the rotate's shuffles need no
    // collective fallback.
    int warp=threadIdx.x/32, token=blockIdx.x*4+warp, stride=gridDim.x*4;
    if constexpr(D==128) {
        if(((((unsigned long long)q)|((unsigned long long)k)|((unsigned long long)v))&7)==0) {
            // BF16, 8B-aligned: (b,h) scales held in registers, next token's loads issued before this token's rotate.
            int lane=threadIdx.x%32, head=blockIdx.y, batch=blockIdx.z, bh=batch*h+head, c=lane*4;
            float qscale=qs[bh], kscale=ks[bh], km[4], vscale[4];
            #pragma unroll
            for(int j=0;j<4;++j){km[j]=kmean[bh*D+c+j]; vscale[j]=vs[bh*D+c+j];}
            auto fetch=[&](int t, uint2 &qb, uint2 &kb, uint2 &vb, int64_t &qi, bool &query_valid) {
                int64_t original=vsa_index(source_map,t,metadata_mask&1);
                qi=vsa_index(query_map,t,metadata_mask&2)-query_offset;
                query_valid=qi>=0 && qi<query_n;
                qb=kb=vb=make_uint2(0,0);
                if(original>=0 && original<source_n) {
                    int64_t src=(((int64_t)batch*source_n+original)*h+head)*D+c;
                    kb=*((const uint2*)((const unsigned short*)k+src));
                    vb=*((const uint2*)((const unsigned short*)v+src));
                    if(query_valid) qb=*((const uint2*)((const unsigned short*)q+src));
                }
            };
            int first=blockIdx.x*4;
            if(first>=padded_n) return;
            uint2 qb,kb,vb; int64_t qi; bool query_valid;
            fetch(token,qb,kb,vb,qi,query_valid);
            for(;;) {
                bool more=stride<padded_n-first;
                uint2 nq=make_uint2(0,0),nk=nq,nv=nq; int64_t nqi=0; bool nvalid=false;
                if(more) fetch(token+stride,nq,nk,nv,nqi,nvalid);
                float qr[4], kr[4], vr[4];
                unpack_bf16x4(qb,qr); unpack_bf16x4(kb,kr); unpack_bf16x4(vb,vr);
                int64_t dst=(((int64_t)batch*padded_n+token)*h+head)*D+c;
                int64_t qdst=(((int64_t)batch*query_n+qi)*h+head)*D+c;
                vsa_quantize_store<D>(qr,kr,vr,&qscale,&kscale,km,vscale,scale,oq,ok,ov,dst,qdst,query_valid);
                if(!more) return;
                first+=stride; token+=stride; qb=nq; kb=nk; vb=nv; qi=nqi; query_valid=nvalid;
            }
        }
    }
    for(int first=blockIdx.x*4;first<padded_n;first+=stride,token+=stride)  // warp-uniform, as above
        vsa_quantize_token<D>(q,k,v,source_map,query_map,kmean,qs,ks,vs,oq,ok,ov,source_n,padded_n,query_n,query_offset,h,metadata_mask,scale,token);
}

extern "C" __global__ void vsa_quantize(
    const void *q,const void *k,const void *v,const void *source_map,const void *query_map,
    const float *kmean,const float *qs,const float *ks,const float *vs,uint8_t *oq,uint8_t *ok,uint8_t *ov,
    int source_n,int padded_n,int query_n,int query_offset,int h,int d,int metadata_mask) {
    float scale=rsqrtf((float)d);
    if(d==64) vsa_quantize_impl<64>(q,k,v,source_map,query_map,kmean,qs,ks,vs,oq,ok,ov,source_n,padded_n,query_n,query_offset,h,metadata_mask,scale);
    else vsa_quantize_impl<128>(q,k,v,source_map,query_map,kmean,qs,ks,vs,oq,ok,ov,source_n,padded_n,query_n,query_offset,h,metadata_mask,scale);
}

extern "C" __global__ void vsa_reduce(const float *stats, float *kmean,
 float *qs, float *ks, float *vs, int n, int d, int nb) {
 int bh=blockIdx.x, c=threadIdx.x%d, group=threadIdx.x/d, groups=blockDim.x/d;
 __shared__ float partial[5][1024];
 float sk=0, lo=INFINITY, hi=-INFINITY, qm=0, vm=0;
 for(int j=group;j<nb;j+=groups){
  int64_t s=((int64_t)bh*nb+j)*6*d+c;
  sk+=stats[s];lo=fminf(lo,stats[s+d]);hi=fmaxf(hi,stats[s+2*d]);
  qm=fmaxf(qm,stats[s+3*d]);vm=fmaxf(vm,stats[s+4*d]);
 }
 int t=threadIdx.x;partial[0][t]=sk;partial[1][t]=lo;partial[2][t]=hi;
 partial[3][t]=qm;partial[4][t]=vm;__syncthreads();
 if(group==0){
  for(int g=1;g<groups;g++) {int i=g*d+c;
   sk+=partial[0][i];lo=fminf(lo,partial[1][i]);hi=fmaxf(hi,partial[2][i]);
   qm=fmaxf(qm,partial[3][i]);vm=fmaxf(vm,partial[4][i]);
  }
  float mean=sk/n;kmean[bh*d+c]=mean;vs[bh*d+c]=vm>0?vm/448.f:1.f;
  partial[0][c]=qm;partial[1][c]=fmaxf(fabsf(lo-mean),fabsf(hi-mean));
 }
 __syncthreads();
 for(int delta=d/2;delta;delta/=2){
  if(t<delta){partial[0][t]=fmaxf(partial[0][t],partial[0][t+delta]);
   partial[1][t]=fmaxf(partial[1][t],partial[1][t+delta]);}
  __syncthreads();
 }
 if(t==0){qs[bh]=partial[0][0]>0?partial[0][0]/448.f:1.f;
 ks[bh]=partial[1][0]>0?partial[1][0]/448.f:1.f;}
}
