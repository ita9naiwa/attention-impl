#include "vc_preprocess.cu"

// Stats D128: BF16 with all three addresses 8B-aligned issues the Q/K/V uint2 loads before any unpack
// (same bits as read_contiguous); anything else uses read_contiguous per input.
template<int D> __device__ __forceinline__ void read_qkv_contiguous(
    const void *q, const void *k, const void *v, int64_t i, int dtype,
    float *rawq, float *rawk, float *rawv) {
    if constexpr(D==128) {
        if(dtype==0) {
            const unsigned short *qa=((const unsigned short*)q)+i;
            const unsigned short *ka=((const unsigned short*)k)+i;
            const unsigned short *va=((const unsigned short*)v)+i;
            if(((((unsigned long long)qa)|((unsigned long long)ka)|((unsigned long long)va))&7)==0) {
                uint2 qb=*((const uint2*)qa);
                uint2 kb=*((const uint2*)ka);
                uint2 vb=*((const uint2*)va);
                rawq[0]=__uint_as_float(qb.x<<16);
                rawq[1]=__uint_as_float(qb.x&0xffff0000u);
                rawq[2]=__uint_as_float(qb.y<<16);
                rawq[3]=__uint_as_float(qb.y&0xffff0000u);
                rawk[0]=__uint_as_float(kb.x<<16);
                rawk[1]=__uint_as_float(kb.x&0xffff0000u);
                rawk[2]=__uint_as_float(kb.y<<16);
                rawk[3]=__uint_as_float(kb.y&0xffff0000u);
                rawv[0]=__uint_as_float(vb.x<<16);
                rawv[1]=__uint_as_float(vb.x&0xffff0000u);
                rawv[2]=__uint_as_float(vb.y<<16);
                rawv[3]=__uint_as_float(vb.y&0xffff0000u);
                return;
            }
        }
    }
    read_contiguous<D>(q,i,dtype,rawq);
    read_contiguous<D>(k,i,dtype,rawk);
    read_contiguous<D>(v,i,dtype,rawv);
}

__device__ int64_t vsa_index(const void *p, int i, bool wide) {
    return wide ? ((const int64_t*)p)[i] : ((const int*)p)[i];
}

template<int D> __device__ void vsa_stats_impl(
    const void *q, const void *k, const void *v, const void *source_map,
    const void *query_map, const void *sizes, float *stats,
    float *pq, float *pk, float *pv, int source_n, int nblocks, int h,
    int block_size, int query_n, int query_offset, int dtype, int metadata_mask, float scale) {
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
    prefetch = dtype==0 && ((((unsigned long long)q)|((unsigned long long)k)|((unsigned long long)v))&7)==0;
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
        // Loop control on r0 only (block_size is a multiple of 8), so it is provably warp-uniform and the rotate's
        // shuffles need no collective fallback; each warp still takes rows warp, warp+8, ... in order.
        for (int r0=0; r0<block_size; r0+=8) {
            int row=r0+warp;
            uint2 nq=make_uint2(0,0), nk=nq, nv=nq; bool nvalid=false;
            if (r0+8<block_size) fetch(row+8,nq,nk,nv,nvalid);
            float rawq[W]={__uint_as_float(qb.x<<16),__uint_as_float(qb.x&0xffff0000u),__uint_as_float(qb.y<<16),__uint_as_float(qb.y&0xffff0000u)};
            float rawk[W]={__uint_as_float(kb.x<<16),__uint_as_float(kb.x&0xffff0000u),__uint_as_float(kb.y<<16),__uint_as_float(kb.y&0xffff0000u)};
            float rawv[W]={__uint_as_float(vb.x<<16),__uint_as_float(vb.x&0xffff0000u),__uint_as_float(vb.y<<16),__uint_as_float(vb.y&0xffff0000u)};
            accumulate(rawq,rawk,rawv,query_valid);
            qb=nq; kb=nk; vb=nv; query_valid=nvalid;
        }
    }
    }
    for (int r0=0; !prefetch && r0<block_size; r0+=8) {
        int row = r0+warp, padded = block*block_size+row;
        int64_t original = vsa_index(source_map,padded,metadata_mask&1), qi = vsa_index(query_map,padded,metadata_mask&2)-query_offset;
        bool valid = original >= 0 && original < source_n;
        bool query_valid = qi >= 0 && qi < query_n;
        int64_t base = (((int64_t)batch*source_n+original)*h+head)*D;
        if constexpr(D==128){
            float rawq[W], rawk[W], rawv[W];
            if(valid){
                read_qkv_contiguous<D>(q,k,v,base+lane*W,dtype,rawq,rawk,rawv);
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
                float qv = valid ? read_input(q,base+c,dtype) : 0;
                float kv = valid ? read_input(k,base+c,dtype) : 0;
                float vv = valid ? read_input(v,base+c,dtype) : 0;
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
    int source_n,int nblocks,int h,int d,int block_size,int query_n,int query_offset,int dtype,int metadata_mask) {
    float scale=rsqrtf((float)d);
    if(d==64) vsa_stats_impl<64>(q,k,v,source_map,query_map,sizes,stats,pq,pk,pv,source_n,nblocks,h,block_size,query_n,query_offset,dtype,metadata_mask,scale);
    else vsa_stats_impl<128>(q,k,v,source_map,query_map,sizes,stats,pq,pk,pv,source_n,nblocks,h,block_size,query_n,query_offset,dtype,metadata_mask,scale);
}

template<int D> __device__ __forceinline__ void vsa_quantize_token(
    const void *q,const void *k,const void *v,const void *source_map,const void *query_map,
    const float *kmean,const float *qs,const float *ks,const float *vs,uint8_t *oq,uint8_t *ok,uint8_t *ov,
    int source_n,int padded_n,int query_n,int query_offset,int h,int dtype,int metadata_mask,float scale,int token) {
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
            read_contiguous<D>(k,src,dtype,kr);
            read_contiguous<D>(v,src,dtype,vr);
            if(query_valid)read_contiguous<D>(q,src,dtype,qr);
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
            qr[j]=valid && query_valid ? read_input(q,src+j,dtype) : 0;
            kr[j]=valid ? read_input(k,src+j,dtype) : 0;
            vr[j]=valid ? read_input(v,src+j,dtype) : 0;
        }
    }
    rotate_contiguous<D,true>(qr,kr,scale);
    #pragma unroll
    for(int j=0;j<W;++j) {
        qr[j]/=qs[bh]; kr[j]=(kr[j]-kmean[bh*D+c+j])/ks[bh]; vr[j]/=vs[bh*D+c+j];
    }
    int64_t dst=(((int64_t)batch*padded_n+token)*h+head)*D+c;
    int64_t qdst=(((int64_t)batch*query_n+qi)*h+head)*D+c;
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

template<int D> __device__ void vsa_quantize_impl(
    const void *q,const void *k,const void *v,const void *source_map,const void *query_map,
    const float *kmean,const float *qs,const float *ks,const float *vs,uint8_t *oq,uint8_t *ok,uint8_t *ov,
    int source_n,int padded_n,int query_n,int query_offset,int h,int dtype,int metadata_mask,float scale) {
    // Grid-stride over tokens (grid.x is capped at 4 x SMs by the launcher); a warp owns every gridDim.x*4-th token.
    // padded_n is a multiple of the 128/256 block (prepare_vsa validates it), so first < padded_n implies first + warp <
    // padded_n: loop control on the CTA's first token is provably warp-uniform and the rotate's shuffles need no
    // collective fallback.
    int warp=threadIdx.x/32, token=blockIdx.x*4+warp, stride=gridDim.x*4;
    if constexpr(D==128) {
        if(dtype==0 && ((((unsigned long long)q)|((unsigned long long)k)|((unsigned long long)v))&7)==0) {
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
                float qr[4]={__uint_as_float(qb.x<<16),__uint_as_float(qb.x&0xffff0000u),__uint_as_float(qb.y<<16),__uint_as_float(qb.y&0xffff0000u)};
                float kr[4]={__uint_as_float(kb.x<<16),__uint_as_float(kb.x&0xffff0000u),__uint_as_float(kb.y<<16),__uint_as_float(kb.y&0xffff0000u)};
                float vr[4]={__uint_as_float(vb.x<<16),__uint_as_float(vb.x&0xffff0000u),__uint_as_float(vb.y<<16),__uint_as_float(vb.y&0xffff0000u)};
                rotate_contiguous<D,true>(qr,kr,scale);
                #pragma unroll
                for(int j=0;j<4;++j) {qr[j]/=qscale; kr[j]=(kr[j]-km[j])/kscale; vr[j]/=vscale[j];}
                int64_t dst=(((int64_t)batch*padded_n+token)*h+head)*D+c;
                int64_t qdst=(((int64_t)batch*query_n+qi)*h+head)*D+c;
                if(query_valid) *(unsigned int*)(oq+qdst)=to_fp8_four(qr[0],qr[1],qr[2],qr[3]);
                *(unsigned int*)(ok+dst)=to_fp8_four(kr[0],kr[1],kr[2],kr[3]);
                *(unsigned int*)(ov+dst)=to_fp8_four(vr[0],vr[1],vr[2],vr[3]);
                if(!more) return;
                first+=stride; token+=stride; qb=nq; kb=nk; vb=nv; qi=nqi; query_valid=nvalid;
            }
        }
    }
    for(int first=blockIdx.x*4;first<padded_n;first+=stride,token+=stride)  // warp-uniform, as above
        vsa_quantize_token<D>(q,k,v,source_map,query_map,kmean,qs,ks,vs,oq,ok,ov,source_n,padded_n,query_n,query_offset,h,dtype,metadata_mask,scale,token);
}

extern "C" __global__ void vsa_quantize(
    const void *q,const void *k,const void *v,const void *source_map,const void *query_map,
    const float *kmean,const float *qs,const float *ks,const float *vs,uint8_t *oq,uint8_t *ok,uint8_t *ov,
    int source_n,int padded_n,int query_n,int query_offset,int h,int d,int dtype,int metadata_mask) {
    float scale=rsqrtf((float)d);
    if(d==64) vsa_quantize_impl<64>(q,k,v,source_map,query_map,kmean,qs,ks,vs,oq,ok,ov,source_n,padded_n,query_n,query_offset,h,dtype,metadata_mask,scale);
    else vsa_quantize_impl<128>(q,k,v,source_map,query_map,kmean,qs,ks,vs,oq,ok,ov,source_n,padded_n,query_n,query_offset,h,dtype,metadata_mask,scale);
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

// Selected IDs are the unchanged caller top-k result. Keep logical parent
// classification before expanding 256-token parents into both 128-token children.
__device__ __forceinline__ int vsa_route_parent(
 const int64_t *selected, int64_t row, int topk, int item,
 int prefix, int document_start, int parents) {
 if (item < prefix) return item;
 int64_t global = selected[row * topk + item - prefix];
 if (global < (int64_t)document_start + prefix ||
     global >= (int64_t)document_start + parents) return -1;
 return (int)(global - document_start);
}

extern "C" __global__ void vsa_routes(
 const int64_t *selected, const int *sizes, int *full_idx, int *full_cnt,
 int *mask_idx, int *mask_cnt, int rows, int topk, int prefix,
 int document_start, int block_size, int capacity, int parents) {
 int row=blockIdx.x, tid=threadIdx.x;
 if (row>=rows) return;
 int factor=block_size/128, total=prefix+topk;
 int64_t offset=(int64_t)row*capacity;
 for (int j=tid;j<capacity;j+=blockDim.x) {
  full_idx[offset+j]=-1; mask_idx[offset+j]=-1;
 }
 __syncthreads();
 if (tid==0) {
  int nf=0,nm=0;
  for (int j=0;j<total;++j) {
   int parent=vsa_route_parent(selected,row,topk,j,prefix,document_start,parents);
   if (parent<0) continue;
   int size=sizes[parent]; nf+=size==block_size; nm+=size>0&&size<block_size;
  }
  full_cnt[row]=nf*factor; mask_cnt[row]=nm*factor;
 }
 for (int j=tid;j<total;j+=blockDim.x) {
  int parent=vsa_route_parent(selected,row,topk,j,prefix,document_start,parents);
  if (parent<0) continue;
  int size=sizes[parent],kind=size==block_size?1:(size>0&&size<block_size?2:0);
  if (!kind) continue;
  int rank=0;
  for (int k=0;k<total;++k) {
   int other=vsa_route_parent(selected,row,topk,k,prefix,document_start,parents);
   if (other<0) continue;
   int os=sizes[other],okind=os==block_size?1:(os>0&&os<block_size?2:0);
   rank+=(okind==kind)&&(other<parent);
  }
  int *out=kind==1?full_idx:mask_idx;
  for (int child=0;child<factor;++child)
   out[offset+rank*factor+child]=parent*factor+child;
 }
}

extern "C" __global__ void vsa_routes_sorted(
 const int64_t *selected, const int *sizes, int *full_idx, int *full_cnt,
 int *mask_idx, int *mask_cnt, int rows, int topk, int prefix,
 int document_start, int block_size, int capacity, int parents) {
 int row=blockIdx.x, tid=threadIdx.x;
 __shared__ int keys[1024]; __shared__ int nf,nm;
 int total=prefix+topk, n=1; while(n<total)n*=2;
 for(int i=tid;i<n;i+=blockDim.x){
  int p=i<total?vsa_route_parent(selected,row,topk,i,prefix,document_start,parents):-1;
  int sz=p>=0?sizes[p]:0;
  keys[i]=sz==block_size?p:(sz>0&&sz<block_size?p+parents:2147483647);
 }
 __syncthreads();
 for(int k=2;k<=n;k*=2)for(int j=k/2;j>0;j/=2){
  for(int i=tid;i<n;i+=blockDim.x){int other=i^j;
   if(other>i){int a=keys[i],b=keys[other]; if((a>b)==((i&k)==0)){keys[i]=b;keys[other]=a;}}
  } __syncthreads();
 }
 if(tid==0){nf=0;nm=0;for(int i=0;i<total;i++){nf+=keys[i]<parents;nm+=keys[i]>=parents&&keys[i]<2147483647;}
  full_cnt[row]=nf*(block_size/128);mask_cnt[row]=nm*(block_size/128);}
 __syncthreads();
 int factor=block_size/128;int64_t offset=(int64_t)row*capacity;
 for(int i=tid;i<capacity;i+=blockDim.x){int ix=i/factor,child=i%factor;
  full_idx[offset+i]=ix<nf?keys[ix]*factor+child:-1;
  mask_idx[offset+i]=ix<nm?(keys[nf+ix]-parents)*factor+child:-1;}
}


// One complete warp owns one short route; sentinel lanes participate in every shuffle.
extern "C" __global__ void vsa_routes_warp(
 const int64_t *selected, const int *sizes, int *full_idx, int *full_cnt,
 int *mask_idx, int *mask_cnt, int rows, int topk, int prefix,
 int document_start, int block_size, int capacity, int parents) {
 int lane=threadIdx.x%32, row=blockIdx.x*4+threadIdx.x/32;
 if (row>=rows) return;
 int parent=lane<prefix+topk?vsa_route_parent(selected,row,topk,lane,prefix,document_start,parents):-1;
 int size=parent>=0?sizes[parent]:0;
 int key=size==block_size?parent:(size>0&&size<block_size?parent+parents:2147483647);
 #pragma unroll
 for (int k=2;k<=32;k*=2) {
  #pragma unroll
  for (int j=k/2;j>0;j/=2) {
   int other=__shfl_xor_sync(0xffffffff,key,j);
   bool ascending=(lane&k)==0, lower=(lane&j)==0;
   key=(ascending==lower)?min(key,other):max(key,other);
  }
 }
 int nf=__popc(__ballot_sync(0xffffffff,key<parents));
 int nm=__popc(__ballot_sync(0xffffffff,key>=parents&&key<2147483647));
 int factor=block_size/128;int64_t offset=(int64_t)row*capacity;
 for(int i=lane;i<capacity;i+=32){full_idx[offset+i]=-1;mask_idx[offset+i]=-1;}
 __syncwarp();
 if(lane==0){full_cnt[row]=nf*factor;mask_cnt[row]=nm*factor;}
 if(key<2147483647){
  bool full=key<parents;int rank=full?lane:lane-nf;
  int *out=full?full_idx:mask_idx;int p=full?key:key-parents;
  for(int child=0;child<factor;++child)out[offset+rank*factor+child]=p*factor+child;
 }
}
