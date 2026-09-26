#include "vc_preprocess.cu"

__device__ int64_t vsa_index(const void *p, int i, bool wide) {
    return wide ? ((const int64_t*)p)[i] : ((const int*)p)[i];
}

// DELAYED (caller-owned scale state): also store the padded E4M3 Q/K/V with the state's scales x margins,
// using vsa_quantize's per-element math and indexing; statistics and pools use the identical code path.
template<int D, bool DELAYED=false> __device__ void vsa_stats_impl(
    const void *q, const void *k, const void *v, const void *source_map,
    const void *query_map, const void *sizes, float *stats,
    float *pq, float *pk, float *pv, int source_n, int nblocks, int h,
    int block_size, int query_n, int query_offset, int dtype, int metadata_mask, float scale,
    const float *skmean=nullptr, const float *sqs=nullptr, const float *sks=nullptr, const float *svs=nullptr,
    uint8_t *oq=nullptr, uint8_t *ok=nullptr, uint8_t *ov=nullptr, int *saturated=nullptr,
    float qk_margin=1.f, float v_margin=1.f) {
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
    for (int row=warp; row<block_size; row+=8) {
        int padded = block*block_size+row;
        int64_t original = vsa_index(source_map,padded,metadata_mask&1), qi = vsa_index(query_map,padded,metadata_mask&2)-query_offset;
        bool valid = original >= 0 && original < source_n;
        bool query_valid = qi >= 0 && qi < query_n;
        int64_t base = (((int64_t)batch*source_n+original)*h+head)*D;
        float qr[W], kr[W], vr[W];
        if constexpr(D==128){
            float rawq[W], rawk[W], rawv[W];
            if(valid){
                read_contiguous<D>(q,base+lane*W,dtype,rawq);
                read_contiguous<D>(k,base+lane*W,dtype,rawk);
                read_contiguous<D>(v,base+lane*W,dtype,rawv);
            }else{
                #pragma unroll
                for(int j=0;j<W;++j){rawq[j]=0;rawk[j]=0;rawv[j]=0;}
            }
            #pragma unroll
            for (int j=0; j<W; ++j) {
                float qv=rawq[j],kv=rawk[j],vv=rawv[j];
                a[5][j] += qv; a[6][j] += kv; a[7][j] += vv;
                a[4][j] = fmaxf(a[4][j], fabsf(vv));
                qr[j] = query_valid ? qv : 0; kr[j] = kv;
                if constexpr(DELAYED) vr[j] = vv;
            }
        }else{
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
                if constexpr(DELAYED) vr[j] = vv;
            }
        }
        rotate_contiguous<D>(qr,kr,scale);
        #pragma unroll
        for (int j=0; j<W; ++j) {
            a[0][j] += kr[j];
            a[1][j] = fminf(a[1][j],kr[j]); a[2][j] = fmaxf(a[2][j],kr[j]);
            a[3][j] = fmaxf(a[3][j],fabsf(qr[j]));
        }
        if constexpr(DELAYED) {
            int c=lane*W;
            float uq=sqs[bh]*qk_margin, uk=sks[bh]*qk_margin;
            #pragma unroll
            for (int j=0; j<W; ++j) {
                qr[j]/=uq; kr[j]=(kr[j]-skmean[bh*D+c+j])/uk; vr[j]/=svs[bh*D+c+j]*v_margin;
            }
            int64_t dst=(((int64_t)batch*nblocks*block_size+padded)*h+head)*D+c;
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
        if constexpr(DELAYED) {  // one compare per (block, channel, tensor) from the block maxima
            float km=skmean[bh*D+c];
            int over=(sums[3]>448.f*sqs[bh]*qk_margin)+(sums[4]>448.f*svs[bh*D+c]*v_margin)
                    +(fmaxf(fabsf(sums[1]-km),fabsf(sums[2]-km))>448.f*sks[bh]*qk_margin);
            if(over) atomicAdd(saturated,over);
        }
    }
}

extern "C" __global__ void vsa_stats(
    const void *q,const void *k,const void *v,const void *source_map,
    const void *query_map,const void *sizes,float *stats,float *pq,float *pk,float *pv,
    int source_n,int nblocks,int h,int d,int block_size,int query_n,int query_offset,int dtype,int metadata_mask) {
    float scale=rsqrtf((float)d);
    if(d==64) vsa_stats_impl<64>(q,k,v,source_map,query_map,sizes,stats,pq,pk,pv,source_n,nblocks,h,block_size,query_n,query_offset,dtype,metadata_mask,scale);
    else vsa_stats_impl<128>(q,k,v,source_map,query_map,sizes,stats,pq,pk,pv,source_n,nblocks,h,block_size,query_n,query_offset,dtype,metadata_mask,scale);
}

// Warm path: vsa_stats plus E4M3 stores with the state's scales x margins (one read of each source row).
extern "C" __global__ __launch_bounds__(256, 3) void vsa_stats_delayed(
    const void *q,const void *k,const void *v,const void *source_map,
    const void *query_map,const void *sizes,float *stats,float *pq,float *pk,float *pv,
    const float *skmean,const float *sqs,const float *sks,const float *svs,uint8_t *oq,uint8_t *ok,uint8_t *ov,
    int *saturated,float qk_margin,float v_margin,
    int source_n,int nblocks,int h,int d,int block_size,int query_n,int query_offset,int dtype,int metadata_mask) {
    float scale=rsqrtf((float)d);
    if(d==64) vsa_stats_impl<64,true>(q,k,v,source_map,query_map,sizes,stats,pq,pk,pv,source_n,nblocks,h,block_size,query_n,query_offset,dtype,metadata_mask,scale,skmean,sqs,sks,svs,oq,ok,ov,saturated,qk_margin,v_margin);
    else vsa_stats_impl<128,true>(q,k,v,source_map,query_map,sizes,stats,pq,pk,pv,source_n,nblocks,h,block_size,query_n,query_offset,dtype,metadata_mask,scale,skmean,sqs,sks,svs,oq,ok,ov,saturated,qk_margin,v_margin);
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
    rotate_contiguous<D>(qr,kr,scale);
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
    int token=blockIdx.x*4+threadIdx.x/32;
    if(token>=padded_n) return;
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

// Fallback: vsa_quantize's per-token math with fresh scales for flagged (b,h) only; 128 tokens per CTA,
// unflagged heads exit immediately.
extern "C" __global__ void vsa_requantize(
    const void *q,const void *k,const void *v,const void *source_map,const void *query_map,
    const float *kmean,const float *qs,const float *ks,const float *vs,uint8_t *oq,uint8_t *ok,uint8_t *ov,
    const int *flag,int source_n,int padded_n,int query_n,int query_offset,int h,int d,int dtype,int metadata_mask) {
    if(!flag[blockIdx.z*h+blockIdx.y]) return;
    float scale=rsqrtf((float)d);
    int end=min(padded_n,(int)blockIdx.x*128+128);
    for(int token=blockIdx.x*128+threadIdx.x/32; token<end; token+=4) {
        if(d==64) vsa_quantize_token<64>(q,k,v,source_map,query_map,kmean,qs,ks,vs,oq,ok,ov,source_n,padded_n,query_n,query_offset,h,dtype,metadata_mask,scale,token);
        else vsa_quantize_token<128>(q,k,v,source_map,query_map,kmean,qs,ks,vs,oq,ok,ov,source_n,padded_n,query_n,query_offset,h,dtype,metadata_mask,scale,token);
    }
}

// Per (b,h): the used scales (state x margin) must cover this call's fresh ranges (K bound via
// |k-kmean_state| <= |k-kmean_fresh| + |kmean_fresh-kmean_state|) and not exceed 2x the steady-state ratio
// (used <= 2*margin*fresh); otherwise flag the head for fresh requantization. Writes the descales actually
// used, then copies the fresh statistics into the state in place. One CTA of d threads per (b,h).
extern "C" __global__ void vsa_check(float *skmean,float *sqs,float *sks,float *svs,const float *fkmean,
    const float *fqs,const float *fks,const float *fvs,int *flag,float *oqs,float *oks,float *ovs,int *fallbacks,
    float qk_margin,float v_margin,int d) {
    int bh=blockIdx.x,c=threadIdx.x; __shared__ int bad; __shared__ float shift[128];
    if(c==0) bad=0;
    __syncthreads();
    float uv=svs[bh*d+c]*v_margin,fv=fvs[bh*d+c];
    shift[c]=fabsf(fkmean[bh*d+c]-skmean[bh*d+c]);
    if(fv>uv||uv>2.f*v_margin*fv) atomicOr(&bad,1);
    __syncthreads();
    if(c==0){
        float ms=0; for(int j=0;j<d;++j) ms=fmaxf(ms,shift[j]);
        float uq=sqs[bh]*qk_margin,uk=sks[bh]*qk_margin,kneed=fks[bh]+ms/448.f;
        if(fqs[bh]>uq||uq>2.f*qk_margin*fqs[bh]||kneed>uk||uk>2.f*qk_margin*fks[bh]) bad=1;
        flag[bh]=bad; if(bad) atomicAdd(fallbacks,1);
        oqs[bh]=bad?fqs[bh]:uq; oks[bh]=bad?fks[bh]:uk;
    }
    __syncthreads();
    ovs[bh*d+c]=bad?fv:uv;
    svs[bh*d+c]=fv; skmean[bh*d+c]=fkmean[bh*d+c];
    if(c==0){sqs[bh]=fqs[bh]; sks[bh]=fks[bh];}
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
