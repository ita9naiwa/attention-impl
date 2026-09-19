"""B300 long-sequence dispatch and compilation-cache regression."""
import pytest
import torch
from flash_attn.cute import interface
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100 as Kernel

@torch.no_grad()
def test_nonpersistent_dispatch_cache():
    if not torch.cuda.is_available() or interface._get_device_arch() != 103:
        pytest.skip("B300 dispatch regression")
    original = Kernel.__init__
    original_cache = interface._flash_attn_fwd.compile_cache
    observed = []
    def spy(self, *args, **kwargs):
        observed.append(kwargs['is_static_persistent'])
        original(self, *args, **kwargs)
    Kernel.__init__ = spy
    try:
        # Constant V gives exactly 0.5 regardless of quantization or tile traversal.
        inputs = {n:torch.zeros((1,n,1,128),device='cuda').to(torch.float8_e4m3fn) for n in (65535,65536)}
        for order in ((65535,65536,65535,65536),(65536,65535,65536,65535)):
            interface._flash_attn_fwd.compile_cache = {}
            observed.clear()
            cache_sizes = []
            for n in order:
                q = inputs[n]
                k = q
                v = torch.full(q.shape,0.5,device='cuda').to(torch.float8_e4m3fn)
                out = torch.full(q.shape,float('nan'),device='cuda',dtype=torch.bfloat16)
                interface._flash_attn_fwd(q,k,v,vc_expcast=True,out=out,num_splits=1)
                assert torch.isfinite(out).all() and torch.all(out == 0.5),n
                cache_sizes.append(len(interface._flash_attn_fwd.compile_cache))
            assert observed == [order[0]<65536,order[1]<65536],observed
            assert cache_sizes == [1,2,2,2],cache_sizes
        print('PASS threshold65535/65536, both cache orders, constant output, NaN poison',flush=True)
    finally:
        Kernel.__init__ = original
        interface._flash_attn_fwd.compile_cache = original_cache

if __name__=='__main__':
    test_nonpersistent_dispatch_cache()
