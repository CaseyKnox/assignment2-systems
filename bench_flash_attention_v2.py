"""
Benchmark FlashAttention-2 PyTorch vs Triton implementation.
Sweeps over seq_len, d_model, dtype.
"""
import torch
from triton.testing import do_bench
from cs336_systems.flash_attn import FlashAttention2, FlashAttnTriton

BATCH = 1
IS_CAUSAL = True
SEQ_LENS = [2**i for i in range(7, 17)]  # 128 .. 65536
EMBED_DIMS = [16, 32, 64, 128]
DTYPES = [torch.float32] # [torch.bfloat16, torch.float32]

def bench(impl_cls, q, k, v, is_causal):
    # warmup
    for _ in range(3):
        out = impl_cls.apply(q, k, v, is_causal)
        try:
            out.sum().backward()
        except Exception as ex:
            continue
        for t in (q, k, v):
            if t.grad is not None:
                t.grad.zero_()
    def fwd():
        return impl_cls.apply(q, k, v, is_causal)
    try:
        fwd_ms = do_bench(fwd, warmup=10, rep=30)
    except Exception as ex:
        print(f"Error during forward benchmark: {ex}")
        fwd_ms = -1
    def e2e():
        out = impl_cls.apply(q, k, v, is_causal)
        out.sum().backward()
    try:
        e2e_ms = do_bench(e2e, warmup=10, rep=30, grad_to_none=[q, k, v])
        # approximate backward only
        bwd_ms = max(e2e_ms - fwd_ms, 0.1)
    except Exception as ex:
        bwd_ms = -1
        e2e_ms = -1
    return fwd_ms, bwd_ms, e2e_ms

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    print(f"{'Seq':>8} {'D':>4} {'dtype':>10} {'impl':>10} {'fwd(ms)':>10} {'bwd(ms)':>10} {'e2e(ms)':>10}")
    print("-"*70)
    for seq in SEQ_LENS:
        for d in EMBED_DIMS:
            for dtype in DTYPES:
                torch.manual_seed(0)
                q = torch.randn(BATCH, seq, d, device='cuda', dtype=dtype, requires_grad=True)
                k = torch.randn(BATCH, seq, d, device='cuda', dtype=dtype, requires_grad=True)
                v = torch.randn(BATCH, seq, d, device='cuda', dtype=dtype, requires_grad=True)
                # PyTorch
                q_p, k_p, v_p = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
                try:
                    f, b, e = bench(FlashAttention2, q_p, k_p, v_p, IS_CAUSAL)
                    print(f"{seq:8d} {d:4d} {str(dtype).split('.')[-1]:>10} {'PyTorch':>10} {f:10.3f} {b:10.3f} {e:10.3f}")
                except Exception as ex:
                    print(f"{seq:8d} {d:4d} {str(dtype).split('.')[-1]:>10} {'PyTorch':>10} ERROR")
                # Triton
                q_t, k_t, v_t = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
                try:
                    f, b, e = bench(FlashAttnTriton, q_t, k_t, v_t, IS_CAUSAL)
                    print(f"{seq:8d} {d:4d} {str(dtype).split('.')[-1]:>10} {'Triton':>10} {f:10.3f} {b:10.3f} {e:10.3f}")
                except Exception as ex:
                    print(f"{seq:8d} {d:4d} {str(dtype).split('.')[-1]:>10} {'Triton':>10} ERROR")

if __name__ == "__main__":
    main()
