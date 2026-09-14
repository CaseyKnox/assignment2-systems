import torch
from torch import nn
from torch.autograd import Function


class FlashAttention2(Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ):
        """
        Slow flash attention 2 forward pass in pure pytorch.
        Used for validating the correctness of the triton kernel.
        """
        batch = q.shape[0] # batch size
        seq = q.shape[1] # sequence length
        d_model = q.shape[2] # model dimension
        assert q.shape[0] == k.shape[0] == v.shape[0], "Batch size mismatch"
        bq = 16 # tile size of q
        bk = 16 # tile size of k
        tq = seq // bq # number of tiles in q (bq x d_model)
        # If the batch sizes are mismatched, we can use k.shape[0] instead of batch-size here
        tk = seq // bk # number of tiles in k (bk x d_model)
        sqrt_dk = q.shape[-1] ** 0.5 # sqrt(d_model)

        o = torch.zeros((batch, seq, d_model)) # output tensor; shape = (batch_size, seq, d_model)
        l = torch.zeros((batch, seq)) # logsumexp tensor; shape = (batch_size, seq)
        for i in range(tq):
            q_i = q[:, i * bq : (i + 1) * bq] # tile of q; shape = (batch, bq, d_model)
            o_i = torch.zeros_like(q_i) # output tile; shape = (batch, bq, d_model)
            l_i = torch.zeros((batch, bq))
            m_i = torch.ones((batch, bq)) * float("-inf")

            for j in range(tk):
                k_j = k[:, j * bk : (j + 1) * bk] # tile of k; shape = (batch, bk, d_model)
                s_i = torch.einsum("bqd,bkd->bqk", q_i, k_j) / sqrt_dk # shape = (batch, bq, bk)
                m_i_old = m_i.clone()
                m_i = torch.max(m_i, s_i.max(dim=-1).values) # shape = (batch, bq)
                p_i = torch.exp(s_i - m_i.unsqueeze(-1))  # shape = (batch, bq, bk)
                alpha = torch.exp(m_i_old - m_i) 
                l_i = alpha * l_i # shape = (batch, bq)
                l_i += p_i.sum(dim=-1) # shape = (batch, bq)
                v_i = v[:, j * bk : (j + 1) * bk] # tile of v; shape = (batch, bk, d_model)
                pv = torch.einsum("bqk,bkd->bqd", p_i, v_i) # shape = (batch, bq, d_model)
                o_i = torch.einsum("bq,bqd->bqd", alpha, o_i) + pv # shape = (batch, bq, d_model)

            o_i = torch.einsum("bq,bqd->bqd", l_i.reciprocal(), o_i) # shape = (batch, bq, d_model)
            l_i = m_i + torch.log(l_i) # shape = (batch, bq)
            o[:, i * bq : (i + 1) * bq] = o_i # shape = (batch, bq, d_model)
            l[:, i * bq : (i + 1) * bq] = l_i # shape = (batch, bq)

        ctx.save_for_backward(q, k, v, o, l)
        print(
            f"q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}, o.shape={o.shape}, l.shape={l.shape}"
        )
        return o

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        raise NotImplementedError("FlashAttention2 backward is not implemented yet.")


import triton
import triton.language as tl

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, # bq
    K_TILE_SIZE: tl.constexpr, # bk
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    seq_len = N_QUERIES

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0), # ?
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(seq_len,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,), # bq
        order=(0,),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0), 
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D), # assume N_KEYS == N_VALS ?
        strides=(stride_vk, stride_vd),
        offsets=(0, 0), 
        block_shape=(K_TILE_SIZE, D), # assume tile size shared with k
        order=(1, 0),
    )

    # Activations
    # init in sram as fp32
    m = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) 
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero") # (Q_TILE_SIZE, D)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero") # (K_TILE_SIZE, D)
        v = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero") # (K_TILE_SIZE, D)

        # Q @ K.t * scale
        s = tl.dot(q, tl.trans(k)) * scale # (bq, D) @ (bk, D).t -> (bq, bk)

        # Mask the padded key with -inf
        offs_k = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        mask = offs_k[None, :] < N_KEYS
        s = tl.where(mask, s, float('-inf'))

        # Online softmax
        m_block = tl.max(s, axis=1)    # (bq,)
        m_new = tl.maximum(m, m_block) # (bq,)
        p = tl.exp(s - m_new[:, None]) # (bq, bk)
        alpha = tl.exp(m - m_new) # (bq,)

        l = l * alpha # (bq,)
        l = l + tl.sum(p, axis=-1) # (bq,)

        # Cast p to V's dtype which may be bfloat16
        pv = tl.dot(p.to(v.dtype), v) # (bq, bk) @ (bk, D) -> (bq, D)
        o = o * alpha[:, None] + pv

        m = m_new

        # Advance the block pointers for the next iteration
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    o_final = o / l[:, None] # (bq, D)
    l = tl.log(l) + m
    tl.store(O_block_ptr, o_final, boundary_check=(0,1))
    tl.store(L_block_ptr, l, boundary_check=(0,))

class FlashAttnTriton(Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ):
        batch = q.shape[0] # batch size
        seq = q.shape[1] # sequence length
        d_model = q.shape[2] # model dimension
        assert q.shape[0] == k.shape[0] == v.shape[0], "Batch size mismatch"
        bq = 16 # tile size of q
        bk = 16 # tile size of k
        tq = seq // bq # number of tiles in q (bq x d_model)
        # If the batch sizes are mismatched, we can use k.shape[0] instead of batch-size here
        tk = seq // bk # number of tiles in k (bk x d_model)
        sqrt_dk = q.shape[-1] ** 0.5 # sqrt(d_model)

        o = torch.zeros((batch, seq, d_model), device='cuda') # output tensor; shape = (batch_size, seq, d_model)
        l = torch.zeros((batch, seq), device='cuda') # logsumexp tensor; shape = (batch_size, seq)

        flash_fwd_kernel[(tq, batch)](
            Q_ptr=q,
            K_ptr=k,
            V_ptr=v,
            O_ptr=o,
            L_ptr=l,
            stride_qb=q.stride(0),
            stride_qq=q.stride(1),
            stride_qd=q.stride(2),
            stride_kb=k.stride(0),
            stride_kk=k.stride(1),
            stride_kd=k.stride(2),
            stride_vb=v.stride(0),
            stride_vk=v.stride(1),
            stride_vd=v.stride(2),
            stride_ob=o.stride(0),
            stride_oq=o.stride(1),
            stride_od=o.stride(2),
            stride_lb=l.stride(0),
            stride_lq=l.stride(1),
            N_QUERIES=seq,
            N_KEYS=seq,
            scale=1/sqrt_dk,
            D=d_model,
            Q_TILE_SIZE=bq,
            K_TILE_SIZE=bk,
        )
        ctx.save_for_backward(q, k, v, o, l)
        return o