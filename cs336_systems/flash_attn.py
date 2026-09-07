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
