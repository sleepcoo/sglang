import random
import unittest

import torch

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd,
    decode_attention_fwd_grouped,
    decode_attention_fwd_normal,
)
from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd,
    redundant_attention,
)
from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)


class TestTritonAttention(unittest.TestCase):

    def _set_all_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setUp(self):
        # Set seeds before each test method
        self._set_all_seeds(42)
    

    def _quantize_tensor(self, tensor):
        """Quantize a tensor to uint8 with scale and zero point."""
        bs, head_num, head_dim = tensor.shape
        quantized = torch.empty(
            (bs, head_num, head_dim), dtype=torch.uint8, device=tensor.device
        )
        scale_zeros = torch.empty(
            (bs, head_num, 2), dtype=torch.float32, device=tensor.device
        )
        print(bs, head_num)
        for i in range(bs):
            for h in range(head_num):
                min_val = tensor[i, h].min().to(torch.float32)
                max_val = tensor[i, h].max().to(torch.float32)
                scale = (max_val - min_val) / 255.0
                zero_point = -min_val / scale
                quantized[i, h] = (
                    (tensor[i, h] / scale + zero_point + 0.5)
                    .clamp(0, 255)
                    .to(torch.uint8)
                )
                scale_zeros[i, h, 0] = scale
                scale_zeros[i, h, 1] = zero_point

        return quantized, scale_zeros

    def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV, D):
        dtype = torch.bfloat16

        b_seq_len_prefix = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device="cuda"
        )
        b_seq_len_extend = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device="cuda"
        )
        b_seq_len = b_seq_len_prefix + b_seq_len_extend
        max_len_in_batch = torch.max(b_seq_len, 0)[0].item()

        b_req_idx = torch.arange(B, dtype=torch.int32, device="cuda")
        req_to_tokens = torch.empty(
            (B, max_len_in_batch), dtype=torch.int32, device="cuda"
        )
        b_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
        b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device="cuda")
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
        for i in range(B):
            req_to_tokens[i, : b_seq_len[i]] = torch.arange(
                b_start_loc[i], b_start_loc[i] + b_seq_len[i]
            )

        total_token_num = torch.sum(b_seq_len).item()
        extend_token_num = torch.sum(b_seq_len_extend).item()
        k_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device="cuda"
        ).normal_(mean=0.1, std=0.2)
        v_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device="cuda"
        ).normal_(mean=0.1, std=0.2)

        k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="cuda")
        v_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="cuda")
        q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
        for i in range(B):
            extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
            extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
            extend_start = b_start_loc_extend[i]
            extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
            k_extend[extend_start:extend_end] = k_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            v_extend[extend_start:extend_end] = v_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            q_extend[extend_start:extend_end] = torch.empty(
                (b_seq_len_extend[i], H_Q, D), dtype=dtype, device="cuda"
            ).normal_(mean=0.1, std=0.2)

        o_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
        o_redundant = torch.empty(
            (extend_token_num, H_Q, D), dtype=dtype, device="cuda"
        )

        b_seq_len_extend = b_seq_len - b_seq_len_prefix
        b_start_loc_extend = torch.zeros_like(b_seq_len)
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
        max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
        k_buffer_quantized, k_scales_zeros = self._quantize_tensor(k_buffer)
        v_buffer_quantized, v_scales_zeros = self._quantize_tensor(v_buffer)
        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer_quantized,
            v_buffer_quantized,
            k_scales_zeros,
            v_scales_zeros,
            req_to_tokens,
            b_req_idx,
            b_seq_len,
            b_seq_len_extend,
            b_start_loc_extend,
            max_len_extend,
        )
        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_redundant,
            k_buffer,
            v_buffer,
            None,
            None,
            req_to_tokens,
            b_req_idx,
            b_seq_len,
            b_seq_len_extend,
            b_start_loc_extend,
            max_len_extend,
        )

        # redundant_attention(
        #     q_extend,
        #     o_redundant,
        #     k_buffer,
        #     v_buffer,
        #     b_req_idx,
        #     b_start_loc,
        #     b_seq_len,
        #     b_seq_len_prefix,
        #     max_len_in_batch,
        # )
        print(o_extend)
        print(o_redundant)
        is_close = torch.allclose(o_extend, o_redundant, rtol=2e-1, atol=1e-2)
        if not is_close:
            # 计算差异
            abs_diff = torch.abs(o_extend - o_redundant)
            rel_diff = abs_diff / (torch.abs(o_redundant) + 1e-8)
            max_abs_diff = torch.max(abs_diff).item()  # 转换为Python标量
            max_rel_diff = torch.max(rel_diff).item()
            mean_abs_diff = torch.mean(abs_diff).item()
            mean_rel_diff = torch.mean(rel_diff).item()
            
            # 找出最大差异的位置
            max_rel_idx = torch.where(rel_diff == max_rel_diff)

            print("\n差异分析:")
            print(f"最大绝对差异: {max_abs_diff:.6f}")
            print(f"最大相对差异: {max_rel_diff:.6f}")
            print(f"平均绝对差异: {mean_abs_diff:.6f}")
            print(f"平均相对差异: {mean_rel_diff:.6f}")
            print("最大相对差异位置:", max_rel_idx)
            print("最大相对差异处的值:")
            print("o_extend:", o_extend[max_rel_idx].flatten())
            print("o_redundant:", o_redundant[max_rel_idx].flatten())
            print("实际差值:", abs_diff[max_rel_idx].flatten())
            
            # 打印形状和非零元素比例
            print("\n张量信息:")
            print("形状:", o_extend.shape)
            print(f"o_extend非零元素比例: {(o_extend != 0).float().mean().item():.4f}")
            print(f"o_redundant非零元素比例: {(o_redundant != 0).float().mean().item():.4f}")
            
            self.assertTrue(False, "输出不匹配，见上方详细信息")
    def test_extend_attention(self):

        # Define the varying parameter values
        attention_values = [128, 96, 80, 13]

        # Loop through the values and call the method
        for value in attention_values:
            self._test_extend_attention_once(19, 24, 12, 4, value)

    # def _test_context_attention_once(self, head_dim, is_causal):
    #     # Set up a simple test case
    #     num_heads = 4
    #     seq_lens = [8, 12]
    #     max_seq_len = max(seq_lens)

    #     # Create random input tensors
    #     q = torch.randn(sum(seq_lens), num_heads, head_dim, device="cuda")
    #     k = torch.randn(sum(seq_lens), num_heads, head_dim, device="cuda")
    #     v = torch.randn(sum(seq_lens), num_heads, head_dim, device="cuda")
    #     o = torch.zeros(sum(seq_lens), num_heads, head_dim, device="cuda")

    #     # Create b_start_loc and b_seq_len tensors
    #     b_start_loc = torch.tensor([0, seq_lens[0]], device="cuda")
    #     b_seq_len = torch.tensor(seq_lens, device="cuda")

    #     context_attention_fwd(
    #         q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=is_causal
    #     )

    #     cu_seq_lens = [0] * (len(seq_lens) + 1)
    #     for i, seq_len in enumerate(seq_lens):
    #         cu_seq_lens[i + 1] = cu_seq_lens[i] + seq_len

    #     for i in range(len(seq_lens)):
    #         start, end = cu_seq_lens[i], cu_seq_lens[i + 1]
    #         o_torch = torch.nn.functional.scaled_dot_product_attention(
    #             q[start:end].permute(1, 0, 2),
    #             k[start:end].permute(1, 0, 2),
    #             v[start:end].permute(1, 0, 2),
    #             is_causal=is_causal,
    #         ).permute(1, 0, 2)

    #         cos_sim = torch.nn.functional.cosine_similarity(
    #             o[start:end].flatten(), o_torch.flatten(), dim=0
    #         )
    #         self.assertTrue(cos_sim.item() > 1 - (1e-5))
    #         self.assertTrue(torch.allclose(o[start:end], o_torch, atol=1e-2))

    # def test_context_attention(self):
    #     head_dim = [128, 96, 80, 13]

    #     for dim in head_dim:
    #         for is_causal in [True, False]:
    #             self._test_context_attention_once(dim, is_causal)

    # def _test_decode_attention_once(self, B, H_Q, H_KV, D):
    #     dtype = torch.bfloat16
    #     seq_len = 10  # This represents the number of tokens already in the sequence
    #     total_tokens = B * seq_len
    #     sm_scale = 1.0 / (D**0.5)
    #     num_kv_splits = 8

    #     # q represents the new token being generated, one per batch
    #     q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

    #     # k_buffer and v_buffer represent all previous tokens
    #     k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
    #     v_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")

    #     # o will have the same shape as q
    #     o = torch.zeros(B, H_Q, D, dtype=dtype, device="cuda")

    #     req_to_token = torch.arange(total_tokens, device="cuda").reshape(B, seq_len)
    #     b_req_idx = torch.arange(B, device="cuda")
    #     b_seq_len = torch.full((B,), seq_len, device="cuda")

    #     attn_logits = torch.empty(
    #         (B, H_Q, num_kv_splits, D + 1),
    #         dtype=torch.float32,
    #         device="cuda",
    #     )
    #     k_scales_zeros = None
    #     v_scales_zeros = None
    #     decode_attention_fwd(
    #         q,
    #         k_buffer,
    #         v_buffer,
    #         k_scales_zeros,
    #         v_scales_zeros,
    #         o,
    #         req_to_token,
    #         b_req_idx,
    #         b_seq_len,
    #         attn_logits,
    #         num_kv_splits,
    #         sm_scale,
    #     )

    # def test_decode_attention(self):
    #     # Here we just to ensure there is no error
    #     # TODO: correctnesss test

    #     # Test configurations
    #     configs = [
    #         (2, 4, 4, 64),  # MHA
    #         (2, 4, 2, 64),  # GQA
    #         (2, 4, 4, 80),  # Non-standard head dim
    #         (2, 4, 4, 13),  # Prime number head dim
    #     ]

    #     for B, H_Q, H_KV, D in configs:
    #         self._test_decode_attention_once(B, H_Q, H_KV, D)

    # def _test_grouped_decode_attention_once(self, B, S, H_Q, H_KV, D, D_V):
    #     dtype = torch.bfloat16
    #     seq_len = S  # This represents the number of tokens already in the sequence
    #     total_tokens = B * seq_len
    #     sm_scale = 1.0 / (D**0.5)
    #     num_kv_splits = 8

    #     # q represents the new token being generated, one per batch
    #     q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

    #     # k_buffer and v_buffer represent all previous tokens
    #     k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
    #     v_buffer = torch.randn(total_tokens, H_KV, D_V, dtype=dtype, device="cuda")

    #     # o will have the same shape as q
    #     o = torch.zeros(B, H_Q, D_V, dtype=dtype, device="cuda")
    #     o_grouped = torch.zeros(B, H_Q, D_V, dtype=dtype, device="cuda")

    #     req_to_token = torch.arange(total_tokens, device="cuda").reshape(B, seq_len)
    #     b_req_idx = torch.arange(B, device="cuda")
    #     b_seq_len = torch.full((B,), seq_len, device="cuda")

    #     attn_logits = torch.empty(
    #         (B, H_Q, num_kv_splits, D_V + 1),
    #         dtype=torch.float32,
    #         device="cuda",
    #     )
    #     k_scales_zeros = None
    #     v_scales_zeros = None
    #     decode_attention_fwd_normal(
    #         q,
    #         k_buffer,
    #         v_buffer,
    #         k_scales_zeros,
    #         v_scales_zeros,
    #         o,
    #         req_to_token,
    #         b_req_idx,
    #         b_seq_len,
    #         attn_logits,
    #         num_kv_splits,
    #         sm_scale,
    #     )

    #     attn_logits1 = torch.empty(
    #         (B, H_Q, num_kv_splits, D_V + 1),
    #         dtype=torch.float32,
    #         device="cuda",
    #     )

    #     decode_attention_fwd_grouped(
    #         q,
    #         k_buffer,
    #         v_buffer,
    #         k_scales_zeros,
    #         v_scales_zeros,
    #         o_grouped,
    #         req_to_token,
    #         b_req_idx,
    #         b_seq_len,
    #         attn_logits1,
    #         num_kv_splits,
    #         sm_scale,
    #     )

    #     cos_sim = torch.nn.functional.cosine_similarity(
    #         o.flatten(), o_grouped.flatten(), dim=0
    #     )
    #     print(cos_sim.item())
    #     self.assertTrue(cos_sim.item() > 0.99)
    #     self.assertTrue(torch.allclose(o, o_grouped, atol=3e-2))

    # def test_grouped_decode_attention(self):
    #     seq_lens = [5, 100, 128, 500]
    #     configs = [
    #         (2, 16, 16, 64, 64),
    #         (2, 16, 1, 64, 64),
    #         (2, 64, 1, 13, 13),
    #         (2, 128, 1, 80, 80),
    #         (2, 128, 2, 512, 512),
    #         (2, 128, 1, 576, 512),
    #     ]

    #     for S in seq_lens:
    #         for B, H_Q, H_KV, D, D_V in configs:
    #             self._test_grouped_decode_attention_once(B, S, H_Q, H_KV, D, D_V)


if __name__ == "__main__":
    unittest.main()
