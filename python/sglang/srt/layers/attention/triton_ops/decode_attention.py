# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

import logging

import triton
import triton.language as tl

from sglang.srt.utils import is_hip

is_hip_ = is_hip()

logger = logging.getLogger(__name__)

# TODO: Remove this when triton>=3.2.0. This issue will not affect performance and accuracy.
logger.warning(
    "The following error message 'operation scheduled before its operands' can be ignored."
)


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    K_Scale_Zeros_Buffer,
    V_Scale_Zeros_Buffer,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_sz_kbs,
    stride_sz_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_sz_vbs,
    stride_sz_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    USE_INT8_KV: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    scale_dtype = K_Scale_Zeros_Buffer.dtype.element_ty

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q, mask=mask_d, other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            if not USE_INT8_KV:
                k = tl.load(
                    K_Buffer + offs_buf_k,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                    other=0.0,
                )
            else:
                # load quantized k
                k_int8 = tl.load(
                    K_Buffer + offs_buf_k,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                    other=0.0,
                )
                # load k scale
                offs_scale_k = (
                    kv_loc[:, None] * stride_sz_kbs * 2
                    + cur_kv_head * stride_buf_kh * 2
                )
                k_scales = tl.load(
                    K_Scale_Zeros_Buffer + offs_scale_k,
                    mask=offs_n[:, None] < split_kv_end,
                    other=1.0
                )
                offs_zeros_k = (
                    kv_loc[:, None] * stride_sz_kbs * 2
                    + cur_kv_head * stride_buf_kh * 2
                    + 1
                )
                k_zeros = tl.load(
                    K_Scale_Zeros_Buffer + offs_zeros_k, 
                    mask=offs_n[:, None] < split_kv_end,
                    other=0
                )
                k = (k_int8 - k_zeros).to(scale_dtype) * k_scales
                
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            if not USE_INT8_KV:
                v = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                    other=0.0,
                )
            else:
                v_int8 = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                    other=0.0,
                )
                # load v scale
                offs_scale_v = (
                    kv_loc[:, None] * stride_sz_vbs * 2
                    + cur_kv_head * stride_buf_vh * 2
                )
                v_scales = tl.load(
                    V_Scale_Zeros_Buffer + offs_scale_v,
                    mask=offs_n[:, None] < split_kv_end,
                    other=1.0
                )
                offs_zeros_v = (
                    kv_loc[:, None] * stride_sz_vbs * 2
                    + cur_kv_head * stride_buf_vh * 2
                    + 1
                )
                v_zeros = tl.load(
                    V_Scale_Zeros_Buffer + offs_zeros_v,
                    mask=offs_n[:, None] < split_kv_end,
                    other=0
                )
                v = (v_int8 - v_zeros).to(scale_dtype) * v_scales

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    k_scale_zeros_buffer,
    v_scale_zeros_buffer,
    att_out,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    logit_cap,
):
    BLOCK = 64
    NUM_KV_SPLITS = num_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]
    
    # assert kv dtype
    USE_INT8_KV = k_buffer[0].dtype == torch.int8

    batch, head_num = B_req_idx.shape[0], q.shape[1]

    grid = (batch, head_num, NUM_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    if USE_INT8_KV:
        k_scale_zeros_stride0 = k_scale_zeros_buffer.stride(0)
        k_scale_zeros_stride1 = k_scale_zeros_buffer.stride(1)
        v_scale_zeros_stride0 = v_scale_zeros_buffer.stride(0)
        v_scale_zeros_stride1 = v_scale_zeros_buffer.stride(1)
    else:
        k_scale_zeros_stride0 = None
        k_scale_zeros_stride1 = None
        v_scale_zeros_stride0 = None
        v_scale_zeros_stride1 = None

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        k_scale_zeros_buffer,
        v_scale_zeros_buffer,
        sm_scale,
        Req_to_tokens,
        B_req_idx,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_scale_zeros_stride0,
        k_scale_zeros_stride1,
        v_buffer.stride(0),
        v_buffer.stride(1),
        v_scale_zeros_stride0,
        v_scale_zeros_stride1,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
        USE_INT8_KV=USE_INT8_KV
    )


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    K_Scale_Zeros_Buffer,
    V_Scale_Zeros_Buffer,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_sz_kbs,
    stride_sz_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_sz_vbs,
    stride_sz_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    USE_INT8_KV: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)
    scales_dtype = K_Scale_Zeros_Buffer.dtype.element_ty

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )
        qpe = tl.load(
            Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
        )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_k = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            if not USE_INT8_KV:
                k = tl.load(
                    K_Buffer + offs_buf_k,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                    other=0.0,
                )
            else:
                k_int8 = tl.load(
                    K_Buffer + offs_buf_k,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                    other=0.0,
                )
                offs_scale_k = (
                    kv_loc[None, :] * stride_sz_kbs * 2
                    + cur_kv_head * stride_sz_kh * 2
                )
                k_scale = tl.load(
                    K_Scale_Zeros_Buffer + offs_scale_k,
                    mask=offs_n[None, :] < split_kv_end,
                    other=1.0,
                )
                offs_zeros_k = (
                    kv_loc[None, :] * stride_sz_kbs * 2
                    + cur_kv_head * stride_sz_kh * 2
                    + 1
                )
                k_zeros = tl.load(
                    K_Scale_Zeros_Buffer + offs_zeros_k, 
                    mask=offs_n[None, :] < split_kv_end,
                    other=0
                )
                k = (k_int8 - k_zeros).to(scales_dtype) * k_scale
                
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = (
                    kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                if not USE_INT8_KV:
                    kpe = tl.load(
                        K_Buffer + offs_buf_kpe,
                        mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                        other=0.0,
                    )
                else:
                    kpe_int8 = tl.load(
                        K_Buffer + offs_buf_kpe,
                        mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                        other=0.0,
                    )
                    offs_scale_kpe = (
                        kv_loc[None, :] * stride_sz_kbs * 2
                        + cur_kv_head * stride_sz_kh * 2
                    )
                    kpe_scale = tl.load(
                        K_Scale_Zeros_Buffer + offs_scale_kpe,
                        mask=offs_n[None, :] < split_kv_end,
                        other=1.0,
                    )
                    offs_zeros_kpe = (
                        kv_loc[None, :] * stride_sz_kbs * 2
                        + cur_kv_head * stride_sz_kh * 2
                        + 1
                    )
                    kpe_zeros = tl.load(
                        K_Scale_Zeros_Buffer + offs_zeros_kpe, 
                        mask=offs_n[None, :] < split_kv_end,
                        other=0
                    )
                    kpe = (kpe_int8 - kpe_zeros).to(scales_dtype) * kpe_scale
                    
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            if not USE_INT8_KV:
                v = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                    other=0.0,
                )
            else:
                v_int8 = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                    other=0.0,
                )
                offs_scale_v = (
                    kv_loc[None, :] * stride_sz_vbs * 2
                    + cur_kv_head * stride_sz_vh * 2
                )
                v_scale = tl.load(
                    V_Scale_Zeros_Buffer + offs_scale_v,
                    mask=offs_n[None, :] < split_kv_end,
                    other=1.0,
                )
                offs_zeros_v = (
                    kv_loc[None, :] * stride_sz_vbs * 2
                    + cur_kv_head * stride_sz_vh * 2
                    + 1
                )
                v_zeros = tl.load(
                    V_Scale_Zeros_Buffer + offs_zeros_v, 
                     mask=offs_n[None, :] < split_kv_end,
                    other=0
                )
                v = (v_int8 - v_zeros).to(scales_dtype) * v_scale

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    k_scale_zeros_buffer,
    v_scale_zeros_buffer,
    att_out,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    logit_cap,
):
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]
    
    # assert kv dtype
    USE_INT8_KV = k_buffer[0].dtype == torch.int8

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = B_req_idx.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK_H = 16
    NUM_KV_SPLITS = num_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    extra_kargs = {}
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    if USE_INT8_KV:
        k_scale_zeros_stride0 = k_scale_zeros_buffer.stride(0)
        k_scale_zeros_stride1 = k_scale_zeros_buffer.stride(1)
        v_scale_zeros_stride0 = v_scale_zeros_buffer.stride(0)
        v_scale_zeros_stride1 = v_scale_zeros_buffer.stride(1)
    else:
        k_scale_zeros_stride0 = None
        k_scale_zeros_stride1 = None
        v_scale_zeros_stride0 = None
        v_scale_zeros_stride1 = None

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        k_scale_zeros_buffer,
        v_scale_zeros_buffer,
        sm_scale,
        Req_to_tokens,
        B_req_idx,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_scale_zeros_stride0,
        k_scale_zeros_stride1,
        v_buffer.stride(0),
        v_buffer.stride(1),
        v_scale_zeros_stride0,
        v_scale_zeros_stride1,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
        USE_INT8_KV=USE_INT8_KV,
        **extra_kargs,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    O,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(
    logits,
    q,
    o,
    v_buffer,
    b_seq_len,
    num_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_kv_splits

    extra_kargs = {}
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        o,
        b_seq_len,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_normal(
    q,
    k_buffer,
    v_buffer,
    k_scale_zeros_buffer,
    v_scale_zeros_buffer,
    o,
    req_to_token,
    b_req_idx,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    logit_cap=0.0,
):
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        k_scale_zeros_buffer,
        v_scale_zeros_buffer,
        attn_logits,
        req_to_token,
        b_req_idx,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(attn_logits, q, o, v_buffer, b_seq_len, num_kv_splits)


def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    k_scale_zeros_buffer,
    v_scale_zeros_buffer,
    req_to_token,
    b_req_idx,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    logit_cap=0.0,
):
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        k_scale_zeros_buffer,
        v_scale_zeros_buffer,
        attn_logits,
        req_to_token,
        b_req_idx,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(attn_logits, q, o, v_buffer, b_seq_len, num_kv_splits)


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    k_scale_zeros_buffer,
    v_scale_zeros_buffer,
    req_to_token,
    b_req_idx,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    logit_cap=0.0,
):
    assert num_kv_splits == attn_logits.shape[2]
    kv_group_num = q.shape[1] // v_buffer.shape[1]

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            k_scale_zeros_buffer,
            v_scale_zeros_buffer,
            o,
            req_to_token,
            b_req_idx,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            logit_cap,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            k_scale_zeros_buffer,
            v_scale_zeros_buffer,
            o,
            req_to_token,
            b_req_idx,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            logit_cap,
        )


@triton.jit
def _quant_int8(val):
    val_min = tl.min(val, 1)
    val_max = tl.max(val, 1)
    scales = (val_max - val_min) / 255
    zeros = -val_min / scales
    q_val = (val / scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
    return q_val, scales, zeros


@triton.jit
def _fwd_kernel_quantize_cache_kv(
    K_Status,
    V_Status,
    Dest_Idx,
    K_Cache,
    V_Cache,
    K_Scale_Zeros,
    V_Scale_Zeros,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_v_bs,
    stride_v_h,
    stride_v_d,
    stride_k_c_bs,
    stride_k_c_h,
    stride_k_c_d,
    stride_v_c_bs,
    stride_v_c_h,
    stride_v_c_d,
    stride_k_sz_bs,
    stride_k_sz_h,
    stride_k_sz_d,
    stride_v_sz_bs,
    stride_v_sz_h,
    stride_v_sz_d,
    head_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    dest_index = tl.load(Dest_Idx + cur_index)
    # k
    k_data = tl.load(
        K_Status
        + cur_index * stride_k_bs
        + offs_h[:, None] * stride_k_h
        + offs_d[None, :] * stride_k_d,
        mask=offs_h[:, None] < head_num,
        other=0.0,
    )
    
    k_quant, k_scales, k_zeros = _quant_int8(k_data)
    
    szd_off = tl.arange(0, 2)
    
    o_ptrs = (
        K_Cache
        + dest_index * stride_k_c_bs
        + offs_h[:, None] * stride_k_c_h
        + offs_d[None, :] * stride_k_c_d
    )
    sz_ptrs_k = K_Scale_Zeros + dest_index * stride_k_sz_bs * 2 + stride_k_sz_h * offs_h[:, None] * 2
    
    tl.store(o_ptrs, k_quant, mask=offs_h[:, None] < head_num)
    
    tl.store(
        sz_ptrs_k + szd_off[None, :] * 1, 
        k_scales, 
        mask=(offs_h[:, None] < head_num) & (szd_off[None, :] < 1)
    )
    tl.store(
        sz_ptrs_k + szd_off[None, :] * 1,
        k_zeros[:, None],
        mask=(offs_h[:, None] < head_num) & (szd_off[None, :] == 1)
    )
    
    # v
    v_data = tl.load(
        V_Status
        + cur_index * stride_v_bs
        + offs_h[:, None] * stride_v_h
        + offs_d[None, :] * stride_v_d,
        mask=offs_h[:, None] < head_num,
        other=0.0,
    )
    
    v_quant, v_scales, v_zeros = _quant_int8(v_data)
    
    o_ptrs = (
        V_Cache
        + dest_index * stride_v_c_bs
        + offs_h[:, None] * stride_v_c_h
        + offs_d[None, :] * stride_v_c_d
    )
    sz_ptrs_v = V_Scale_Zeros + dest_index * stride_v_sz_bs * 2 + stride_v_sz_h * offs_h[:, None] * 2
    
    tl.store(o_ptrs, v_quant, mask=offs_h[:, None] < head_num)
    tl.store(
        sz_ptrs_v + szd_off[None, :] * 1, 
        v_scales, 
        mask=(offs_h[:, None] < head_num) & (szd_off[None, :] < 1)
    )
    tl.store(
        sz_ptrs_v + szd_off[None, :] * 1,
        v_zeros[:, None],
        mask=(offs_h[:, None] < head_num) & (szd_off[None, :] == 1)
    )


@torch.no_grad()
def quantize_cache_kv(
    k_status, 
    v_status, 
    dest_idx, 
    k_quantized_out, 
    k_scale_zeros,
    v_quantized_out,
    v_scale_zeros,
):
    bs = dest_idx.shape[0]
    k_head_num = k_status.shape[1]
    k_head_dim = k_status.shape[2]
    assert k_status.shape[1] == k_quantized_out.shape[1] and k_status.shape[2] == k_quantized_out.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(k_head_num)
    grid = (bs,)
    num_warps = 1

    _fwd_kernel_quantize_cache_kv[grid](
        k_status,
        v_status,
        dest_idx,
        k_quantized_out,
        v_quantized_out,
        k_scale_zeros,
        v_scale_zeros,
        k_status.stride(0),
        k_status.stride(1),
        k_status.stride(2),
        v_status.stride(0),
        v_status.stride(1),
        v_status.stride(2),
        k_quantized_out.stride(0),
        k_quantized_out.stride(1),
        k_quantized_out.stride(2),
        v_quantized_out.stride(0),
        v_quantized_out.stride(1),
        v_quantized_out.stride(2),
        k_scale_zeros.stride(0),
        k_scale_zeros.stride(1),
        k_scale_zeros.stride(2),
        v_scale_zeros.stride(0),
        v_scale_zeros.stride(1),
        v_scale_zeros.stride(2),
        k_head_num,
        BLOCK_DMODEL=k_head_dim,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
    return