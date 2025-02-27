# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py
import argparse
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, TypedDict, Optional, Callable
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import torch
import triton
from ray.experimental.tqdm_ray import tqdm
from transformers import AutoConfig

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import get_config_dtype_str
from sglang.srt.utils import is_hip, direct_register_custom_op, get_device_name

_is_hip_ = is_hip()

from sglang.srt.layers.moe.ep_moe.kernels import (
    grouped_gemm_triton,
    post_reorder_triton_kernel,
    pre_reorder_triton_kernel,
    run_moe_ep_preproess,
    silu_and_mul_triton_kernel,
)
from sglang.srt.layers.moe.topk import select_experts

def get_available_gpu_count():
    """Get the number of available GPUs."""
    return torch.cuda.device_count()

def get_config_file_name(
    E: int, N: int, ep_size: int, dtype: Optional[str], block_shape: Optional[int] = None
) -> str:
    device_name = get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    )
    ep_str = "" if ep_size == 0 else f"EP={ep_size}"
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}{ep_str}.json"

def ep_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    # from ep_moe
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    use_blockwise_fp8: bool = False,
    # ep config
    num_experts: int = 256,
    # fp8_dtype: torch.types = torch.float8_e4m3fn,
    num_experts_per_partition: int = 128,
    start_expert_id: int = 0,
    end_expert_id: int = 127,
    use_fp8_w8a8: bool = False,
    w1_scale_inv: Optional[torch.Tensor] = None,
    w2_scale_inv: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
) -> torch.Tensor:
    fp8_dtype = torch.float8_e4m3fn

    reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(topk_ids, num_experts)

    gateup_input = torch.empty(
        (int(hidden_states.shape[0] * top_k), hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=(
            fp8_dtype
            if (use_fp8_w8a8 and not use_blockwise_fp8)
            else hidden_states.dtype
        ),
    )

    if use_fp8_w8a8 and not use_blockwise_fp8:
        max_value = (
            torch.max(hidden_states).repeat(num_experts_per_partition).to(torch.float32)
        )
        w1_input_scale = max_value / torch.finfo(fp8_dtype).max
    else:
        w1_input_scale = None

    # PreReorder
    pre_reorder_triton_kernel[(hidden_states.shape[0],)](
        hidden_states,
        gateup_input,
        src2dst,
        topk_ids,
        w1_input_scale,
        start_expert_id,
        end_expert_id,
        top_k,
        hidden_states.shape[1],
        BLOCK_SIZE=512,
    )

    seg_indptr_cur_rank = seg_indptr[start_expert_id : end_expert_id + 2]
    weight_indices_cur_rank = torch.arange(
        0,
        num_experts_per_partition,
        device=hidden_states.device,
        dtype=torch.int64,
    )

    # GroupGemm-0
    gateup_output = torch.empty(
        gateup_input.shape[0],
        w1.shape[1],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    gateup_output = grouped_gemm_triton(
        a=gateup_input,
        b=w1,
        c=gateup_output,
        batch_size=num_experts_per_partition,
        weight_column_major=True,
        seg_indptr=seg_indptr_cur_rank,
        weight_indices=weight_indices_cur_rank,
        use_fp8_w8a8=use_fp8_w8a8,
        scale_a=w1_input_scale,
        scale_b=w1_scale_inv,
        block_shape=block_shape,
    )

    # Act
    down_input = torch.empty(
        gateup_output.shape[0],
        gateup_output.shape[1] // 2,
        device=gateup_output.device,
        dtype=(
            fp8_dtype
            if (use_fp8_w8a8 and not use_blockwise_fp8)
            else hidden_states.dtype
        ),
    )
    if use_fp8_w8a8 and not use_blockwise_fp8:
        w2_input_scale = torch.ones(
            num_experts_per_partition,
            dtype=torch.float32,
            device=hidden_states.device,
        )
    else:
        w2_input_scale = None

    silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
        gateup_output,
        down_input,
        gateup_output.shape[1],
        reorder_topk_ids,
        w2_input_scale,
        start_expert_id,
        end_expert_id,
        BLOCK_SIZE=512,
    )

    # GroupGemm-1
    down_output = torch.empty(
        down_input.shape[0],
        w2.shape[1],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    down_output = grouped_gemm_triton(
        a=down_input,
        b=w2,
        c=down_output,
        batch_size=num_experts_per_partition,
        weight_column_major=True,
        seg_indptr=seg_indptr_cur_rank,
        weight_indices=weight_indices_cur_rank,
        use_fp8_w8a8=use_fp8_w8a8,
        scale_a=w2_input_scale,
        scale_b=w2_scale_inv,
        block_shape=block_shape,
    )

    # PostReorder
    output = torch.empty_like(hidden_states)
    post_reorder_triton_kernel[(hidden_states.size(0),)](
        down_output,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        start_expert_id,
        end_expert_id,
        top_k,
        hidden_states.size(1),
        BLOCK_SIZE=512,
    )
    return output

def ep_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    # ep config
    num_experts: int = 256,
    fp8_dtype: torch.types = torch.float8_e4m3fn,
    num_experts_per_partition: int = 128,
    start_expert_id: int = 0,
    end_expert_id: int = 127,
    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    use_fp8_w8a8: bool = False,
    w1_scale_inv: Optional[torch.Tensor] = None,
    w2_scale_inv: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
) -> torch.Tensor:
    use_blockwise_fp8 = block_shape is not None
    topk_weights, topk_ids = select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        # correction_bias=correction_bias, #skip this in test
        custom_routing_function=custom_routing_function,
    )

    output = ep_experts(
        hidden_states,
        w1,
        w2,
        top_k,
        # from ep_moe
        topk_weights,
        topk_ids,
        use_blockwise_fp8,
        # ep config
        num_experts,
        # fp8_dtype,
        num_experts_per_partition,
        start_expert_id,
        end_expert_id,
        use_fp8_w8a8,
        w1_scale_inv,
        w2_scale_inv,
        block_shape,
    )
    return output

def tune(
    gpu_id: int,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: List[int],
    search_space: List[Dict[str, int]],
    ep_size: int,
) -> Dict[str, int]:
    torch.cuda.set_device(gpu_id)
    print(f"Starting tuning on GPU {gpu_id} with batch sizes {num_tokens}")

    best_config = None
    best_time = float("inf")
    for config in tqdm(search_space):
        try:
            kernel_time = benchmark_config(
                config,
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a16,
                block_shape,
                num_iters=10,
                ep_size=ep_size,
            )
        except triton.runtime.autotuner.OutOfResources:
            # Some configurations may be invalid and fail to compile.
            continue

        if kernel_time < best_time:
            best_time = kernel_time
            best_config = config
    now = datetime.now()
    print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
    assert best_config is not None
    return best_config

def _distribute(inputs):
    return tune(*inputs)

class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: List[int] = None,
    num_iters: int = 100,
    ep_size: int = 4,
) -> float:
    torch.set_default_device("cuda")
    # print(f"default device in bs={num_tokens} is cuda:{torch.cuda.current_device()}")
    if block_shape is None:
        block_shape = [config["BLOCK_SIZE_N"], config["BLOCK_SIZE_K"]]
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    if use_int8_w8a16:
        w1 = torch.randint(
            -127,
            127,
            (
                num_experts,
                shard_intermediate_size,
                hidden_size,
            ),
            dtype=torch.int8,
        )
        w2 = torch.randint(
            -127,
            127,
            (
                num_experts,
                hidden_size,
                shard_intermediate_size // 2,
            ),
            dtype=torch.int8,
        )
    else:
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype
        )
    gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32)

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn(
            (num_experts, 2 * shard_intermediate_size), dtype=torch.float32
        )
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)
    if use_fp8_w8a8:
        if block_shape is None:
            w1_scale = torch.randn(num_experts, dtype=torch.float32)
            w2_scale = torch.randn(num_experts, dtype=torch.float32)
            a1_scale = torch.randn(1, dtype=torch.float32)
            a2_scale = torch.randn(1, dtype=torch.float32)
        else:
            block_n, block_k = block_shape[0], block_shape[1]
            n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
            n_tiles_w2 = (hidden_size + block_n - 1) // block_n
            k_tiles_w1 = (hidden_size + block_k - 1) // block_k
            k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
            w1_scale = torch.rand(
                (num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.float32
            )
            w2_scale = torch.rand(
                (num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.float32
            )

        w1 = w1.to(torch.float8_e4m3fnuz if _is_hip_ else torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fnuz if _is_hip_ else torch.float8_e4m3fn)

    input_gating = torch.empty(num_tokens, num_experts, dtype=torch.float32)

    num_experts_per_partition = num_experts // ep_size
    cur_rank = torch.randint(0, ep_size - 1, (1,), dtype=torch.int32)[0].item()
    start_id = cur_rank * num_experts_per_partition
    end_id = start_id + num_experts_per_partition - 1

    def prepare(i: int):
        input_gating.copy_(gating_output[i])

    def run():
        with torch.inference_mode():
            ep_moe(
                hidden_states=x,
                w1=w1,
                w2=w2,
                router_logits=input_gating,
                top_k=topk,
                renormalize=False,
                use_fp8_w8a8=use_fp8_w8a8,
                w1_scale_inv=w1_scale,
                w2_scale_inv=w2_scale,
                block_shape=block_shape,
                num_experts=num_experts,
                num_experts_per_partition=num_experts_per_partition,
                start_expert_id=start_id,
                end_expert_id=end_id,
            )

    # JIT compilation & warmup
    run()
    torch.cuda.synchronize()

    # Capture 10 invocations with CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: List[float] = []
    for i in range(num_iters):
        prepare(i)
        torch.cuda.synchronize()

        start_event.record()
        graph.replay()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    graph.reset()
    return avg


def get_rocm_configs_compute_bound() -> List[Dict[str, int]]:
    configs: List[BenchmarkConfig] = []
    waves_per_eu_range = 0
    for num_stages in [2]:
        for block_m in [32, 64, 128, 256]:
            for block_k in [32, 64, 128, 256]:
                for block_n in [16, 32, 64, 128, 256]:
                    for num_warps in [1, 2, 4, 8]:
                        for group_size in [1, 4, 8, 16, 32]:
                            configs.append(
                                {
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_SIZE_M": group_size,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                    "waves_per_eu": waves_per_eu_range,
                                }
                            )
    return configs


def get_configs_compute_bound() -> List[Dict[str, int]]:
    # Reduced search space for faster tuning.
    # TODO(woosuk): Increase the search space and use a performance model to
    # prune the search space.
    configs: List[BenchmarkConfig] = []
    if _is_hip_:
        configs = get_rocm_configs_compute_bound()
    else:
        for num_stages in [2, 3, 4, 5]:
            for block_m in [16, 32, 64, 128, 256]:
                for block_k in [64, 128, 256]:
                    for block_n in [32, 64, 128, 256]:
                        for num_warps in [4, 8]:
                            for group_size in [1, 16, 32, 64]:
                                configs.append(
                                    {
                                        "BLOCK_SIZE_M": block_m,
                                        "BLOCK_SIZE_N": block_n,
                                        "BLOCK_SIZE_K": block_k,
                                        "GROUP_SIZE_M": group_size,
                                        "num_warps": num_warps,
                                        "num_stages": num_stages,
                                    }
                                )
    return configs


def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
        **(
            {"waves_per_eu": config["waves_per_eu"]} if "waves_per_eu" in config else {}
        ),
    }


def save_configs(
    configs: Dict[int, BenchmarkConfig],
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: List[int],
) -> None:
    dtype_str = get_config_dtype_str(
        dtype, use_int8_w8a16=use_int8_w8a16, use_fp8_w8a8=use_fp8_w8a8
    )

    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    filename = get_config_file_name(
        num_experts,
        shard_intermediate_size // 2,
        dtype_str,
        block_shape,
    )

    print(f"Writing best config to {filename}...")
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def main(args: argparse.Namespace):
    print(args)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if config.architectures[0] == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "JambaForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "Qwen2MoeForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]:
        E = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    else:
        # Default: Mixtral
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size

    hidden_size = config.hidden_size
    dtype = config.torch_dtype
    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"
    block_shape = None
    if (
        hasattr(config, "quantization_config")
        and "weight_block_size" in config.quantization_config
    ):
        block_shape = config.quantization_config["weight_block_size"]
        assert len(block_shape) == 2

    if args.batch_size is None:
        batch_sizes = [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
        ]
    else:
        batch_sizes = [args.batch_size]

    num_gpus = get_available_gpu_count()

    search_space = get_configs_compute_bound()
    if block_shape is not None:
        block_n, block_k = block_shape[0], block_shape[1]
        search_space = [
            config
            for config in search_space
            if block_k % config["BLOCK_SIZE_K"] == 0
        ]
    print(f"Start tuning over {len(search_space)} configurations...")

    tune_args = [
            (
                i % num_gpus,
                batch_size,
                E,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a16,
                block_shape,
                search_space,
                args.ep_size,
            )
            for i, batch_size in enumerate(batch_sizes)
        ]

    start = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(num_gpus) as pool:
        configs = pool.map(_distribute, tune_args)

    best_configs = {
        M: sort_config(config) for M, config in zip(batch_sizes, configs)
    }
    save_configs(
        best_configs,
        E,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype,
        use_fp8_w8a8,
        use_int8_w8a16,
        block_shape,
    )
    end = time.time()
    print(f"Tuning took {end - start:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    parser.add_argument("--tp-size", "-tp", type=int, default=2)
    parser.add_argument(
        "--dtype", type=str, choices=["auto", "fp8_w8a8", "int8_w8a16"], default="auto"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--ep-size", type=int, default=4)
    args = parser.parse_args()

    main(args)
