import torch


def setup_cuda_device():
    torch_device = torch.device("cuda")
    assert torch.cuda.is_available()
    torch.set_default_device(torch_device)
    return torch_device


def benchmark_forward(
    forward,
    *,
    warmup_iters,
    measure_iters,
):
    with torch.inference_mode():
        for _ in range(warmup_iters):
            result = forward()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(measure_iters):
            result = forward()
        end_event.record()
        torch.cuda.synchronize()
        total_ms = start_event.elapsed_time(end_event)

    return total_ms / measure_iters, result


def add_common_benchmark_args(
    parser,
    *,
    warmup_default,
    measure_default,
):
    parser.add_argument(
        "--sequence_lengths", type=int, nargs="+",
        default=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_query_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--warmup_iters", type=int, default=warmup_default)
    parser.add_argument("--measure_iters", type=int, default=measure_default)
    return parser


def format_run_description(
    benchmark_name,
    *,
    device,
    args,
    extra_fields=None,
):
    base = (
        f"Benchmarking {benchmark_name} on {device} "
        f"(batch={args.batch_size}, q_heads={args.num_query_heads}, "
        f"kv_heads={args.num_kv_heads}, head_dim={args.head_dim}"
    )
    if extra_fields:
        extras = ", ".join(f"{name}={value}" for name, value in extra_fields)
        base = f"{base}, {extras}"
    return f"{base})"


def format_iteration_summary(
    *,
    warmup_iters,
    measure_iters,
):
    return f"Warmup iterations: {warmup_iters}, Timed iterations: {measure_iters}"
