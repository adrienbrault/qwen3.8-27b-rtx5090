"""Outside-capture replay timing and independent trace accounting."""
import math

TIMING = 'external_events_whole_interleaved_replay-v2'


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def replay_us(cuda, graph):
    # The graph itself contains NO event-record nodes. Events bracket the
    # complete replay on its launch stream, including every helper/weight read.
    if cuda.is_current_stream_capturing():
        raise RuntimeError('replay timing must run outside capture')
    start, end = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end) * 1000
    if not positive(elapsed):
        raise ValueError('nonfinite or nonpositive replay interval')
    return elapsed


def validate_replay_timer(torch):
    """GPU-only sanity gate: sixteen GEMMs must take longer than one."""
    import statistics
    x = torch.randn((1024, 1024), dtype=torch.bfloat16, device='cuda')
    w, out = torch.empty_like(x).normal_(), torch.empty_like(x)
    for _ in range(3):
        torch.mm(x, w, out=out)
    torch.cuda.synchronize()
    graphs = []
    for count in (1, 16):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(count):
                torch.mm(x, w, out=out)
        graph.replay()
        torch.cuda.synchronize()
        graphs.append(graph)
    samples = [[replay_us(torch.cuda, graph) for _ in range(3)] for graph in graphs]
    if statistics.median(samples[1]) <= 2 * statistics.median(samples[0]):
        raise ValueError('outside-graph timer failed work-scaling sanity gate')
    return {'single_work_us': samples[0], 'sixteen_work_us': samples[1]}


def trace_layers(events, calls=48):
    """Attribute CUDA kernels between each before/after weight-stream kernel.

    Input is the independent Chrome trace, not captured event timestamps.
    Fail closed if the profiler cannot expose graph kernels or attribution.
    Copies, memset and reduction helpers within each bracket are included.
    """
    gpu = sorted((e for e in events if e.get('ph') == 'X'
                  and e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')),
                 key=lambda e: e['ts'])
    layers, group, reading = [], [], False
    stream_ids = {e.get('args', {}).get('stream') for e in gpu}
    if len(stream_ids) != 1 or None in stream_ids or not gpu:
        raise ValueError('need one ordered CUDA stream for trace attribution')
    for event in gpu:
        if not positive(event.get('dur')):
            raise ValueError('invalid CUDA trace duration')
        if '_stream_weights' in event['name']:
            if reading:
                if not group:
                    raise ValueError('empty BA trace bracket')
                layers.append(group)
                group = []
            reading = not reading
        elif reading:
            group.append(event)
        else:
            raise ValueError('unattributed CUDA helper outside BA brackets')
    if reading or len(layers) != calls:
        raise ValueError('need exactly 48 complete BA trace brackets')
    costs = [sum(e['dur'] for e in group) for group in layers]
    return {'per_layer_us': costs, 'b1_trace_us_per_step': sum(costs),
            'all_kernel_us': sum(e['dur'] for e in gpu),
            'cuda_symbols': sorted({e['name'] for group in layers for e in group}),
            'kernel_counts': [len(group) for group in layers]}
