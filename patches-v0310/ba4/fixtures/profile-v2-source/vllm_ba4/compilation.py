"""Keep BA4 opaque only in the size-four piecewise specialization.

The symbolic node is a trace marker. General-size Inductor must see the
original linear, including its neighbors, rather than execute its fallback
behind a custom-op boundary. Do not mutate the shared Dynamo graph.
"""
import copy

import torch


def _dispatch_nodes(graph):
    return [n for n in graph.graph.nodes if n.op == "call_function"
            and n.target == torch.ops.vllm.sm12x_ba4_dispatch.default]


def compile_sizes(sizes, graph, compiled_runnables):
    # Warm/AOT loading has no FX graph. The serialized size-four entry is
    # authoritative there, so cold and warm paths build the same range table.
    selected = (_dispatch_nodes(graph) if graph is not None
                else compiled_runnables is not None and "(4, 4)" in compiled_runnables)
    if selected and 4 not in (sizes or []):
        return [*(sizes or []), 4]
    return sizes


def graph_for_range(graph, compile_range):
    general = not (compile_range.is_single_size() and compile_range.start == 4)
    probes = [n for n in graph.graph.nodes if n.op == "call_function"
              and n.target == torch.ops.vllm.sm12x_ba4_offset_metadata.default
              and not n.users]
    if not probes and (not general or not _dispatch_nodes(graph)):
        return graph
    result = copy.deepcopy(graph)
    # These pure, unused metadata probes have served Dynamo's trace-time
    # admission checks. Remove exactly them, never generic dead-code cleanup
    # (attention nodes can carry mutations that FX does not recognize).
    for node in list(result.graph.nodes):
        if (node.op == "call_function"
                and node.target == torch.ops.vllm.sm12x_ba4_offset_metadata.default
                and not node.users):
            result.graph.erase_node(node)
    nodes = _dispatch_nodes(result) if general else []
    result._ba4_scratch_inputs = {n.args[2].name for n in nodes}
    for node in nodes:
        x, weight, *_ = node.args
        node.target = torch._C._nn.linear
        node.args = (x, weight, None)
        node.kwargs = {}
    # Do not run generic FX DCE: vLLM attention packets carry mutation
    # side effects that FX Node.is_impure does not necessarily recognize.
    result.graph.lint()
    result.recompile()
    return result


def prune_inputs(graph, args):
    """Remove the now-unused scratch from general-size compiler inputs.

    The piecewise caller retains its shared signature; selection of these
    inputs belongs to that caller, outside the generated numerical program.
    """
    placeholders = [n for n in graph.graph.nodes if n.op == "placeholder"]
    removed = {n for n in placeholders if not n.users
               and n.name in graph._ba4_scratch_inputs}
    indices = tuple(i for i, n in enumerate(placeholders) if n not in removed)
    if len(indices) == len(placeholders):
        return graph, args, None
    for node in removed:
        graph.graph.erase_node(node)
    graph.recompile()
    return graph, [args[i] for i in indices], indices


_INPUTS = ";ba4-inputs="  # no underscore: AOT keys use rsplit("_", 1)


def artifact_key(key, indices):
    return key if indices is None else key + _INPUTS + ",".join(map(str, indices))


def loaded_key(key, runnables):
    matches = [k for k in runnables if k == key or k.startswith(key + _INPUTS)]
    if len(matches) != 1:
        raise RuntimeError("BA4 compiled input signature missing or ambiguous: " + key)
    found = matches[0]
    indices = (tuple(map(int, found.split(_INPUTS, 1)[1].split(",")))
               if _INPUTS in found else None)
    return found, indices
