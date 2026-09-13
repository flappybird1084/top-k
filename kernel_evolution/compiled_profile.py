"""Deterministic attribution of observed CUDA work to Inductor source regions.

Fused region time is an upper bound on any constituent operation's time, never
an estimate of its exclusive cost. Unknown names and external kernels stay in
the report; neither eager timing nor a guessed allocation fills those gaps.
"""
import ast
import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path


_HEADER = re.compile(r'^# Topologically Sorted Source Nodes: \[(.*?)\], Original ATen: \[(.*?)\]', re.M)
_ASSIGN = re.compile(r'^(\w+)\s*=\s*async_compile\.triton\(', re.M)
_KERNEL_NAME = re.compile(r'\btriton_[A-Za-z0-9_]+')


def parse_source(source, path='<source>'):
    """Read generated code as data; never import or execute compiler output."""
    tree = ast.parse(source)
    assignments = {node.lineno: node for node in tree.body if isinstance(node, ast.Assign)}
    regions = []
    previous = 0
    for match in _ASSIGN.finditer(source):
        prefix = source[previous:match.start()]
        headers = list(_HEADER.finditer(prefix))
        header = headers[-1] if headers else None
        lineno = source.count('\n', 0, match.start()) + 1
        node = assignments.get(lineno)
        call = node.value if node is not None else None
        body = ''
        if isinstance(call, ast.Call) and len(call.args) > 1:
            value = call.args[1]
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                body = value.value
        metadata = prefix[header.start():] if header else ''
        aten = [s.strip() for s in header.group(2).split(',') if s.strip()] if header else []
        nodes = [s.strip() for s in header.group(1).split(',') if s.strip()] if header else []
        regions.append(dict(kernel_name=match.group(1), source_path=str(path), source_line=lineno,
                            source_hash=hashlib.sha256(body.encode()).hexdigest(),
                            original_aten=aten, source_nodes=nodes, graph_fragment=metadata,
                            has_triton_jit='@triton.jit' in body))
        previous = match.end()
    return regions


def classify(region, known_ops, ema_decay=None):
    """Identify associated contracts, allowing explicit shared-region accounting."""
    aten = set(region['original_aten'])
    associated = []
    if 'aten.native_layer_norm_backward' in aten:
        associated.append('layer_norm_backward')
    if 'aten.gelu' in aten and aten.intersection({'aten.addmm', 'aten.mm', 'aten.linear'}):
        associated.append('gelu_mlp')
    if 'aten.gather' in aten and 'aten.add' in aten:
        associated.append('masked_gather_add')
    # Generic pointwise additions are not evidence of EMA. Require the exact
    # observed adapter decay and the compiler's arithmetic dataflow constants.
    if ema_decay is not None and aten == {'aten.mul', 'aten.add', 'aten.copy_'}:
        constants = re.findall(r'target=torch\.ops\.aten\.mul\.Tensor\]\(args = \(%\w+, ([\d.eE+-]+)\)', region['graph_fragment'])
        if len(constants) == 2:
            values = sorted(float(x) for x in constants)
            expected = sorted([float(ema_decay), 1. - float(ema_decay)])
            if all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) for a, b in zip(values, expected)):
                associated.append('ema_update')
    return sorted(set(associated).intersection(known_ops))


def attribute(regions, kernel_events, *, steps, step_time_ms, known_ops, ema_decay=None):
    """kernel_events are CUDA leaf events: {'name': str, 'duration_us': float}."""
    if steps <= 0 or step_time_ms <= 0:
        raise ValueError('Positive profiled step count and measured step duration required')
    index = defaultdict(list)
    for region in regions:
        index[region['kernel_name']].append(region)
    totals = defaultdict(float)
    counts = defaultdict(int)
    for event in kernel_events:
        duration = float(event['duration_us'])
        if not math.isfinite(duration) or duration < 0:
            raise ValueError('Invalid CUDA event duration')
        totals[event['name']] += duration / steps / 1000
        counts[event['name']] += 1
    rows = []
    by_op = {op: dict(region_time_ms=0., regions=[], attribution='associated_fused_region_upper_bound') for op in known_ops}
    for name, duration in sorted(totals.items(), key=lambda pair: -pair[1]):
        tokens = _KERNEL_NAME.findall(name)
        matches = [r for token in set(tokens) for r in index.get(token, [])]
        # Duplicate source captures may describe the same kernel. Conflicting
        # metadata/source hashes for a reused name make attribution ambiguous.
        signatures = {(r['source_hash'], tuple(r['original_aten'])) for r in matches}
        unique = len(signatures) == 1
        associated = classify(matches[0], known_ops, ema_decay) if unique else []
        row = dict(name=name, cuda_time_ms=duration, calls=counts[name],
                   pct_step_time=100 * duration / step_time_ms,
                   mapping='matched' if unique else ('ambiguous' if matches else 'unmapped'),
                   associated_ops=associated,
                   sources=[{k:r[k] for k in ('source_path','source_line','kernel_name','original_aten','source_nodes','source_hash')} for r in matches])
        rows.append(row)
        for op in associated:
            by_op[op]['region_time_ms'] += duration
            by_op[op]['regions'].append(row)
    for entry in by_op.values():
        entry['pct_step_time_upper_bound'] = 100 * entry['region_time_ms'] / step_time_ms
    return dict(backend='whole_step_inductor', steps=steps, step_time_ms=step_time_ms,
                total_cuda_time_ms=sum(totals.values()), kernels=rows, operations=by_op,
                unmapped_cuda_time_ms=sum(r['cuda_time_ms'] for r in rows if r['mapping']!='matched'),
                accounting='CUDA leaf durations; fused region costs overlap between associated operations and are not exclusive operation time')


def profile_harness(harness):
    """Profile the actual warmed compiled training step, including AdamW/hooks.

    Caller owns the GPU. Call after full-step compilation/source capture and
    correctness validation. Profiling does not propose or supply any kernels.
    """
    import statistics
    import torch
    from kernel_evolution.runtime import cuda_times
    if harness.config.get('step_backend') != 'inductor':
        raise ValueError('Compiled attribution requires the inductor step backend')
    fn = harness.step_callable({})
    harness.restore()
    for _ in range(harness.config['profile_warmup']):
        fn()
    torch.cuda.synchronize()
    steps = harness.config['profile_steps']
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            fn()
    torch.cuda.synchronize()
    elapsed = statistics.median(cuda_times(fn, 10, 3))
    events = [dict(name=e.name, duration_us=e.time_range.elapsed_us()) for e in prof.events()
              if e.device_type == torch.autograd.DeviceType.CUDA and not e.cpu_children]
    regions = [region for path in harness.compiled_step_sources for region in parse_source(Path(path).read_text(),path)]
    decays = {float(binding.decay) for binding in harness.ema_bindings}
    report = attribute(regions,events,steps=steps,step_time_ms=elapsed,
                       known_ops=list(harness.cases),ema_decay=next(iter(decays)) if len(decays)==1 else None)
    report['source_paths'] = list(harness.compiled_step_sources)
    harness.restore()
    return report


def augment_targets(profile, report, *, allowed_ops, min_pct_step_time):
    """Use associated compiled region cost for eligibility, explicitly as a bound.

    A candidate still must win gate 4 against the complete compiled step. The
    bound is a search filter, never a claimed opportunity or expected speedup.
    """
    for target in profile['targets']:
        op = target.get('fusion_of', target['id'])
        data = report['operations'].get(op, {})
        cost = data.get('region_time_ms', 0.)
        pct = data.get('pct_step_time_upper_bound', 0.)
        target['eager_pct_step_time'] = target['pct_step_time']
        target['eager_op_time_ms'] = target['op_time_ms']
        target.update(pct_step_time=pct,op_time_ms=cost,compiled_attribution=data,
                      timing_basis='associated_compiled_region_upper_bound',
                      eligible=op in allowed_ops and pct >= min_pct_step_time and bool(data.get('regions')))
    profile['compiled_profile'] = report
    profile['eager_step_time_ms'] = profile['step_time_ms']
    profile['step_time_ms'] = report['step_time_ms']
    return profile
