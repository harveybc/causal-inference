"""Declared temporal DAG utilities: descendants, d-separation and the back-door criterion.

The DAG is a declaration by the analyst (each edge an assumption that can be
attacked). These functions only check what the declaration implies; they never
learn or certify a graph.
"""

from __future__ import annotations


def _edges(dag):
    return [(str(a), str(b)) for a, b in dag.get("edges", [])]


def _nodes(dag):
    nodes = set(map(str, dag.get("nodes", [])))
    for a, b in _edges(dag):
        nodes.update((a, b))
    return nodes


def parents(dag, node):
    return {a for a, b in _edges(dag) if b == node}


def children(dag, node):
    return {b for a, b in _edges(dag) if a == node}


def descendants(dag, node):
    seen, stack = set(), [node]
    while stack:
        for c in children(dag, stack.pop()):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def ancestors(dag, nodes):
    seen, stack = set(nodes), list(nodes)
    while stack:
        for p in parents(dag, stack.pop()):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def is_acyclic(dag):
    return all(n not in descendants(dag, n) for n in _nodes(dag))


def d_separated(dag, xs, ys, zs):
    """True when every x is d-separated from every y given zs (moralised ancestral graph)."""
    xs, ys, zs = set(xs), set(ys), set(zs)
    keep = ancestors(dag, xs | ys | zs)
    sub = [(a, b) for a, b in _edges(dag) if a in keep and b in keep]
    und = {n: set() for n in keep}
    for a, b in sub:
        und[a].add(b)
        und[b].add(a)
    for child in keep:
        ps = [a for a, b in sub if b == child]
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                und[ps[i]].add(ps[j])
                und[ps[j]].add(ps[i])
    seen = set(xs)
    stack = list(xs)
    while stack:
        n = stack.pop()
        if n in ys:
            return False
        for m in und.get(n, ()):
            if m not in seen and m not in zs:
                seen.add(m)
                stack.append(m)
    return True


def backdoor_check(dag, treatment, outcome, adjustment, observed=None):
    """Return (ok, reasons, excluded) for the back-door criterion on the declared DAG.

    ``excluded`` names the descendants of the treatment (mediators, colliders through
    the treatment, later outcomes) so a reader can attack the choice.
    """
    reasons = []
    nodes = _nodes(dag)
    adjustment = list(adjustment)
    if not is_acyclic(dag):
        return False, ["DAG_NOT_ACYCLIC"], []
    for name in (treatment, outcome):
        if name not in nodes:
            reasons.append("NODE_NOT_IN_DAG")
    unknown = [z for z in adjustment if z not in nodes]
    if unknown:
        reasons.append("ADJUSTMENT_NODE_NOT_IN_DAG")
    if observed is not None:
        hidden = [z for z in adjustment if z not in observed]
        if hidden:
            reasons.append("ADJUSTMENT_VARIABLE_NOT_OBSERVED")
    desc = descendants(dag, treatment)
    excluded = sorted(desc - {outcome})
    if any(z in desc for z in adjustment):
        reasons += ["ADJUSTMENT_CONTAINS_DESCENDANT", "BACKDOOR_NOT_SATISFIED"]
    if not reasons:
        cut = {"nodes": list(nodes), "edges": [[a, b] for a, b in _edges(dag) if a != treatment]}
        if not d_separated(cut, {treatment}, {outcome}, set(adjustment)):
            reasons.append("BACKDOOR_NOT_SATISFIED")
    return (not reasons), sorted(set(reasons)), excluded
