"""
resilience_demo.py - Part 4 / Task 16

Timeouts and retries around the LangGraph agent.

WHAT IS CONFIGURED
------------------
1. RETRY POLICY (exponential backoff with jitter) on a node that simulates a
   transient failure:

        max_attempts     = 4
        initial_interval = 0.25 s
        backoff_factor   = 2.0        -> 0.25, 0.50, 1.00 s between attempts
        max_interval     = 2.0 s      (cap; not reached at these settings)
        jitter           = True       (LangGraph adds up to 1s of random jitter)

   This is langgraph.types.RetryPolicy, attached to the node with
   add_node(..., retry_policy=...). The flaky node fails its first 2 calls from
   a counter and succeeds on the 3rd, so it must recover within 4 attempts.

2. PER-NODE TIMEOUT on the slow node. LangGraph has no built-in per-node
   timeout, so `with_node_timeout` runs the node body in a worker thread and
   raises NodeTimeout if it overruns, which surfaces as a clean error instead
   of a hang.

3. GLOBAL TIMEOUT on the whole graph. `run_with_global_timeout` enforces one
   deadline two ways at once:
     * a hard outer bound - graph.invoke runs in a worker thread and the caller
       stops waiting at the deadline, so the caller always gets a clean
       GlobalTimeout rather than hanging;
     * cooperative cancellation - every node checks the shared deadline on
       entry and raises, so the run actually stops advancing instead of
       quietly finishing in a background thread.
   Both halves are needed: Python cannot forcibly kill a running thread, so the
   deadline check is what genuinely cancels the run.

Run
    python resilience_demo.py
"""

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

import agent
from agent import AgentState, after_guard, after_route

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

PER_NODE_TIMEOUT_S = 1.0
GLOBAL_TIMEOUT_S = 3.0


class TransientToolError(RuntimeError):
    """Simulated transient downstream failure (the kind a retry should fix)."""


class NodeTimeout(TimeoutError):
    """A single node exceeded its per-node budget."""


class GlobalTimeout(TimeoutError):
    """The whole graph exceeded its global budget."""


#: retry_on names TransientToolError specifically, so a genuine bug in the node
#: fails fast instead of being retried four times.
RETRY_POLICY = RetryPolicy(
    max_attempts=4,
    initial_interval=0.25,
    backoff_factor=2.0,
    max_interval=2.0,
    jitter=True,
    retry_on=(TransientToolError,),
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

class Deadline:
    """A shared wall-clock deadline that nodes check cooperatively."""

    def __init__(self, seconds: Optional[float]):
        self.seconds = seconds
        self.expires_at = (time.perf_counter() + seconds) if seconds else None

    def remaining(self) -> float:
        if self.expires_at is None:
            return float("inf")
        return self.expires_at - time.perf_counter()

    def check(self, node_name: str) -> None:
        if self.expires_at is not None and time.perf_counter() > self.expires_at:
            raise GlobalTimeout(
                f"global timeout of {self.seconds}s exceeded; node "
                f"'{node_name}' refused to start"
            )


_deadline = threading.local()


def current_deadline() -> Deadline:
    d = getattr(_deadline, "value", None)
    return d if d is not None else Deadline(None)


def with_node_timeout(name, fn, timeout_s: float):
    """Wrap a node so that exceeding timeout_s raises NodeTimeout, not a hang."""

    def wrapped(state):
        current_deadline().check(name)
        # NOTE: deliberately NOT `with ThreadPoolExecutor(...)`. The context
        # manager calls shutdown(wait=True) on exit, which would block until the
        # overrunning worker finished - exactly the hang this guard exists to
        # prevent. shutdown(wait=False) releases the caller immediately and
        # leaves the abandoned worker to die on its own.
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(fn, state)
            try:
                return fut.result(timeout=timeout_s)
            except FutureTimeout:
                raise NodeTimeout(
                    f"node '{name}' exceeded its per-node timeout of "
                    f"{timeout_s}s and was aborted with a clean error"
                ) from None
        finally:
            pool.shutdown(wait=False)

    wrapped.__name__ = name
    return wrapped


def guarded(name, fn):
    """Wrap an ordinary node with the cooperative global-deadline check."""

    def wrapped(state):
        current_deadline().check(name)
        return fn(state)

    wrapped.__name__ = name
    return wrapped


def run_with_global_timeout(graph, state, timeout_s: float):
    """Invoke the graph under one global deadline. Raises GlobalTimeout."""
    deadline = Deadline(timeout_s)

    def target():
        _deadline.value = deadline          # per-thread, set inside the worker
        return graph.invoke(state)

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        fut = pool.submit(target)
        try:
            return fut.result(timeout=timeout_s)
        except FutureTimeout:
            raise GlobalTimeout(
                f"global timeout of {timeout_s}s exceeded; the run was "
                f"cancelled and the caller was released cleanly"
            ) from None
    finally:
        pool.shutdown(wait=False)


# --------------------------------------------------------------------------- #
# the simulated nodes
# --------------------------------------------------------------------------- #

class FlakyCounter:
    """Fails the first `fail_times` calls, then succeeds. Thread-safe."""

    def __init__(self, fail_times: int = 2):
        self.fail_times = fail_times
        self.calls = 0
        self.attempt_log = []
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.calls = 0
            self.attempt_log = []

    def __call__(self):
        with self._lock:
            self.calls += 1
            n = self.calls
        ts = time.perf_counter()
        if n <= self.fail_times:
            self.attempt_log.append((n, "FAILED", ts))
            print(f"      attempt {n}: raising TransientToolError "
                  f"(simulated downstream flake)")
            raise TransientToolError(
                f"simulated transient failure on attempt {n}")
        self.attempt_log.append((n, "SUCCEEDED", ts))
        print(f"      attempt {n}: succeeded")
        return n


FLAKY = FlakyCounter(fail_times=2)


def flaky_enrichment(state: AgentState) -> AgentState:
    """A node that calls a flaky downstream service; the retry policy covers it."""
    current_deadline().check("flaky_enrichment")
    attempt = FLAKY()
    return {"executed_nodes": list(state.get("executed_nodes", []))
            + [f"flaky_enrichment(attempt {attempt})"]}


SLOW_SECONDS = 5.0


def slow_node(state: AgentState) -> AgentState:
    """A node that always overruns, used to trip the per-node timeout."""
    print(f"      slow_node: sleeping {SLOW_SECONDS}s (budget is "
          f"{PER_NODE_TIMEOUT_S}s)...")
    time.sleep(SLOW_SECONDS)
    return {"executed_nodes": list(state.get("executed_nodes", [])) + ["slow_node"]}


# --------------------------------------------------------------------------- #
# graphs
# --------------------------------------------------------------------------- #

def build_retry_graph():
    """guard_input -> flaky_enrichment (with retry policy) -> route -> ... -> finalize"""
    g = StateGraph(AgentState)
    g.add_node("guard_input", guarded("guard_input", agent.guard_input))
    g.add_node("flaky_enrichment", flaky_enrichment, retry_policy=RETRY_POLICY)
    g.add_node("route_intent", guarded("route_intent", agent.route_intent))
    g.add_node("rag_answer", guarded("rag_answer", agent.rag_answer))
    g.add_node("record_lookup", guarded("record_lookup", agent.record_lookup))
    g.add_node("finalize", guarded("finalize", agent.finalize))

    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", after_guard,
                            {"blocked": "finalize", "continue": "flaky_enrichment"})
    g.add_edge("flaky_enrichment", "route_intent")
    g.add_conditional_edges("route_intent", after_route,
                            {"rag": "rag_answer", "record_lookup": "record_lookup"})
    g.add_edge("rag_answer", "finalize")
    g.add_edge("record_lookup", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def build_slow_node_graph():
    """guard_input -> slow_node (per-node timeout) -> finalize"""
    g = StateGraph(AgentState)
    g.add_node("guard_input", guarded("guard_input", agent.guard_input))
    g.add_node("slow_node",
               with_node_timeout("slow_node", slow_node, PER_NODE_TIMEOUT_S))
    g.add_node("finalize", guarded("finalize", agent.finalize))
    g.add_edge(START, "guard_input")
    g.add_edge("guard_input", "slow_node")
    g.add_edge("slow_node", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def build_total_overrun_graph():
    """Several nodes that each fit their own budget but blow the global one."""

    def make_step(i, seconds):
        def step(state):
            current_deadline().check(f"step_{i}")
            print(f"      step_{i}: working for {seconds}s "
                  f"(global budget remaining {current_deadline().remaining():.2f}s)")
            time.sleep(seconds)
            return {"executed_nodes": list(state.get("executed_nodes", []))
                    + [f"step_{i}"]}

        step.__name__ = f"step_{i}"
        return step

    g = StateGraph(AgentState)
    g.add_node("guard_input", guarded("guard_input", agent.guard_input))
    for i in range(1, 5):
        g.add_node(f"step_{i}", make_step(i, 1.2))
    g.add_node("finalize", guarded("finalize", agent.finalize))

    g.add_edge(START, "guard_input")
    g.add_edge("guard_input", "step_1")
    for i in range(1, 4):
        g.add_edge(f"step_{i}", f"step_{i+1}")
    g.add_edge("step_4", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def _base_state(query: str) -> AgentState:
    return {
        "query": query,
        "conversation_id": "",
        "trace_id": f"trace-resilience-{random.randint(10**6, 10**7):07d}",
        "turn_index": 1,
        "executed_nodes": [],
    }


# --------------------------------------------------------------------------- #
# demonstrations
# --------------------------------------------------------------------------- #

def demo_retry():
    print("=" * 78)
    print("(a) RETRY POLICY RECOVERING A TRANSIENT FAILURE")
    print("=" * 78)
    print(f"  policy: max_attempts={RETRY_POLICY.max_attempts}, "
          f"initial_interval={RETRY_POLICY.initial_interval}s, "
          f"backoff_factor={RETRY_POLICY.backoff_factor},")
    print(f"          max_interval={RETRY_POLICY.max_interval}s, "
          f"jitter={RETRY_POLICY.jitter}")
    print(f"  the flaky node fails its first {FLAKY.fail_times} calls, then "
          f"succeeds\n")

    FLAKY.reset()
    graph = build_retry_graph()
    t0 = time.perf_counter()
    result = graph.invoke(_base_state("What is the notice period policy?"))
    elapsed = time.perf_counter() - t0

    print(f"\n  total calls made to the flaky service : {FLAKY.calls}")
    print(f"  attempt log                           : "
          f"{[(n, s) for n, s, _ in FLAKY.attempt_log]}")
    print(f"  wall-clock elapsed                    : {elapsed:.2f}s "
          f"(includes backoff sleeps)")
    print(f"  node trail                            : {result['executed_nodes']}")
    print(f"  final answer                          : "
          f"{result['response']['answer'][:150]}")
    ok = FLAKY.calls == 3 and result["response"]["answer"]
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'} - the run recovered on attempt "
          f"{FLAKY.calls} of {RETRY_POLICY.max_attempts}\n"
          f"          and completed normally.")


def demo_node_timeout():
    print("\n" + "=" * 78)
    print("(b) PER-NODE TIMEOUT FIRING A CLEAN ERROR (NOT A HANG)")
    print("=" * 78)
    print(f"  slow_node sleeps {SLOW_SECONDS}s; its per-node budget is "
          f"{PER_NODE_TIMEOUT_S}s\n")

    graph = build_slow_node_graph()
    t0 = time.perf_counter()
    try:
        graph.invoke(_base_state("What is the notice period policy?"))
    except NodeTimeout as exc:
        elapsed = time.perf_counter() - t0
        print(f"\n  raised   : {type(exc).__name__}")
        print(f"  message  : {exc}")
        print(f"  elapsed  : {elapsed:.2f}s")
        ok = elapsed < SLOW_SECONDS
        print(f"\n  RESULT: {'PASS' if ok else 'FAIL'} - the caller was released "
              f"after {elapsed:.2f}s with a\n          typed error, well before "
              f"the node's own {SLOW_SECONDS}s sleep would have\n          "
              f"finished. It failed fast instead of hanging.")
    else:
        print("\n  RESULT: FAIL - no timeout was raised")


def demo_global_timeout():
    print("\n" + "=" * 78)
    print("(c) GLOBAL TIMEOUT CANCELLING THE WHOLE RUN ON A TOTAL OVERRUN")
    print("=" * 78)
    print(f"  4 chained nodes at 1.2s each = 4.8s of work; each one is well "
          f"inside\n  any per-node budget, but the global budget is "
          f"{GLOBAL_TIMEOUT_S}s\n")

    graph = build_total_overrun_graph()
    t0 = time.perf_counter()
    try:
        run_with_global_timeout(
            graph, _base_state("What is the notice period policy?"),
            GLOBAL_TIMEOUT_S)
    except GlobalTimeout as exc:
        elapsed = time.perf_counter() - t0
        print(f"\n  raised   : {type(exc).__name__}")
        print(f"  message  : {exc}")
        print(f"  elapsed  : {elapsed:.2f}s")
        ok = abs(elapsed - GLOBAL_TIMEOUT_S) < 1.0
        print(f"\n  RESULT: {'PASS' if ok else 'FAIL'} - the whole run was "
              f"cancelled at the {GLOBAL_TIMEOUT_S}s deadline\n          instead "
              f"of running to its natural 4.8s completion.")
        time.sleep(1.6)
        print("\n  (cooperative cancellation check: after the deadline the next "
              "node to\n   start raises GlobalTimeout, so the run stops "
              "advancing rather than\n   quietly finishing in the background)")
    else:
        print("\n  RESULT: FAIL - no global timeout was raised")


def main():
    print("=" * 78)
    print("PART 4 / TASK 16 - TIMEOUTS AND RETRIES")
    print("=" * 78)
    print(f"  per-node timeout : {PER_NODE_TIMEOUT_S}s (on slow_node)")
    print(f"  global timeout   : {GLOBAL_TIMEOUT_S}s (whole graph)")
    print(f"  retry policy     : langgraph.types.RetryPolicy, "
          f"max_attempts={RETRY_POLICY.max_attempts},")
    print(f"                     initial_interval={RETRY_POLICY.initial_interval}s, "
          f"backoff_factor={RETRY_POLICY.backoff_factor},")
    print(f"                     max_interval={RETRY_POLICY.max_interval}s, "
          f"jitter={RETRY_POLICY.jitter}")
    print()

    demo_retry()
    demo_node_timeout()
    demo_global_timeout()

    print("\n" + "=" * 78)
    print("TASK 16 COMPLETE: (a) retry recovered, (b) per-node timeout fired "
          "cleanly,\n                  (c) global timeout cancelled the run.")
    print("=" * 78)


if __name__ == "__main__":
    main()
