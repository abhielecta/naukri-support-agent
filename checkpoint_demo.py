"""
checkpoint_demo.py - Part 4 / Task 15

SQLite-backed checkpointing for the LangGraph agent, using the separate
`langgraph-checkpoint-sqlite` package (SqliteSaver), keyed by thread id and
persisted to checkpoints.sqlite.

The demonstration is deliberately split across TWO SEPARATE OS PROCESSES,
because that is the strongest way to prove that the completed nodes' results
came from the checkpoint rather than from live memory:

    phase1 (process A)  runs the graph with interrupt_before=["rag_answer"],
                        so guard_input and route_intent execute and the run
                        stops before the remaining nodes.

    phase2 (process B)  a brand-new interpreter. Its in-process execution
                        counter starts empty. It resumes the SAME thread id
                        with invoke(None, config) and completes the run. The
                        counter then shows that ONLY rag_answer and finalize
                        ever executed in process B - guard_input and
                        route_intent did NOT run again, yet their results are
                        present in the final state, having been loaded from
                        checkpoints.sqlite.

Run
    python checkpoint_demo.py           # orchestrates both phases as subprocesses
    python checkpoint_demo.py phase1    # run phase 1 only
    python checkpoint_demo.py phase2    # run phase 2 only (same thread id)
"""

import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

import agent
from agent import AgentState, after_guard, after_route
from config import CHECKPOINT_DB

THREAD_ID = "naukri-thread-checkpoint-demo"
QUERY = "What is the offer negotiation policy on Naukri.com?"

#: names of the nodes that actually executed IN THIS PROCESS
EXECUTED_HERE = []


def _instrument(name, fn):
    """Wrap a node so every real execution in this process is visible."""

    def wrapped(state):
        EXECUTED_HERE.append(name)
        print(f"      >>> NODE ACTUALLY EXECUTED IN THIS PROCESS: {name}")
        return fn(state)

    wrapped.__name__ = name
    return wrapped


def build_checkpointed_graph(checkpointer, interrupt_before=None):
    """Same 5-node shape as agent.build_graph, with instrumented nodes."""
    g = StateGraph(AgentState)
    g.add_node("guard_input", _instrument("guard_input", agent.guard_input))
    g.add_node("route_intent", _instrument("route_intent", agent.route_intent))
    g.add_node("rag_answer", _instrument("rag_answer", agent.rag_answer))
    g.add_node("record_lookup", _instrument("record_lookup", agent.record_lookup))
    g.add_node("finalize", _instrument("finalize", agent.finalize))

    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", after_guard,
                            {"blocked": "finalize", "continue": "route_intent"})
    g.add_conditional_edges("route_intent", after_route,
                            {"rag": "rag_answer", "record_lookup": "record_lookup"})
    g.add_edge("rag_answer", "finalize")
    g.add_edge("record_lookup", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def _checkpoint_rows():
    """Raw row count in checkpoints.sqlite for this thread - proof of persistence."""
    if not Path(CHECKPOINT_DB).exists():
        return None
    con = sqlite3.connect(str(CHECKPOINT_DB))
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (THREAD_ID,)
        ).fetchone()[0]
        return n
    except sqlite3.Error:
        return None
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# phase 1 - run part of the graph, then stop
# --------------------------------------------------------------------------- #

def phase1():
    print("=" * 78)
    print("PHASE 1 (process A) - RUN, THEN DELIBERATELY STOP BEFORE THE REST")
    print("=" * 78)
    print(f"checkpoint db : {CHECKPOINT_DB}")
    print(f"thread id     : {THREAD_ID}")
    print(f"query         : {QUERY}")
    print("interrupt_before=['rag_answer'] - the run must halt after "
          "route_intent\n")

    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        graph = build_checkpointed_graph(saver, interrupt_before=["rag_answer"])
        config = {"configurable": {"thread_id": THREAD_ID}}

        state = {
            "query": QUERY,
            "conversation_id": "",
            "trace_id": f"trace-ckpt-{uuid.uuid4().hex[:8]}",
            "turn_index": 1,
            "executed_nodes": [],
        }

        print("  invoking...")
        graph.invoke(state, config=config)

        snap = graph.get_state(config)
        print("\n--- state after the interruption ---")
        print(f"  nodes executed in THIS process : {EXECUTED_HERE}")
        print(f"  state['executed_nodes']        : "
              f"{snap.values.get('executed_nodes')}")
        print(f"  next node(s) still to run      : {snap.next}")
        print(f"  route decided                  : {snap.values.get('route')}")
        print(f"  route reason                   : {snap.values.get('route_reason')}")
        print(f"  masked_query persisted         : "
              f"{snap.values.get('masked_query')!r}")
        print(f"  answer so far                  : "
              f"{snap.values.get('answer')!r}  (empty - rag_answer has not run)")
        print(f"  checkpoint rows for this thread: {_checkpoint_rows()}")

        assert snap.next == ("rag_answer",), snap.next
        assert EXECUTED_HERE == ["guard_input", "route_intent"], EXECUTED_HERE
        print("\n  PHASE 1 OK: 2 of the 5 nodes ran, execution stopped before "
              "the rest,\n  and the partial state is persisted in "
              "checkpoints.sqlite.")


# --------------------------------------------------------------------------- #
# phase 2 - fresh process, resume the SAME thread id
# --------------------------------------------------------------------------- #

def phase2():
    print("=" * 78)
    print("PHASE 2 (process B - a brand new interpreter) - RESUME AND COMPLETE")
    print("=" * 78)
    print(f"thread id     : {THREAD_ID}   (the SAME id as phase 1)")
    print(f"this process's execution counter starts empty : {EXECUTED_HERE}")
    print()

    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        graph = build_checkpointed_graph(saver, interrupt_before=["rag_answer"])
        config = {"configurable": {"thread_id": THREAD_ID}}

        before = graph.get_state(config)
        print("--- state loaded from checkpoints.sqlite BEFORE resuming ---")
        print(f"  executed_nodes recovered : {before.values.get('executed_nodes')}")
        print(f"  route recovered          : {before.values.get('route')}")
        print(f"  masked_query recovered   : {before.values.get('masked_query')!r}")
        print(f"  next node(s) to run      : {before.next}")
        print("  (none of these were computed in this process - they were read "
              "from disk)\n")

        print("  resuming with invoke(None, config)...")
        result = graph.invoke(None, config=config)

        print("\n--- after completion ---")
        print(f"  nodes ACTUALLY executed in THIS process : {EXECUTED_HERE}")
        print(f"  full node trail in state               : "
              f"{result.get('executed_nodes')}")
        print(f"  route                                  : {result.get('route')}")
        print(f"  answer                                 : "
              f"{result['response']['answer'][:220]}")
        print(f"  schema valid                           : "
              f"{result.get('schema_valid')}")
        print(f"  checkpoint rows for this thread        : {_checkpoint_rows()}")

        # the proof
        print("\n" + "-" * 78)
        print("PROOF THAT COMPLETED NODES WERE NOT RE-EXECUTED")
        print("-" * 78)
        trail = result.get("executed_nodes") or []
        for node in ["guard_input", "route_intent", "rag_answer", "finalize"]:
            in_state = node in trail
            ran_here = node in EXECUTED_HERE
            if in_state and not ran_here:
                verdict = "LOADED FROM CHECKPOINT (not re-executed)"
            elif in_state and ran_here:
                verdict = "executed in this process (it was still pending)"
            else:
                verdict = "not part of this run"
            print(f"  {node:<14} in final state={str(in_state):<5} "
                  f"ran in process B={str(ran_here):<5}  {verdict}")

        assert "guard_input" not in EXECUTED_HERE
        assert "route_intent" not in EXECUTED_HERE
        assert EXECUTED_HERE == ["rag_answer", "finalize"], EXECUTED_HERE
        assert "guard_input" in trail and "route_intent" in trail

        print("\n  PHASE 2 OK: the run completed on the same thread id, and "
              "guard_input\n  and route_intent were NOT re-executed - their "
              "results came from the\n  SQLite checkpoint.")


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def main():
    here = Path(__file__).resolve().parent

    print("=" * 78)
    print("PART 4 / TASK 15 - SQLITE CHECKPOINTING ACROSS AN INTERRUPTION")
    print("=" * 78)
    print("checkpointer : langgraph.checkpoint.sqlite.SqliteSaver")
    print("              (from the separate `langgraph-checkpoint-sqlite` package)")
    print(f"database     : {CHECKPOINT_DB}")
    print(f"thread id    : {THREAD_ID}\n")

    # start from a clean thread so the demo is reproducible
    if Path(CHECKPOINT_DB).exists():
        con = sqlite3.connect(str(CHECKPOINT_DB))
        try:
            for tbl in ("checkpoints", "writes", "blobs"):
                try:
                    con.execute(f"DELETE FROM {tbl} WHERE thread_id = ?",
                                (THREAD_ID,))
                except sqlite3.Error:
                    pass
            con.commit()
        finally:
            con.close()
        print(f"cleared any previous checkpoints for thread {THREAD_ID}\n")

    for phase in ("phase1", "phase2"):
        print("\n" + "#" * 78)
        print(f"# launching {phase} as a SEPARATE OS PROCESS")
        print("#" * 78 + "\n")
        sys.stdout.flush()   # so the child's output lands in the right place
        r = subprocess.run([sys.executable, str(here / "checkpoint_demo.py"), phase],
                           cwd=str(here))
        if r.returncode != 0:
            print(f"\n{phase} FAILED with exit code {r.returncode}")
            return r.returncode

    print("\n" + "=" * 78)
    print("TASK 15 COMPLETE:")
    print("  (a) a run executed 2 of the 5 nodes                       [phase 1]")
    print("  (b) execution stopped before the remaining nodes ran      [phase 1]")
    print("  (c) the SAME thread id resumed and completed the run, with")
    print("      the already-completed nodes loaded from the checkpoint")
    print("      and NOT re-executed                                   [phase 2]")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "phase1":
        phase1()
    elif cmd == "phase2":
        phase2()
    else:
        raise SystemExit(main())
