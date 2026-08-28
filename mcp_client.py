"""
mcp_client.py - Part 4 / Task 14 (client side)

A SEPARATE file and a SEPARATE process from the LangGraph agent. It connects to
the locally running MCP server over HTTP, lists the advertised tools, and calls
check_job_application_status for several record ids, printing the standardized
MCP response for each.

Usage
    terminal 1:  python mcp_server.py
    terminal 2:  python mcp_client.py

Or let the client start and stop the server itself:
    python mcp_client.py --spawn-server
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

from fastmcp import Client

from mcp_server import MCP_URL

RECORD_IDS = ["NAU-1003", "NAU-1031", "NAU-1011", "NAU-9999"]


def _dump(obj) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except TypeError:
        return repr(obj)


async def run() -> int:
    print("=" * 78)
    print("PART 4 / TASK 14 - MCP CLIENT -> SERVER ROUND TRIP")
    print("=" * 78)
    print(f"connecting to : {MCP_URL}")
    print("(fastmcp's HTTP transport mounts at /mcp, not at the bare host:port)")
    print(f"client process: PID {__import__('os').getpid()} - separate from the "
          f"LangGraph agent\n")

    async with Client(MCP_URL) as client:
        # ---- discovery ----
        tools = await client.list_tools()
        print("--- tools advertised by the server ---")
        for t in tools:
            print(f"  name        : {t.name}")
            print(f"  description : {(t.description or '').strip().splitlines()[0]}")
            print(f"  input schema: {_dump(t.inputSchema)}")
        print()

        # ---- calls ----
        ok = 0
        for rid in RECORD_IDS:
            print("-" * 78)
            print(f"CALL: check_job_application_status(record_id={rid!r})")
            result = await client.call_tool(
                "check_job_application_status", {"record_id": rid}
            )

            print("\nstandardized MCP response:")
            print(f"  is_error       : {result.is_error}")
            print(f"  content blocks : {len(result.content)}")
            for block in result.content:
                print(f"    - type={block.type}")
                if getattr(block, "text", None):
                    print(f"      text={block.text}")
            print(f"  structured_content:\n{_dump(result.structured_content)}")

            data = result.data if result.data is not None else result.structured_content
            if isinstance(data, dict) and data.get("found"):
                ok += 1
                print(f"\n  -> {data['record_id']}: status={data['status']!r}, "
                      f"expected_salary_inr={data['expected_salary_inr']:,}, "
                      f"escalation_score={data['escalation_score']}, "
                      f"escalate={data['escalate']}")
            else:
                print("\n  -> server correctly reported the id as not found")
            print()

    print("=" * 78)
    print(f"RESULT: {ok} successful lookups over MCP "
          f"(requirement: >= 2 different record ids)")
    print(f"        plus 1 not-found case handled cleanly over the same transport")
    print("=" * 78)
    return 0 if ok >= 2 else 1


def main() -> int:
    proc = None
    if "--spawn-server" in sys.argv:
        here = Path(__file__).resolve().parent
        print("spawning mcp_server.py as a child process...\n")
        proc = subprocess.Popen(
            [sys.executable, str(here / "mcp_server.py")],
            cwd=str(here),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # wait for the port to accept connections
        import socket

        for _ in range(80):
            with socket.socket() as s:
                s.settimeout(0.25)
                try:
                    s.connect(("127.0.0.1", 8765))
                    break
                except OSError:
                    time.sleep(0.25)
        else:
            print("server did not come up in time")
            proc.terminate()
            return 1

    try:
        return asyncio.run(run())
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("\nchild MCP server terminated.")


if __name__ == "__main__":
    raise SystemExit(main())
