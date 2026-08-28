"""
mcp_server.py - Part 4 / Task 14 (server side)

Wraps check_job_application_status as an MCP tool and serves it over fastmcp's
HTTP transport, which mounts at /mcp:

    http://127.0.0.1:8765/mcp        <- this is the client endpoint
    http://127.0.0.1:8765           <- NOT the endpoint; the bare root 404s

Run
    python mcp_server.py

Then, in a SEPARATE process, run the client:
    python mcp_client.py
"""

from typing import Any, Dict

from fastmcp import FastMCP

from tools import check_job_application_status as _lookup

HOST = "127.0.0.1"
PORT = 8765
MCP_PATH = "/mcp"
MCP_URL = f"http://{HOST}:{PORT}{MCP_PATH}"

mcp = FastMCP("naukri-support-tools")


@mcp.tool
def check_job_application_status(record_id: str) -> Dict[str, Any]:
    """Look up the status of a Naukri.com job application by its record id.

    Use this whenever a recruiter or candidate asks about one specific
    application rather than about hiring policy in general.

    Args:
        record_id: The application identifier, formatted like "NAU-1003".
            Valid ids in the current dataset run from NAU-1000 to NAU-1047.

    Returns:
        A dictionary describing the application. On success it contains
        found=True, the pipeline status (Applied, Screening, Interview
        Scheduled, Offered or Rejected), the candidate's expected_salary_inr,
        days_since_created, flagged_priority_review, a designed
        escalation_score in [0, 1], the escalation_threshold it is compared
        against, a boolean escalate, and a human-readable explanation of the
        score arithmetic. If the id does not exist, it contains found=False
        and an error message.
    """
    return _lookup(record_id)


if __name__ == "__main__":
    print(f"Starting MCP server on {MCP_URL}")
    print("tool exposed: check_job_application_status")
    mcp.run(transport="http", host=HOST, port=PORT)
