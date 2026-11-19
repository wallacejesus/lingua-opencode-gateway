from typing import Any

from mcp.server.fastmcp import FastMCP


def create_mcp(metering, exact_callback):
    server = FastMCP("Lingua Metering")

    @server.tool()
    async def session_report(session: str | None = None) -> dict[str, Any]:
        """Report token and cost totals for a session. Omit session to use the most recent session."""
        if not session:
            latest = metering.latest()
            session = latest.get("session") if latest else None
        report = metering.report(session)
        report["resolved_session"] = session
        return report

    @server.tool()
    async def context_report(session: str | None = None) -> dict[str, Any]:
        """Report the latest request's original estimated context and billed usage."""
        if not session:
            latest = metering.latest()
            session = latest.get("session") if latest else None
        records = metering._records_for(session)
        if not records:
            return {"session": session, "requests": 0}
        result = dict(records[-1])
        result.pop("original_messages", None)
        result.pop("compressed_messages", None)
        return result

    @server.tool()
    async def compression_stats() -> dict[str, Any]:
        """Report cumulative compression, token, and cost totals."""
        return metering.stats()

    @server.tool()
    async def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
        """List recent sessions and their token/cost savings."""
        return metering.sessions(max(1, min(limit, 100)))

    @server.tool()
    async def verify_exact(session: str | None = None, text: str | None = None) -> dict[str, Any]:
        """Explicitly measure exact original tokens. This makes one uncompressed upstream call and may incur cost."""
        return await exact_callback(session=session, text=text)

    return server
