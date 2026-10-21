"""
Async HTTP client for the retrosynthesis tool server.

``ToolClient`` handles round-robin routing across server instances, session
affinity, and connection pooling.
"""
import asyncio
import uuid

import aiohttp


class BaseClient:
    """Shared routing logic for the tool server client."""

    def __init__(self, base_url, base_port, num_instances, timeout):
        self.base_url = base_url
        self.base_port = base_port
        self.num_instances = num_instances
        self.timeout = timeout
        self.rr_counter = 0
        self.session_to_port = {}

    def get_url(self, session_id=None):
        """Round-robin for new sessions, hash-routed for existing."""
        if self.num_instances <= 1:
            return f"{self.base_url}:{self.base_port}"
        if session_id is not None and session_id in self.session_to_port:
            return f"{self.base_url}:{self.session_to_port[session_id]}"
        port = self.base_port + (self.rr_counter % self.num_instances)
        self.rr_counter += 1
        if session_id is not None:
            self.session_to_port[session_id] = port
        return f"{self.base_url}:{port}"


class ToolClient(BaseClient):
    """Async HTTP client for the retrosynthesis tool server.

    Features:
    - Round-robin assignment of new sessions across server instances
    - Session affinity (subsequent calls route to the same instance)
    - Persistent ``aiohttp.ClientSession`` with connection pooling
    """

    def __init__(
        self,
        base_url="http://127.0.0.1",
        base_port=8100,
        num_instances=1,
        timeout=1800,
    ):
        super().__init__(base_url, base_port, num_instances, timeout)
        self.session = None

    def get_aiohttp_session(self):
        if self.session is None or self.session.closed:
            # keepalive_timeout=0 disables keepalive reuse, avoiding
            # ServerDisconnectedError from stale pooled connections.
            # Between tool calls, connections idle during LLM generation
            # (seconds to minutes), exceeding the server's keepalive.
            connector = aiohttp.TCPConnector(keepalive_timeout=0)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        return self.session

    async def post(self, url, payload, retries=3):
        session = self.get_aiohttp_session()
        for attempt in range(retries):
            try:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except aiohttp.ClientConnectionError:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))

    async def create_session(
        self,
        target,
        predict_templates_top_k=20,
        session_id=None,
    ):
        """Create a tool session on the server.

        Returns the server-assigned session_id.
        """
        session_id = session_id or uuid.uuid4().hex[:12]
        url = self.get_url(session_id)
        result = await self.post(
            f"{url}/create_session",
            {
                "target": target,
                "session_id": session_id,
                "predict_templates_top_k": predict_templates_top_k,
            },
        )
        return result["session_id"]

    async def invoke_tool(self, session_id, name, args):
        """Invoke a single tool on the server. Returns the result dict."""
        url = self.get_url(session_id)
        return await self.post(
            f"{url}/invoke_tool",
            {"session_id": session_id, "tool_name": name, "arguments": args},
        )

    async def delete_session(self, session_id):
        """Delete a session. Returns ``{graph_state, tool_calls_log, route_steps}``."""
        url = self.get_url(session_id)
        result = await self.post(f"{url}/delete_session", {"session_id": session_id})
        self.session_to_port.pop(session_id, None)
        return result

    async def get_session_log(self, session_id):
        """Read the session's tool_calls_log without deleting it."""
        url = self.get_url(session_id)
        return await self.post(f"{url}/get_session_log", {"session_id": session_id})

    async def get_tool_schemas(self):
        """Fetch precomputed tool schemas from the server."""
        url = self.get_url()
        session = self.get_aiohttp_session()
        async with session.get(f"{url}/tool_schemas") as resp:
            resp.raise_for_status()
            return await resp.json()

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

