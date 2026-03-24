# Memory MCP Server — Setup Guide

A locally-running MCP server that gives AI agents persistent, semantically-retrieved memory.
Agents call tools instead of loading markdown files. Only relevant memories come back.

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com) running locally with `nomic-embed-text` pulled

---

## Installation

**1. Install dependencies**

```bash
cd /path/to/memory-mcp-server
uv venv
uv pip install "mcp[cli]>=1.0.0" "numpy>=1.26.0"
```

**2. Pull the embedding model (if not already done)**

```bash
ollama pull nomic-embed-text
```

**3. Register the MCP server with Claude Code (global — available in all sessions)**

```bash
claude mcp add memory-mcp-server -s user -- \
  /path/to/memory-mcp-server/.venv/bin/python3 \
  /path/to/memory-mcp-server/memory_server.py
```

The database (`memory.db`) is created automatically on first run.

---

## Agent Registration

Each agent must register once to receive an API key.

### Option A — Auto-registration via MCP tool (default, recommended)

On first use, call `memory_register` with your project directory path:

```
memory_register(agent_id="stewie", team_id="innovationteam", env_path="/home/user/git/myteam")
```

The server registers the agent, writes credentials to `<env_path>/.env`, and confirms.
**From that point on, no API key is needed** — all tools load it automatically from `.env`.

The `.env` file will contain:
```
MEMORY_API_KEY=<your_key>
MEMORY_AGENT_ID=stewie
MEMORY_TEAM_ID=innovationteam
```

### Option B — Manual registration via CLI (admin-controlled)

Use this when `AUTO_REGISTER = False` or when pre-provisioning agents:

```bash
.venv/bin/python3 register_agent.py --agent-id stewie --team-id innovationteam
```

---

## Configuration

Edit these constants at the top of `memory_server.py`:

| Setting              | Default | Description |
|----------------------|---------|-------------|
| `AUTO_REGISTER`      | `True`  | Allow agents to self-register via `memory_register` tool |
| `CONFLICT_THRESHOLD` | `0.92`  | Cosine similarity above this triggers a conflict flag |
| `CONFLICT_TTL_DAYS`  | `30`    | Days before superseded memory versions are pruned |
| `EMBED_MODEL`        | `nomic-embed-text` | Ollama embedding model |

To disable auto-registration (admin-only mode), set `AUTO_REGISTER = False` and use `register_agent.py`.

---

## Tools Reference

| Tool              | Description |
|-------------------|-------------|
| `memory_register` | Register a new agent, receive an API key |
| `memory_store`    | Save a memory (agent or team scope) |
| `memory_retrieve` | Fetch semantically relevant memories for a query |
| `memory_update`   | Update a memory you created (clears conflicts) |
| `memory_delete`   | Delete a memory you created |
| `memory_list`     | List memories, filtered by scope or tags |

---

## Updating Your CLAUDE.md

Replace your flat markdown memory instructions with MCP tool calls:

**Before:**
```markdown
*At the start of every session, read `stewie_memory.md` to restore context.*
*When the supervisor shares something important, update `stewie_memory.md`.*
```

**After:**
```markdown
*At the start of every session, call `memory_retrieve` to restore relevant context — no API key needed, credentials load automatically from .env.*
*When the supervisor shares something important, decisions are made, or tasks are completed — call `memory_store` immediately. Don't wait.*
*Use scope='agent' for private memories, scope='team' for context the whole team should share.*
*If no .env exists yet, call memory_register with agent_id, team_id, and env_path to set up credentials once.*
```

---

## Security Notes

- API keys are hashed (SHA-256) at rest — plaintext keys are never stored.
- Scoping is enforced server-side from the API key — agents cannot impersonate each other.
- `AUTO_REGISTER = True` means any process that can reach the server can create an identity.
  For a local-only setup this is fine. If the server is ever network-accessible, set `AUTO_REGISTER = False`
  and use manual registration only.

---

## Swapping the Embedding Backend

All embedding calls go through one function in `memory_server.py`:

```python
def get_embedding(text: str) -> list[float]:
    ...
```

To switch from Ollama to `sentence-transformers` or any other backend,
edit only this function. Nothing else in the server needs to change.
