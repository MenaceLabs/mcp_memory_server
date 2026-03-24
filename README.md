# MCP Memory Server

A locally-running MCP server that gives AI agents persistent, semantically-retrieved memory across sessions.

Instead of loading flat markdown files into context, agents call tools. Only relevant memories come back — not everything ever stored.

---

## How It Works

Memories are stored in a local SQLite database with vector embeddings generated via Ollama. On retrieval, the agent sends a plain-text query and gets back the most semantically relevant memories ranked by cosine similarity.

Each memory is scoped to an agent (private) or a team (shared). API key authentication ensures agents can't access each other's private memories.

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com) running locally with `nomic-embed-text` pulled

```bash
ollama pull nomic-embed-text
```

---

## Installation

```bash
git clone https://github.com/MenaceLabs/mcp_memory_server.git
cd mcp_memory_server

uv venv
uv pip install "mcp[cli]>=1.0.0" "numpy>=1.26.0"
```

**Register with Claude Code (global — available in all sessions):**

```bash
claude mcp add memory-mcp-server -s user -- \
  /path/to/mcp_memory_server/.venv/bin/python3 \
  /path/to/mcp_memory_server/memory_server.py
```

---

## Agent Registration

### Auto-registration (recommended)

On first use, call `memory_register` and pass your project directory path. Credentials are written to `.env` automatically — no manual key handling ever again.

```
memory_register(agent_id="myagent", team_id="myteam", env_path="/path/to/project")
```

### Manual registration (admin-controlled)

Set `AUTO_REGISTER = False` in `memory_server.py` and use the CLI:

```bash
.venv/bin/python3 register_agent.py --agent-id myagent --team-id myteam
```

---

## Tools

| Tool | Description |
|------|-------------|
| `memory_register` | Register an agent, receive an API key, optionally auto-save to `.env` |
| `memory_store` | Save a memory (agent or team scope) |
| `memory_retrieve` | Fetch semantically relevant memories for a plain-text query |
| `memory_update` | Update a memory you created |
| `memory_delete` | Delete a memory you created |
| `memory_list` | List memories filtered by scope or tags |

All tools except `memory_register` will load credentials automatically from `.env` or the `MEMORY_API_KEY` environment variable if `api_key` is not explicitly passed.

---

## Configuration

Edit the constants at the top of `memory_server.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `AUTO_REGISTER` | `True` | Allow agents to self-register via tool |
| `CONFLICT_THRESHOLD` | `0.92` | Cosine similarity above this flags a conflict |
| `CONFLICT_TTL_DAYS` | `30` | Days before superseded memory versions are pruned |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |

---

## Swapping the Embedding Backend

All embedding calls go through one function:

```python
def get_embedding(text: str) -> list[float]:
    ...
```

Edit only this function to switch from Ollama to `sentence-transformers` or any other backend. Nothing else in the server needs to change.

---

## Security

- API keys are hashed (SHA-256) at rest — plaintext keys are never stored in the database
- Identity is resolved server-side from the key — agents cannot spoof each other
- `memory.db` and `.env` are excluded from git via `.gitignore`
- With `AUTO_REGISTER = True`, any process that can reach the server can create an identity — suitable for local use only. Set to `False` for tighter control.

---

## Full Setup Guide

See [SETUP.md](SETUP.md) for complete setup instructions including CLAUDE.md integration patterns.
