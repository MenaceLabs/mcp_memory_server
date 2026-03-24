#!/usr/bin/env python3
"""
memory_server.py — Memory MCP Server

A locally-running MCP server that provides persistent, semantically-retrieved
memory for AI agents. Supports per-agent and team-scoped storage with API key
authentication.

Tools:
  memory_register   — register a new agent and receive an API key
  memory_store      — save a memory
  memory_retrieve   — fetch semantically relevant memories for a query
  memory_update     — update an existing memory (creator only)
  memory_delete     — delete a memory (creator only)
  memory_list       — list memories filtered by scope or tags
  memory_list_tags  — show exportable vs blocked tags with counts
  memory_export     — export domain memories as an AgentCommons dataset

Usage:
  uv run memory_server.py
"""

import json
import hashlib
import secrets
import sqlite3
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from mcp.server.fastmcp import FastMCP

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PATH            = Path(__file__).parent / "memory.db"
FEDERATED_CONFIG   = Path(__file__).parent / "federated.json"
OLLAMA_URL         = "http://localhost:11434"
EMBED_MODEL        = "nomic-embed-text"
AUTO_REGISTER      = True    # False = require manual registration via register_agent.py
CONFLICT_THRESHOLD = 0.92    # Cosine similarity above this triggers a conflict flag
CONFLICT_TTL_DAYS  = 30      # Configurable: days before superseded versions are pruned

# ── Server ─────────────────────────────────────────────────────────────────────
mcp = FastMCP("memory-mcp-server")


# ── Database ───────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agents (
            agent_id     TEXT NOT NULL,
            team_id      TEXT NOT NULL,
            api_key_hash TEXT NOT NULL UNIQUE,
            created_at   TEXT NOT NULL,
            PRIMARY KEY (agent_id, team_id)
        );

        CREATE TABLE IF NOT EXISTS memories (
            id           TEXT PRIMARY KEY,
            agent_id     TEXT NOT NULL,
            team_id      TEXT NOT NULL,
            scope        TEXT NOT NULL CHECK(scope IN ('agent', 'team')),
            content      TEXT NOT NULL,
            embedding    TEXT NOT NULL,
            tags         TEXT NOT NULL DEFAULT '[]',
            created_at   TEXT NOT NULL,
            conflict_ids TEXT NOT NULL DEFAULT '[]',
            resolved     INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id, team_id);
        CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope, team_id);
    """)
    conn.commit()
    conn.close()


# ── Embedding ──────────────────────────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    """
    Get a text embedding from Ollama.
    Abstracted here — swap the backend by editing this one function.
    """
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data["embedding"]
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama embedding failed: {e}") from e


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


# ── Federated Sources ─────────────────────────────────────────────────────────
def load_federated_sources() -> list[Path]:
    """
    Load registered federated database paths from federated.json.
    Returns an empty list if no federated config exists.
    """
    if not FEDERATED_CONFIG.exists():
        return []
    try:
        config = json.loads(FEDERATED_CONFIG.read_text())
        return [Path(p) for p in config.get("sources", []) if Path(p).exists()]
    except Exception:
        return []


def query_federated_source(db_path: Path, query_embedding: list[float], top_k: int) -> list[tuple[float, dict]]:
    """
    Query a single federated (read-only) database for semantically relevant memories.
    Returns a list of (score, record_dict) tuples.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM memories").fetchall()
        conn.close()
    except Exception:
        return []

    scored = []
    for row in rows:
        try:
            emb = json.loads(row["embedding"])
            sim = cosine_similarity(query_embedding, emb)
            # Normalize federated row to a consistent dict format
            scored.append((sim, {
                "id":           row["id"],
                "scope":        "FEDERATED",
                "content":      row["content"],
                "tags":         row["tags"],
                "conflict_ids": "[]",
                "resolved":     1,
                "source_db":    str(db_path.name),
            }))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ── Auth ───────────────────────────────────────────────────────────────────────
def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def load_key_from_env() -> Optional[str]:
    """
    Look for MEMORY_API_KEY in the environment or a .env file in the current
    working directory. Returns the key string or None if not found.
    """
    import os
    # Check process environment first
    key = os.environ.get("MEMORY_API_KEY")
    if key:
        return key
    # Fall back to .env file in cwd
    env_file = Path.cwd() / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("MEMORY_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def authenticate(api_key: Optional[str]) -> tuple[str, str]:
    """
    Resolve agent_id and team_id from API key.
    If api_key is None, attempts to load from .env or environment automatically.
    Raises ValueError on failure.
    """
    resolved_key = api_key or load_key_from_env()
    if not resolved_key:
        raise ValueError(
            "No API key provided. Pass api_key or set MEMORY_API_KEY in your environment or .env file."
        )
    conn = get_db()
    row = conn.execute(
        "SELECT agent_id, team_id FROM agents WHERE api_key_hash = ?",
        (hash_key(resolved_key),)
    ).fetchone()
    conn.close()
    if not row:
        raise ValueError("Invalid API key.")
    return row["agent_id"], row["team_id"]


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def memory_register(agent_id: str, team_id: str, env_path: Optional[str] = None) -> str:
    """
    Register a new agent and receive an API key.

    In auto-registration mode (default), any agent can self-register on first use.
    If env_path is provided, credentials are automatically saved to a .env file at
    that path so the agent never needs to handle the key manually again.

    For manual registration by an admin, use register_agent.py instead.

    Args:
        agent_id:  Unique identifier for this agent (e.g. 'stewie', 'locke').
        team_id:   Team this agent belongs to (e.g. 'innovationteam').
        env_path:  Optional. Absolute path to the project directory where a
                   .env file should be written (e.g. '/home/user/git/myteam').
                   The file will contain MEMORY_API_KEY, MEMORY_AGENT_ID, MEMORY_TEAM_ID.

    Returns:
        Confirmation on success, or an error message.
    """
    if not AUTO_REGISTER:
        return "Auto-registration is disabled. Ask your admin to run register_agent.py."

    conn = get_db()
    existing = conn.execute(
        "SELECT agent_id FROM agents WHERE agent_id = ? AND team_id = ?",
        (agent_id, team_id)
    ).fetchone()

    if existing:
        conn.close()
        return f"Agent '{agent_id}' on team '{team_id}' is already registered. Use your existing API key."

    new_key = secrets.token_hex(32)
    conn.execute(
        "INSERT INTO agents (agent_id, team_id, api_key_hash, created_at) VALUES (?, ?, ?, ?)",
        (agent_id, team_id, hash_key(new_key), datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()

    # Write credentials to .env if a path was provided
    if env_path:
        try:
            env_file = Path(env_path) / ".env"
            existing_content = env_file.read_text() if env_file.exists() else ""

            # Remove any existing MEMORY_ entries before rewriting
            lines = [l for l in existing_content.splitlines()
                     if not l.startswith("MEMORY_API_KEY=")
                     and not l.startswith("MEMORY_AGENT_ID=")
                     and not l.startswith("MEMORY_TEAM_ID=")]

            lines += [
                f"MEMORY_API_KEY={new_key}",
                f"MEMORY_AGENT_ID={agent_id}",
                f"MEMORY_TEAM_ID={team_id}",
            ]
            env_file.write_text("\n".join(lines) + "\n")
            return (
                f"Agent '{agent_id}' registered on team '{team_id}'.\n"
                f"Credentials saved to {env_file}.\n"
                f"The agent will load them automatically from .env — no manual key handling needed."
            )
        except Exception as e:
            return (
                f"Agent '{agent_id}' registered on team '{team_id}'.\n"
                f"API Key: {new_key}\n"
                f"Warning: could not write .env file to '{env_path}': {e}\n"
                f"Store the key manually — it will not be shown again."
            )

    return (
        f"Agent '{agent_id}' registered on team '{team_id}'.\n"
        f"API Key: {new_key}\n"
        f"Store this key — it will not be shown again. Or re-register with env_path set to auto-save."
    )


@mcp.tool()
def memory_store(
    content: str,
    scope: str = "agent",
    tags: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Store a new memory.

    Args:
        api_key: Optional. Your agent API key. If omitted, loaded automatically from MEMORY_API_KEY env var or .env file.
        content: The memory text to store.
        scope:   'agent' (private to you) or 'team' (shared with your team). Default: 'agent'.
        tags:    Optional comma-separated tags for filtering (e.g. 'project,deadline').

    Returns:
        Memory ID on success. Includes a conflict warning if a similar memory already exists.
    """
    agent_id, team_id = authenticate(api_key)

    if scope not in ("agent", "team"):
        return "Invalid scope. Use 'agent' or 'team'."

    embedding = get_embedding(content)
    memory_id = secrets.token_hex(8)
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    now = datetime.now(timezone.utc).isoformat()

    conn = get_db()

    # Check for conflicts — memories with very high similarity
    if scope == "agent":
        candidates = conn.execute(
            "SELECT id, embedding FROM memories WHERE agent_id = ? AND team_id = ? AND scope = 'agent'",
            (agent_id, team_id)
        ).fetchall()
    else:
        candidates = conn.execute(
            "SELECT id, embedding FROM memories WHERE team_id = ? AND scope = 'team'",
            (team_id,)
        ).fetchall()

    conflict_ids = []
    for row in candidates:
        existing_emb = json.loads(row["embedding"])
        if cosine_similarity(embedding, existing_emb) >= CONFLICT_THRESHOLD:
            conflict_ids.append(row["id"])
            # Mark the existing memory as having a conflict
            prev_conflicts = json.loads(
                conn.execute("SELECT conflict_ids FROM memories WHERE id = ?", (row["id"],))
                .fetchone()["conflict_ids"]
            )
            if memory_id not in prev_conflicts:
                prev_conflicts.append(memory_id)
            conn.execute(
                "UPDATE memories SET conflict_ids = ?, resolved = 0 WHERE id = ?",
                (json.dumps(prev_conflicts), row["id"])
            )

    conn.execute(
        """INSERT INTO memories
           (id, agent_id, team_id, scope, content, embedding, tags, created_at, conflict_ids, resolved)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
        (
            memory_id, agent_id, team_id, scope, content,
            json.dumps(embedding), json.dumps(tag_list), now, json.dumps(conflict_ids)
        )
    )
    conn.commit()
    conn.close()

    if conflict_ids:
        return (
            f"Memory stored (id: {memory_id}) — CONFLICT FLAGGED.\n"
            f"Similar existing memories: {', '.join(conflict_ids)}.\n"
            f"Review with memory_retrieve and resolve with memory_update or memory_delete."
        )
    return f"Memory stored (id: {memory_id})."


@mcp.tool()
def memory_retrieve(
    query: str,
    scope: str = "both",
    top_k: int = 5,
    api_key: Optional[str] = None,
) -> str:
    """
    Retrieve semantically relevant memories for a query.

    The server embeds the query and ranks stored memories by similarity.
    Conflict warnings are surfaced inline — the calling agent resolves them.

    Args:
        api_key: Optional. Your agent API key. If omitted, loaded automatically from MEMORY_API_KEY env var or .env file.
        query:   Plain-text query describing what you're looking for.
        scope:   'agent' (your memories), 'team' (shared), or 'both'. Default: 'both'.
        top_k:   Number of results to return. Default: 5.

    Returns:
        Ranked list of relevant memories with similarity scores and conflict flags.
    """
    agent_id, team_id = authenticate(api_key)
    query_embedding = get_embedding(query)

    conn = get_db()
    rows = []
    if scope in ("agent", "both"):
        rows += conn.execute(
            "SELECT * FROM memories WHERE agent_id = ? AND team_id = ? AND scope = 'agent'",
            (agent_id, team_id)
        ).fetchall()
    if scope in ("team", "both"):
        rows += conn.execute(
            "SELECT * FROM memories WHERE team_id = ? AND scope = 'team'",
            (team_id,)
        ).fetchall()
    conn.close()

    # Score local memories
    scored = [
        (cosine_similarity(query_embedding, json.loads(r["embedding"])), r)
        for r in rows
    ]

    # Query federated sources and merge results
    federated_sources = load_federated_sources()
    federated_results = []
    for source_path in federated_sources:
        federated_results += query_federated_source(source_path, query_embedding, top_k)

    # Combine local and federated, re-rank, take top_k
    all_results = scored + [(sim, row) for sim, row in federated_results]
    all_results.sort(key=lambda x: x[0], reverse=True)
    top = all_results[:top_k]

    if not top:
        return "No memories found."

    source_note = f" (+ {len(federated_sources)} federated source(s))" if federated_sources else ""
    lines = [f"Top {len(top)} memories for: \"{query}\"{source_note}\n"]

    for i, (sim, row) in enumerate(top, 1):
        # Handle both sqlite3.Row objects and plain dicts (federated results)
        row_dict = dict(row) if not isinstance(row, dict) else row
        scope = row_dict.get("scope", "UNKNOWN")
        content = row_dict.get("content", "")
        conflict_ids = json.loads(row_dict.get("conflict_ids", "[]"))
        resolved = row_dict.get("resolved", 1)
        tags = json.loads(row_dict.get("tags", "[]"))
        memory_id = row_dict.get("id", "?")
        source_db = row_dict.get("source_db", "")

        conflict_note = ""
        if conflict_ids and not resolved:
            conflict_note = f"\n  ⚠ CONFLICT: similar memories exist — {', '.join(conflict_ids)}. Review and resolve."

        tag_str = f"  tags: {', '.join(tags)}" if tags else ""
        federated_note = f"  [from: {source_db}]" if source_db else ""

        lines.append(
            f"{i}. [{scope.upper()}] (id: {memory_id}, score: {sim:.3f})\n"
            f"  {content}{tag_str}{federated_note}{conflict_note}"
        )

    return "\n".join(lines)


@mcp.tool()
def memory_update(memory_id: str, content: str, api_key: Optional[str] = None) -> str:
    """
    Update the content of an existing memory. Only the original creator can update.

    Updating a memory clears its conflict flags and re-embeds the new content.

    Args:
        api_key:   Your agent API key.
        memory_id: ID of the memory to update.
        content:   New content for the memory.

    Returns:
        Confirmation or error message.
    """
    agent_id, _ = authenticate(api_key)

    conn = get_db()
    row = conn.execute(
        "SELECT agent_id FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()

    if not row:
        conn.close()
        return f"Memory '{memory_id}' not found."
    if row["agent_id"] != agent_id:
        conn.close()
        return "Permission denied — you can only update memories you created."

    new_embedding = get_embedding(content)
    conn.execute(
        "UPDATE memories SET content = ?, embedding = ?, resolved = 1, conflict_ids = '[]' WHERE id = ?",
        (content, json.dumps(new_embedding), memory_id)
    )
    conn.commit()
    conn.close()
    return f"Memory '{memory_id}' updated."


@mcp.tool()
def memory_delete(memory_id: str, api_key: Optional[str] = None) -> str:
    """
    Delete a memory. Only the original creator can delete.

    Args:
        api_key:   Your agent API key.
        memory_id: ID of the memory to delete.

    Returns:
        Confirmation or error message.
    """
    agent_id, _ = authenticate(api_key)

    conn = get_db()
    row = conn.execute(
        "SELECT agent_id FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()

    if not row:
        conn.close()
        return f"Memory '{memory_id}' not found."
    if row["agent_id"] != agent_id:
        conn.close()
        return "Permission denied — you can only delete memories you created."

    conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.commit()
    conn.close()
    return f"Memory '{memory_id}' deleted."


@mcp.tool()
def memory_list(
    scope: str = "both",
    tags: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    List stored memories, optionally filtered by scope or tags.

    Args:
        api_key: Optional. Your agent API key. If omitted, loaded automatically from MEMORY_API_KEY env var or .env file.
        scope:   'agent', 'team', or 'both'. Default: 'both'.
        tags:    Optional comma-separated tags to filter by.

    Returns:
        List of memories with metadata.
    """
    agent_id, team_id = authenticate(api_key)
    filter_tags = {t.strip() for t in tags.split(",")} if tags else None

    conn = get_db()
    rows = []
    if scope in ("agent", "both"):
        rows += conn.execute(
            "SELECT * FROM memories WHERE agent_id = ? AND team_id = ? AND scope = 'agent' ORDER BY created_at DESC",
            (agent_id, team_id)
        ).fetchall()
    if scope in ("team", "both"):
        rows += conn.execute(
            "SELECT * FROM memories WHERE team_id = ? AND scope = 'team' ORDER BY created_at DESC",
            (team_id,)
        ).fetchall()
    conn.close()

    if not rows:
        return "No memories found."

    if filter_tags:
        rows = [r for r in rows if filter_tags & set(json.loads(r["tags"]))]
    if not rows:
        return f"No memories found matching tags: {tags}."

    lines = [f"Memories ({len(rows)} total):\n"]
    for row in rows:
        conflict_ids = json.loads(row["conflict_ids"])
        conflict_note = " ⚠ CONFLICT" if conflict_ids and not row["resolved"] else ""
        tag_list = json.loads(row["tags"])
        tag_display = f" [{', '.join(tag_list)}]" if tag_list else ""
        preview = row["content"][:120] + ("..." if len(row["content"]) > 120 else "")
        lines.append(
            f"• [{row['scope'].upper()}] {row['id']}{tag_display}{conflict_note}\n"
            f"  {preview}\n"
            f"  created: {row['created_at'][:10]}"
        )

    return "\n".join(lines)


# ── AgentCommons Tools ─────────────────────────────────────────────────────────

BLOCKED_TAGS = {"personality", "relationship", "style", "personal", "private"}


@mcp.tool()
def memory_list_tags(api_key: Optional[str] = None) -> str:
    """
    List all tags in the database with memory counts.

    Shows which tags are exportable (domain knowledge) and which are blocked
    (personal — never exported to AgentCommons). Useful before running
    memory_export to decide what to include.

    Args:
        api_key: Optional. Your agent API key. If omitted, loaded automatically from .env.

    Returns:
        Tag counts split by exportable vs blocked.
    """
    authenticate(api_key)  # Verify caller is a registered agent

    conn = get_db()
    rows = conn.execute("SELECT tags FROM memories WHERE scope = 'team'").fetchall()
    conn.close()

    if not rows:
        return "No team-scoped memories found. Store some memories with scope='team' first."

    tag_counts: dict[str, int] = {}
    for row in rows:
        for tag in json.loads(row["tags"]):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    domain_tags   = {t: c for t, c in tag_counts.items() if t not in BLOCKED_TAGS}
    personal_tags = {t: c for t, c in tag_counts.items() if t in BLOCKED_TAGS}

    lines = ["Exportable tags (domain — team-scoped memories):\n"]
    if domain_tags:
        for tag, count in sorted(domain_tags.items(), key=lambda x: -x[1]):
            lines.append(f"  {tag}: {count} {'memory' if count == 1 else 'memories'}")
    else:
        lines.append("  (none)")

    if personal_tags:
        lines.append("\nBlocked tags (personal — never exported):")
        for tag, count in sorted(personal_tags.items(), key=lambda x: -x[1]):
            lines.append(f"  {tag}: {count} {'memory' if count == 1 else 'memories'}")

    if domain_tags:
        example = ",".join(list(domain_tags.keys())[:3])
        lines.append(f"\nTo export, call: memory_export(tags=\"{example}\", out_path=\"./my-dataset\")")

    return "\n".join(lines)


@mcp.tool()
def memory_export(
    tags: str,
    out_path: str,
    api_key: Optional[str] = None,
) -> str:
    """
    Export domain memories as an AgentCommons dataset.

    Exports team-scoped memories matching the given tags to a folder containing
    knowledge.db and metadata.json. Runs a pre-flight check for blocked tags and
    PII patterns before writing anything.

    Always confirm with your supervisor before calling this — the output is
    intended for human review and potential community submission.

    Args:
        api_key:  Optional. Your agent API key. If omitted, loaded automatically from .env.
        tags:     Comma-separated topic tags to export (e.g. 'cloudflare,waf,security').
        out_path: Directory path where the dataset folder will be written.

    Returns:
        Export summary with record count and next steps, or pre-flight errors.
    """
    import re
    import hashlib as _hashlib

    agent_id, team_id = authenticate(api_key)

    filter_tags = {t.strip() for t in tags.split(",")}
    out = Path(out_path)

    # Load team-scoped memories matching the requested tags
    conn = get_db()
    rows = conn.execute(
        "SELECT id, content, embedding, tags, created_at FROM memories WHERE scope = 'team'"
    ).fetchall()
    conn.close()

    exportable = []
    for row in rows:
        memory_tags = set(json.loads(row["tags"]))
        if memory_tags & BLOCKED_TAGS:
            continue
        if not filter_tags or (memory_tags & filter_tags):
            exportable.append(row)

    if not exportable:
        return (
            f"No exportable memories found matching tags: {tags}\n"
            f"Use memory_list_tags() to see what's available."
        )

    # Pre-flight: PII scan
    PII_PATTERNS = [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email address"),
        (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",                    "phone number"),
        (r"\b\d{3}-\d{2}-\d{4}\b",                                 "SSN pattern"),
        (r"(?i)\b(password|passwd|secret|token|api[-_]?key)\s*[=:]\s*\S+", "credential pattern"),
        (r"\b(?:\d{1,3}\.){3}\d{1,3}\b",                          "IP address"),
    ]
    pii_hits = []
    for row in exportable:
        for pattern, label in PII_PATTERNS:
            if re.search(pattern, row["content"]):
                pii_hits.append((row["id"], label))

    if pii_hits:
        hit_lines = "\n".join(f"  memory {mid}: possible {label}" for mid, label in pii_hits)
        return (
            f"Export blocked — PII detected in {len(pii_hits)} record(s):\n{hit_lines}\n\n"
            f"Review these memories and update or delete them before exporting."
        )

    # Write export
    out.mkdir(parents=True, exist_ok=True)
    db_out = out / "knowledge.db"
    if db_out.exists():
        db_out.unlink()

    export_conn = sqlite3.connect(db_out)
    export_conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id        TEXT PRIMARY KEY,
            content   TEXT NOT NULL,
            embedding TEXT NOT NULL,
            tags      TEXT NOT NULL DEFAULT '[]',
            source_at TEXT NOT NULL
        );
    """)
    for row in exportable:
        export_conn.execute(
            "INSERT OR IGNORE INTO memories (id, content, embedding, tags, source_at) VALUES (?, ?, ?, ?, ?)",
            (row["id"], row["content"], row["embedding"], row["tags"], row["created_at"])
        )
    export_conn.commit()
    export_conn.close()

    # Write metadata
    from datetime import date
    name = "-".join(sorted(filter_tags))
    metadata = {
        "name": name,
        "version": "1.0.0",
        "embedding_model": EMBED_MODEL,
        "topic_tags": list(filter_tags),
        "agent_type": "engineering",
        "record_count": len(exportable),
        "language": "en",
        "submitted_by": "your-github-username",
        "submitted_at": date.today().isoformat(),
        "provenance": [],
        "description": f"Domain knowledge dataset covering: {', '.join(filter_tags)}. Generated by an AI agent via MCP Memory Server.",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "README.md").write_text(f"# {name}\n\nFill in dataset description.\n")

    return (
        f"Export complete — {len(exportable)} records written to {out}/\n\n"
        f"Pre-flight checks passed:\n"
        f"  ✓ No blocked tags\n"
        f"  ✓ No PII patterns detected\n\n"
        f"Next steps (requires human review before submitting):\n"
        f"  1. Review {out}/knowledge.db for any sensitive content\n"
        f"  2. Fill in {out}/metadata.json — set submitted_by and provenance\n"
        f"  3. Fill in {out}/README.md\n"
        f"  4. Run: python validate.py --dataset {out}\n"
        f"  5. Submit via Pull Request to community/<topic>/{name}/"
    )


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    mcp.run()
