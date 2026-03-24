#!/usr/bin/env python3
"""
register_agent.py — Manual agent registration CLI

Use this to pre-provision agents without them self-registering via the MCP tool.
Useful when AUTO_REGISTER is disabled or when an admin wants to control identity creation.

Usage:
  python register_agent.py --agent-id stewie --team-id innovationteam
"""

import argparse
import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "memory.db"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def register(agent_id: str, team_id: str) -> None:
    if not DB_PATH.exists():
        print(f"Error: database not found at {DB_PATH}")
        print("Start memory_server.py at least once to initialize the database.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    existing = conn.execute(
        "SELECT agent_id FROM agents WHERE agent_id = ? AND team_id = ?",
        (agent_id, team_id)
    ).fetchone()

    if existing:
        print(f"Agent '{agent_id}' on team '{team_id}' is already registered.")
        conn.close()
        return

    new_key = secrets.token_hex(32)
    conn.execute(
        "INSERT INTO agents (agent_id, team_id, api_key_hash, created_at) VALUES (?, ?, ?, ?)",
        (agent_id, team_id, hash_key(new_key), datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()

    print(f"Registered: agent_id='{agent_id}', team_id='{team_id}'")
    print(f"API Key:    {new_key}")
    print("Store this key — it will not be shown again.")


def main():
    parser = argparse.ArgumentParser(
        description="Register an agent with the Memory MCP Server."
    )
    parser.add_argument("--agent-id", required=True, help="Agent identifier (e.g. 'stewie')")
    parser.add_argument("--team-id",  required=True, help="Team identifier (e.g. 'innovationteam')")
    args = parser.parse_args()
    register(args.agent_id, args.team_id)


if __name__ == "__main__":
    main()
