from mcp.server.fastmcp import FastMCP
import sqlite3
import os

# Database setup
DB_PATH = "sample.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # Create table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product TEXT,
        revenue INTEGER,
        sale_date TEXT
    )
    """)

    # Check if already seeded
    cur.execute("SELECT COUNT(*) FROM sales")
    count = cur.fetchone()[0]

    if count == 0:
        cur.executemany(
            "INSERT INTO sales (product, revenue, sale_date) VALUES (?, ?, ?)",
            [
                ("Laptop", 120000, "2025-01-01"),
                ("Phone", 80000, "2025-01-02"),
                ("Tablet", 45000, "2025-01-03"),
            ]
        )

    conn.commit()
    conn.close()


# Initialize DB once
init_db()

mcp = FastMCP("sql-db-tools")


@mcp.tool()
def list_tables() -> list[str]:
    """List all tables in the database"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    conn.close()
    return tables


@mcp.tool()
def describe_table(table_name: str) -> list[dict]:
    """Describe columns of a table: name, type, nullable, primary_key"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    columns = [
        {
            "name": col[1],
            "type": col[2],
            "nullable": not bool(col[3]),
            "primary_key": bool(col[5]),
        }
        for col in cur.fetchall()
    ]
    conn.close()
    return columns


@mcp.tool()
def run_select_query(query: str) -> list[dict]:
    """Run SAFE read-only SELECT queries only"""
    if not query.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query)
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


if __name__ == "__main__":
    # Run as stdio MCP server
    mcp.run()