import sqlite3
import uuid
from datetime import datetime, timedelta
import os


class SQLiteChatSessionStore:
    """Very much naive, local chat session storage based on SQLite."""

    def __init__(self, db_path="data", session_timeout=3600 * 24):
        self.conn = sqlite3.connect(
            os.path.join(db_path, "sessions.db"), check_same_thread=False
        )
        self.session_timeout = session_timeout
        self.create_tables()

    def create_tables(self):
        # Create sessions table
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    expires_at DATETIME NOT NULL
                )
            """
            )
            # Create messages table
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """
            )

    # Session management methods
    def create_session(self, user_id):
        session_id = str(uuid.uuid4())
        created_at = datetime.now()
        expires_at = created_at + timedelta(seconds=self.session_timeout)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO sessions (session_id, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
            """,
                (session_id, user_id, created_at, expires_at),
            )
        return session_id

    def get_session(self, session_id):
        cursor = self.conn.execute(
            """
            SELECT session_id, user_id, created_at, expires_at
            FROM sessions WHERE session_id = ?
        """,
            (session_id,),
        )
        row = cursor.fetchone()
        if row:
            session = {
                "session_id": row[0],
                "user_id": row[1],
                "created_at": row[2],
                "expires_at": row[3],
            }
            # Check if expired
            if datetime.now() > datetime.fromisoformat(session["expires_at"]):
                self.delete_session(session_id)
                return None
            return session
        return None

    def delete_session(self, session_id):
        with self.conn:
            self.conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            self.conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )

    def clean_expired_sessions(self):
        with self.conn:
            self.conn.execute(
                "DELETE FROM sessions WHERE expires_at < ?", (datetime.now(),)
            )
            self.conn.execute(
                """
                DELETE FROM messages
                WHERE session_id NOT IN (SELECT session_id FROM sessions)
            """
            )

    def get_active_session_by_user(self, user_id):
        cursor = self.conn.execute(
            """
            SELECT session_id
            FROM sessions
            WHERE user_id = ? AND expires_at > ?
        """,
            (user_id, datetime.now()),
        )
        row = cursor.fetchone()
        if row:
            return row[0]  # Return session ID
        return None  # No active session found

    # Message management methods
    def add_message(self, session_id, sender, content):
        if sender not in ["user", "assistant"]:
            raise ValueError("Sender must be 'user' or 'assistant'")
        timestamp = datetime.now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO messages (session_id, sender, content, timestamp)
                VALUES (?, ?, ?, ?)
            """,
                (session_id, sender, content, timestamp),
            )

    def get_messages(self, session_id):
        cursor = self.conn.execute(
            """
            SELECT sender, content, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """,
            (session_id,),
        )
        messages = [
            {"sender": row[0], "content": row[1], "timestamp": row[2]}
            for row in cursor.fetchall()
        ]
        return messages


if __name__ == "__main__":
    # Example usage
    store = SQLiteChatSessionStore()

    # Create a session
    session_id = store.create_session(user_id="user123")
    print(f"Session created: {session_id}")

    # Add messages to the session
    store.add_message(session_id, sender="user", content="Hello, assistant!")
    store.add_message(
        session_id, sender="assistant", content="Hi there! How can I help you?"
    )
    store.add_message(session_id, sender="user", content="Can you tell me a joke?")
    store.add_message(
        session_id,
        sender="assistant",
        content="Why don't scientists trust atoms? Because they make up everything!",
    )

    # Retrieve messages
    messages = store.get_messages(session_id)
    for msg in messages:
        print(f"[{msg['timestamp']}] {msg['sender']}: {msg['content']}")
