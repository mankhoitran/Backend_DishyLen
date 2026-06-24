import sqlite3
import os

def migrate():
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "food_agent.db"))
    print(f"Migrating database at: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Create a new table without google_sub and with is_guest
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email VARCHAR(255) UNIQUE NOT NULL,
            hashed_password VARCHAR(255),
            is_guest BOOLEAN NOT NULL DEFAULT 0,
            name VARCHAR(255) NOT NULL,
            picture_url TEXT NOT NULL,
            allergies TEXT NOT NULL,
            created_at DATETIME NOT NULL
        )
        """)

        # Copy data from old table to new table
        # We assume users table already has email, hashed_password, name, picture_url, allergies, created_at
        cursor.execute("""
        INSERT INTO users_new (id, email, hashed_password, is_guest, name, picture_url, allergies, created_at)
        SELECT id, email, hashed_password, 0, name, picture_url, allergies, created_at FROM users
        """)

        # Drop old table
        cursor.execute("DROP TABLE users")

        # Rename new table
        cursor.execute("ALTER TABLE users_new RENAME TO users")

        # Recreate indexes
        cursor.execute("CREATE UNIQUE INDEX ix_users_email ON users (email)")
        cursor.execute("CREATE INDEX ix_users_id ON users (id)")

        conn.commit()
        print("Migration successful: removed google_sub, added is_guest.")

    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
