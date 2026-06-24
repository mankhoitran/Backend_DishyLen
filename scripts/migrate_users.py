import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'food_agent.db')

def migrate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if hashed_password already exists
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if "hashed_password" in columns:
        print("Migration already applied.")
        return

    print("Starting migration...")
    
    # Create new table
    cursor.execute("""
        CREATE TABLE users_new (
            id INTEGER NOT NULL PRIMARY KEY, 
            google_sub VARCHAR(255), 
            email VARCHAR(255) NOT NULL, 
            hashed_password VARCHAR(255),
            name VARCHAR(255) NOT NULL, 
            picture_url TEXT NOT NULL, 
            allergies TEXT NOT NULL, 
            created_at DATETIME NOT NULL
        )
    """)
    
    # Copy data
    cursor.execute("""
        INSERT INTO users_new (id, google_sub, email, name, picture_url, allergies, created_at)
        SELECT id, google_sub, email, name, picture_url, allergies, created_at FROM users
    """)
    
    # Drop old table and rename new table
    cursor.execute("DROP TABLE users")
    cursor.execute("ALTER TABLE users_new RENAME TO users")
    
    # Recreate indices
    cursor.execute("CREATE UNIQUE INDEX ix_users_email ON users (email)")
    cursor.execute("CREATE UNIQUE INDEX ix_users_google_sub ON users (google_sub)")
    cursor.execute("CREATE INDEX ix_users_id ON users (id)")
    
    conn.commit()
    conn.close()
    print("Migration completed successfully.")

if __name__ == "__main__":
    migrate()
