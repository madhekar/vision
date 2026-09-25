import json
import sqlite3


def drop_table(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
    """
     DROP TABLE IF EXISTS documents_fts;
    """ 
    )

def setup_database_and_load_json(json_path, db_path):
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Create the main table to store all data fields
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            uri TEXT,
            id TEXT PRIMARY KEY,
            src TEXT,
            ts TEXT,
            type TEXT,
            latlon TEXT,
            loc TEXT,
            ppt TEXT,
            caption TEXT,
            text TEXT
        )
    """
    )

    # 2. Create the FTS5 virtual table (externally content-backed for efficiency)
    # Since only 'text' is used for searching, it's the only indexed column.
    cursor.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            text,
            content='documents',
            content_rowid='rowid'
        )
    """
    )

    # Create triggers to keep the FTS index automatically updated on inserts
    cursor.execute(
        """
        CREATE TRIGGER IF NOT EXISTS t_documents_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, text) VALUES (new.rowid, new.text);
        END;
    """
    )

    # 3. Read JSON data and insert into the database
    with open(json_path, "r", encoding="utf-8") as f:
        # Assumes JSON is a list of objects: [{"uri": "...", "id": "..."}, ...]
        # If it is JSON Lines (one JSON per line), use: data = [json.loads(line) for line in f]
        # data = json.load(f)
        data = [json.loads(line) for line in f]

    insert_query = """
        INSERT OR IGNORE INTO documents (uri, id, src, ts, type, latlon, loc, ppt, caption, text)
        VALUES (:uri, :id, :src, :ts, :type, :latlon, :loc, :ppt, :caption, :text)
    """

    # Batch insert for high performance
    cursor.executemany(insert_query, data)
    conn.commit()

    print(
        f"Successfully loaded {len(data)} records and populated the FTS5 index."
    )
    conn.close()


def search_documents(query_string, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # SQL query joining the FTS index with the main table to pull all fields
    search_query = """
        SELECT d.id, d.uri, d.caption, d.text, bm25(documents_fts) as rank
        FROM documents_fts df
        JOIN documents d ON df.rowid = d.rowid
        WHERE documents_fts MATCH ?
        ORDER BY rank ASC
        LIMIT 10;
    """

    cursor.execute(search_query, (query_string,))
    results = cursor.fetchall()
    #print(f"---> {results}")
    for row in results:
        print(f"--->ID: {row[0]} | Rank: {row[4]:.4f} | Caption: {row[2]} | Text: {row[3]}")

    conn.close()

if __name__=="__main__":

    # Define file paths
    JSON_FILE_PATH = "metadata.json"
    DB_FILE_PATH = "search_index.db"

    drop_table(db_path=DB_FILE_PATH)
    # Run the loader
    setup_database_and_load_json(JSON_FILE_PATH, DB_FILE_PATH)
    # Example usage:
    search_documents("Esha dressed in traditional Indian attire",DB_FILE_PATH)