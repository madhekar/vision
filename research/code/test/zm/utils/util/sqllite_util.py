import re
import json
import sqlite3
import pandas as pd


# A basic list of common English stop words
STOP_WORDS = {"your", "i","me","my","myself","we","our","ours","ourselves","you", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",}
#{"the", "is", "at", "which", "on", "and", "a", "an", "to", "in", "for", "with", "of"}

def clean_and_format_query(user_input: str) -> str:
    # 1. Lowercase and remove all non-alphanumeric/non-space characters
    clean_input = re.sub(r'[^\w\s]', '', user_input.lower())
    
    # 2. Tokenize and filter out common stop words
    words = [word for word in clean_input.split() if word not in STOP_WORDS]
    
    # 3. Format for FTS5 (joining words implies an 'AND' relationship)
    # Adding '*' turns them into prefix matches (e.g., "sql*" matches "sqlite")
    fts5_query = " ".join([f"{word}*" for word in words])
    
    return fts5_query

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
            content_rowid='rowid',
            tokenize="porter unicode61"
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
        #data = json.load(f)
        data = [json.loads(line) for line in f]
    # df = pd.read_json(json_path, lines=True)
    # df["uri"] = df["uri"].str.replace(
    #         "input-data/img",
    #         "final-data/img" #+ image_final_path,
    #    )

    insert_query = """
        INSERT OR IGNORE INTO documents (uri, id, src, ts, type, latlon, loc, ppt, caption, text)
        VALUES (:uri, :id, :src, :ts, :type, :latlon, :loc, :ppt, :caption, :text)
    """

    # Batch insert for high performance
    cursor.executemany(insert_query, data)
    conn.commit()

    print(f"Successfully loaded {len(data)} records and populated the FTS5 index.")
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
        LIMIT 5;
    """

    cursor.execute(search_query, (query_string,))
    results = cursor.fetchall()
    #print(f"---> {results}")
    for row in results:
        print(f"--->ID: {row[0]} | Rank: {row[4]:.4f} | Caption: {row[2]} | Text: {row[3]}")

    conn.close()

if __name__ == "__main__":
    # Define file paths
    JSON_FILE_PATH = "metadata.json"
    DB_FILE_PATH = "search_index.db"
    # Run the loader
    setup_database_and_load_json(JSON_FILE_PATH, DB_FILE_PATH)
    # Example usage:
    search_documents("Esha dressed in traditional Indian attire")