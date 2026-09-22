
import sqlite3

# 1. Connect to an in-memory database (or specify a file like 'search.db')
conn = sqlite3.connect(":memory:")
cursor = conn.cursor()

# 2. Create the main data storage table
cursor.execute("""
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    body TEXT
);
""")

# 3. Create the FTS5 virtual table for BM25 search
# We use 'unicode61' for case-insensitive matching and stripping punctuation.
cursor.execute("""
CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
    title,
    body,
    content='articles',
    content_rowid='id',
    tokenize='unicode61'
);
""")

# 4. Create Triggers to automatically keep the BM25 index in sync
# This ensures that whenever you insert/update/delete in 'articles', 
# 'articles_fts' updates instantly and automatically.
cursor.execute("""
CREATE TRIGGER IF NOT EXISTS articles_ai AFTER INSERT ON articles BEGIN
  INSERT INTO articles_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
END;
""")

cursor.execute("""
CREATE TRIGGER IF NOT EXISTS articles_ad AFTER DELETE ON articles BEGIN
  INSERT INTO articles_fts(articles_fts, rowid, title, body) VALUES('delete', old.id, old.title, old.body);
END;
""")

cursor.execute("""
CREATE TRIGGER IF NOT EXISTS articles_au AFTER UPDATE ON articles BEGIN
  INSERT INTO articles_fts(articles_fts, rowid, title, body) VALUES('delete', old.id, old.title, old.body);
  INSERT INTO articles_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
END;
""")

# 5. Insert sample data into the main table
# The triggers will automatically populate 'articles_fts'
sample_data = [
    ("SQLite Basics", "Learn how to use SQLite for fast local data storage."),
    ("Advanced Search in Python", "Python can leverage SQLite FTS5 for built-in BM25 full-text search."),
    ("Database Performance Optimization", "Indexes make relational database lookups incredibly fast and performant.")
]

cursor.executemany("INSERT INTO articles (title, body) VALUES (?, ?);", sample_data)
conn.commit()

# 6. Execute a BM25 Search Query
# We search for 'SQLite search'. 
# Note: bm25() requires ASC sorting because a LOWER score means HIGHER relevance in SQLite.
search_query = "SQLite search"

cursor.execute("""
    SELECT 
        id, 
        title, 
        body, 
        bm25(articles_fts) AS score
    FROM articles_fts
    WHERE articles_fts MATCH ?
    ORDER BY score ASC;
""", (search_query,))

results = cursor.fetchall()

# 7. Print the ranked results
print(f"Search Results for: '{search_query}' (Sorted by BM25 relevance)\n")
for row in results:
    doc_id, title, body, score = row
    print(f"ID: {doc_id} | Score: {score:.4f}")
    print(f"Title: {title}")
    print(f"Body:  {body}")
    print("-" * 50)

# Clean up
conn.close()

