import asyncio
import json
import aiosqlite

# Define file paths
JSON_FILE_PATH = "metadata.json"
DB_FILE_PATH = "search_index_async.db"


async def setup_database_and_load_json(json_path, db_path):
    # Connect asynchronously
    async with aiosqlite.connect(db_path) as db:
        # 1. Create the main data table
        await db.execute(
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

        # 2. Create the FTS5 virtual table pointing to the main table
        await db.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                text,
                content='documents',
                content_rowid='rowid'
            )
        """
        )

        # Create trigger to automatically sync inserts into FTS index
        await db.execute(
            """
            CREATE TRIGGER IF NOT EXISTS t_documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, text) VALUES (new.rowid, new.text);
            END;
        """
        )
        await db.commit()

        # 3. Read JSON data (standard synchronous read is fine for initialization)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        insert_query = """
            INSERT OR IGNORE INTO documents (uri, id, src, ts, type, latlon, loc, ppt, caption, text)
            VALUES (:uri, :id, :src, :ts, :type, :latlon, :loc, :ppt, :caption, :text)
        """

        # Batch insert asynchronously
        await db.executemany(insert_query, data)
        await db.commit()
        print(
            f"Successfully loaded {len(data)} records and populated FTS5 index."
        )


async def search_documents(query_string, db_path):
    async with aiosqlite.connect(db_path) as db:
        # Notice bm25(documents_fts) uses the exact table name
        search_query = """
            SELECT d.id, d.uri, d.caption, d.text, bm25(documents_fts) as rank
            FROM documents_fts df
            JOIN documents d ON df.rowid = d.rowid
            WHERE documents_fts MATCH ?
            ORDER BY rank ASC
            LIMIT 10;
        """

        # Execute and fetch results asynchronously
        async with db.execute(search_query, (query_string,)) as cursor:
            results = await cursor.fetchall()

            if not results:
                print(f"No results found for: '{query_string}'")
                return

            for row in results:
                print(
                    f"ID: {row[0]} | Rank: {row[4]:.4f} | Caption: {row[2]} | Text: {row[3]}"
                )


async def main():
    # Run the setup and database insertion
    await setup_database_and_load_json(JSON_FILE_PATH, DB_FILE_PATH)

    # Perform an async test search
    print("\n--- Running Asynchronous Search ---")
    await search_documents("Esha and Anjali are dressed in traditional Indian attire", DB_FILE_PATH)


# Execute the asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())
