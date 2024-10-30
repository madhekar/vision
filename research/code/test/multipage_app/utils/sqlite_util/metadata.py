import os
import sqlite3 as sl
from datetime import datetime, timezone
import pandas as pd


class Metadata:
    def __init__(self, dbpath, dbname):
        self.con = sl.connect(os.path.join(dbpath, dbname))
        self.cur = self.con.cursor()

    def create_metadata_tbl_if_not_exists(self):
        self.cur.execute(
            "CREATE TABLE IF NOT EXISTS Metadata (name TEXT PRIMARY KEY, desc TEXT, lat NUMERIC, lon NUMERIC)"
        )
        self.con.commit()

    def insert_metadata(self, name, desc, lat, lon):
        self.cur.execute(
            "INSERT INTO Metadata VALUES(?, ?, ?, ?))", (name, desc, lat, lon)
        )
        self.con.commit()

    def bulk_insert(self, data):
        columns = ", ".join(data[0].keys())
        place_holders = ", ".join(["?"] * len(data[0]))
        insert_query = f"INSERT INTO Metadata ({columns}) VALUES ({place_holders})"
        print(insert_query)
        self.cur.executemany(insert_query, [tuple(row.values()) for row in data])
        self.con.commit()

    def read_metadata(self, name=None):
        if name:
            self.cur.execute(
                "SELECT name, desc, lat, lon FROM Metadata WHERE name=?", (name)
            )
            return self.cur.fetchone()
        else:
            self.cur.execute("SELECT * FROM Metadata")
            return self.cur.fetchall()

    def get_number_of_rows(self):
        self.cur.execute("SELECT COUNT(1) FROM Metadata")
        return self.cur.fetchall()

    def delete_metadata(self, name):
        if name:
            self.cur.execute("DELETE FROM Metadata WHERE name = ?", (name,))
        else:
            self.cur.execute("DELETE FROM Metadata")
        self.con.commit()

    def close(self):
        self.con.close()
