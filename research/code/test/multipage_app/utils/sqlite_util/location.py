import os
import sqlite3 as sl
from datetime import datetime, timezone
import pandas as pd


class Location:
    def __init__(self, dbpath, dbname):
        self.con = sl.connect(os.path.join(dbpath, dbname))
        self.cur = self.con.cursor()

    def create_location_tbl_if_not_exists(self):
        self.cur.execute("CREATE TABLE IF NOT EXISTS Location (name TEXT PRIMARY KEY, desc TEXT, lat NUMERIC, lon NUMERIC)")
        self.con.commit()

    def insert_location(self, name, desc, lat, lon):
        self.cur.execute("INSERT INTO Location VALUES(?, ?, ?, ?))",(name, desc, lat, lon, datetime.now().isoformat()))
        self.con.commit()

    def bulk_insert(self, data):
        columns = ', '.join(data[0].keys())
        place_holders = ', '.join(['?'] * len(data[0]))
        insert_query = f"INSERT INTO Location ({columns}) VALUES ({place_holders})" 
        print(insert_query)
        self.cur.executemany(insert_query, [tuple(row.values()) for row in data])    
        self.con.commit()

    def read_location(self, name=None):
        if name:
           self.cur.execute("SELECT name, desc, lat, lon FROM Location WHERE name=?",(name))
           return self.cur.fetchone()
        else:
           self.cur.execute("SELECT * FROM Location")   
           return self.cur.fetchall()

    def get_number_of_rows(self):
        self.cur.execute("SELECT COUNT(1) FROM Location") 
        return self.cur.fetchall()     

    def delete_location(self, name):
        if name:
           self.cur.execute("DELETE FROM Location WHERE name = ?",(name,))
        else:
            self.cur.execute("DELETE FROM Location")   
        self.con.commit()

    def close(self):
        self.con.close()

 
if __name__=='__main__':
    db_path = "/home/madhekar/work/home-media-app/data/app-data/db/"
    db_name = "zesha_sqlite"
    columns = ["name TEXT PRIMARY KEY", "desc TEXT", "lat NUMERIC", "lon NUMERIC"]
    data = [
        {
            "name": "ca-sageMesa",
            "desc": "home location",
            "lat": 32.9687,
            "lon": -117.184196,
        },
        {
            "name": "ca-hickman",
            "desc": "elementary school",
            "lat": 32.91499,
            "lon": -117.15336,
        },
        {
            "name": "ca-qualcomm",
            "desc": "qualcomm work",
            "lat": 32.89732,
            "lon": -117.195686,
        },
        {"name": "ca-berkeley", "desc": "UCB", "lat": 37.871899, "lon": -122.258537},
        {
            "name": "ca-torrypineshigh",
            "desc": "Torry Pines High San Diego",
            "lat": 32.89208,
            "lon": -117.23907,
        },
    ]

    db_con = Location(dbpath=db_path, dbname=db_name)

    db_con.create_location_tbl_if_not_exists()

    # db_con.bulk_insert(data=data)
    n = db_con.get_number_of_rows()
    if n[0][0] != 0:
        t_arr = db_con.read_location()
        df = pd.DataFrame(t_arr)
        df.columns = ["name", "desc", "lat", "lon"]
        df.set_index('name', inplace=True)
    else:
        df = pd.DataFrame(columns=["name", "desc", "lat", "lon"])
        df.set_index('name', inplace=True)
    print(df)

    db_con.close()