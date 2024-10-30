import os
import sqlite3 as sl
from datetime import datetime, timezone


class Location:
    def __init__(self, dbpath, dbname):
        self.con = sl.connect(os.path.join(dbpath, dbname))
        self.cur = self.con.cursor()

    def create_location_tbl_if_not_exists(self):
        self.cur.execute("CREATE TABLE if not exists Location(name string primary key, desc string, lat float, lon float, timestamp string)")
        self.cur.commit()

    def insert_location(self, name, desc, lat, lon):
        self.cur.execute("INSERT INTO Location VALUES(?, ?, ?, ?))",(name, desc, lat, lon, datetime.now()))
        self.cur.commit()

    def read_location(self, name=None):
        if name:
           self.cur.execute("select name, desc, lat, lon from Location where name=?",(name))
           return self.cur.fetchone()
        else:
           self.cur.execute("select * from Location")   
           return self.cur.fetchall()

    def get_number_of_rows(self):
        self.cur.execute("select count(*) from Location") 
        return self.cur.fetchone()     

    def delete_location(self, name):
        self.cur.execute("Delete from Location where name = ?",(name,))
        self.cur.commit()

    def close(self):
        self.con.close()

 
