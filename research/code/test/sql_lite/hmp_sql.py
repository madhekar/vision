import os
import sqlite3 as sl3
from datetime import datetime, timezone

def create_or_connect_database(db_path, db_name):
    return sl3.connect(os.path.join(db_path, db_name))

def create_tables(con):
    cur = con.cursor()
    cur.execute("CREATE TABLE storage(name string primary key, description string, serial_number string, timestamp string)")
    cur.execute("CREATE TABLE storage_audit(storage_name string foreign key, duplicate_files number(10), bad_quality_files number(10), missing_data_files number(10), metadata_generated_files number(10), vector_db_loaded_files number(10), memory_usage string, folders_added number(10), files_added number(10), timestamp string )")
    cur.commit()

def initialize_storage(con, storage_name, description, serial_number):
    cur = con.cursor()
    cur.execute("INSERT INTO storage VALUES(?, ?, ?, ?))", (storage_name, description, serial_number, datetime.now())) 
    cur.commit()

def upsert_storage_audit(con, sn, df, bqf, mdf, mgf, vlf, mu, fod, fid ):
    cur = con.cursor()
    cur.execute("INSERT INTO storage_audit VALUES (?,?,?,?,?,?,?,?,?)", (sn, df, bqf, mdf, mgf, vlf, mu, fod, fid, datetime.now()))
