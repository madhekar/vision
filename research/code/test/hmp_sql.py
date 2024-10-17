import os
import sqlite3 as sl3

def create_or_connect_database(db_path, db_name):
    return sl3.connect(os.path.join(db_path, db_name))

def create_tables(con):
    cur = con.cursor()
    cur.execute("CREATE TABLE store(name string primary key, description string, serial_number string, ts string)")
    cur.execute("CREATE TABLE store_attributes(store_name string foreign key, duplicate_files number(10), bad_quality_files number(10), missing_data_files number(10), metadata_generated_files number(10), vector_db_loaded_files number(10), memory_usage string )")
    cur.execute("CREATE TABLE transactions(store_name string foreign key, )")
