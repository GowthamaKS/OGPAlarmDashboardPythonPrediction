# test_db.py
from config import DB_CONFIG
import mysql.connector
conn = mysql.connector.connect(**DB_CONFIG)
print("Connected successfully!")
conn.close()