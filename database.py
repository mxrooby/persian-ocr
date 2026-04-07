import sqlite3
from datetime import datetime

DB_NAME = "persian_ocr.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Table 1: Persian alphabet reference
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persian_alphabet (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persian_character TEXT NOT NULL,
            latin_equivalent TEXT NOT NULL
        )
    """)

    # Table 2: Recognition history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recognition_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persian_result TEXT NOT NULL,
            latin_result TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    # Populate alphabet table if empty
    cursor.execute("SELECT COUNT(*) FROM persian_alphabet")
    if cursor.fetchone()[0] == 0:
        alphabet = [
            ('ا', '-'), ('ب', 'b'), ('پ', 'p'), ('ت', 't'),
            ('ث', 's'), ('ج', 'j'), ('چ', 'ch'), ('ح', 'h'),
            ('خ', 'kh'), ('د', 'd'), ('ذ', 'z'), ('ر', 'r'),
            ('ز', 'z'), ('ژ', 'zh'), ('س', 's'), ('ش', 'sh'),
            ('ص', 's'), ('ض', 'z'), ('ط', 't'), ('ظ', 'z'),
            ('ع', "'"), ('غ', 'gh'), ('ف', 'f'), ('ق', 'q'),
            ('ک', 'k'), ('گ', 'g'), ('ل', 'l'), ('م', 'm'),
            ('ن', 'n'), ('و', 'w'), ('ه', 'h'), ('ی', 'y'),
        ]
        cursor.executemany(
            "INSERT INTO persian_alphabet (persian_character, latin_equivalent) VALUES (?, ?)",
            alphabet
        )

    conn.commit()
    conn.close()

def save_recognition(persian_result, latin_result):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO recognition_history (persian_result, latin_result, timestamp) VALUES (?, ?, ?)",
        (persian_result, latin_result, timestamp)
    )
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT persian_result, latin_result, timestamp FROM recognition_history ORDER BY id DESC LIMIT 10"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows

def clear_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM recognition_history")
    conn.commit()
    conn.close()