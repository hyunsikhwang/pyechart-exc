import sqlite3
import pandas as pd
from datetime import datetime

class BondDBHandler:
    def __init__(self, db_name="bond_data.db"):
        self.db_name = db_name
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_name)

    def init_db(self):
        """Initializes the database table if it doesn't exist."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bond_yields (
                STAT_CODE TEXT,
                ITEM_CODE1 TEXT,
                ITEM_NAME1 TEXT,
                TIME TEXT,
                DATA_VALUE REAL,
                PRIMARY KEY (STAT_CODE, ITEM_CODE1, TIME)
            )
        ''')
        conn.commit()
        conn.close()

    def get_last_date(self, stat_code, item_code1):
        """Returns the latest date (YYYYMMDD) stored for the given bond code."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT MAX(TIME) FROM bond_yields
            WHERE STAT_CODE = ? AND ITEM_CODE1 = ?
        ''', (stat_code, item_code1))
        result = cursor.fetchone()
        conn.close()

        return result[0] if result and result[0] else None

    def save_data(self, df):
        """Saves the DataFrame to the database. Expects columns matching the schema."""
        if df.empty:
            return

        conn = self.get_connection()
        # We process row by row or use executemany to handle PRIMARY KEY conflicts
        # Using INSERT OR REPLACE to update existing records or insert new ones
        data_to_insert = []
        for _, row in df.iterrows():
            time_val = row['TIME']
            if isinstance(time_val, pd.Timestamp):
                time_val = time_val.strftime('%Y%m%d')
            elif isinstance(time_val, str) and '-' in time_val:
                 # Handle cases like YYYY-MM-DD if they occur
                time_val = time_val.replace('-', '')

            data_to_insert.append((
                row['STAT_CODE'],
                row['ITEM_CODE1'],
                row['ITEM_NAME1'],
                time_val,
                row['DATA_VALUE']
            ))

        cursor = conn.cursor()
        cursor.executemany('''
            INSERT OR REPLACE INTO bond_yields (STAT_CODE, ITEM_CODE1, ITEM_NAME1, TIME, DATA_VALUE)
            VALUES (?, ?, ?, ?, ?)
        ''', data_to_insert)

        conn.commit()
        conn.close()

    def get_all_data(self, stat_codes, item_code1s):
        """
        Retrieves all data for the given lists of codes.
        Returns a pandas DataFrame.
        """
        conn = self.get_connection()

        # Construct placeholders for IN clause
        # Actually app.py fetches multiple codes.
        # We can implement a method that fetches everything or filters.
        # Let's just fetch all data matching the requested codes.

        query = "SELECT * FROM bond_yields WHERE STAT_CODE IN ({}) AND ITEM_CODE1 IN ({})"
        # This is a bit complex if pairs matter.
        # But looking at app.py, it iterates and concatenates.
        # So maybe we just provide a method to get data for specific pairs or all.

        # Let's just select * and let pandas filter or iterate.
        # Or better, iterate in python as per original app logic?
        # No, SQL is faster.

        # Let's keep it simple: fetch all data, then filter in memory if needed,
        # or fetch by specific pair if the data volume isn't huge.
        # Given the "large amount of data" (43,000 rows), fetching all is fine.

        df = pd.read_sql_query("SELECT * FROM bond_yields", conn)
        conn.close()

        return df
