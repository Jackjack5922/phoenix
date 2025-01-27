import mysql.connector
from mysql.connector import Error
import pandas as pd
from typing import Dict, Optional

class DatabaseConnection:
    def __init__(self, host: str, database: str, user: str, password: str):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None

    def connect(self) -> Optional[mysql.connector.connection.MySQLConnection]:
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password
            )
            if self.connection.is_connected():
                print(f"Successfully connected to MySQL database: {self.database}")
                return self.connection
        except Error as e:
            print(f"Error connecting to MySQL database: {e}")
            return None

    def fetch_data(self, table_name: str) -> Optional[pd.DataFrame]:
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
            
            query = f"""
                SELECT ID, Boolean, Value1, Value2, Category, Text
                FROM {table_name}
            """
            
            return pd.read_sql(query, self.connection)
        except Error as e:
            print(f"Error executing query for {table_name}: {e}")
            return None
        
    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Database connection closed.")

def load_data_from_mysql(db_config: Dict[str, str], table_name: str) -> Optional[pd.DataFrame]:
    """
    Load data from MySQL database using the provided configuration.
    
    Args:
        db_config (Dict[str, str]): Database configuration dictionary
        table_name (str): Name of the table to query
        
    Returns:
        Optional[pd.DataFrame]: DataFrame containing the queried data or None if error occurs
    """
    db = DatabaseConnection(**db_config)
    df = db.fetch_data(table_name)
    db.close()
    return df

if __name__ == "__main__":
    # Example usage
    config = {
        'host': 'localhost',
        'database': 'your_database',
        'user': 'your_username',
        'password': 'your_password'
    }
    
    df = load_data_from_mysql(config, 'your_table')
    if df is not None:
        print(df.head())