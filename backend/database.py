import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    connection = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password=os.getenv("MYSQL_PASSWORD"),
        database="student_performance_db"
    )

    return connection


if __name__ == "__main__":
    connection = get_connection()

    if connection.is_connected():
        print("✅ MySQL connection successful!")

    connection.close()
    print("✅ MySQL connection closed!")