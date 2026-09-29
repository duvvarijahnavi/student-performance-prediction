import mysql.connector


def get_connection():
    connection = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="Janu@$336699",
        database="student_performance_db"
    )

    return connection


if __name__ == "__main__":
    connection = get_connection()

    if connection.is_connected():
        print("✅ MySQL connection successful!")

    connection.close()
    print("✅ MySQL connection closed!")