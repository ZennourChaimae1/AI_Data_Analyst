import os
import psycopg2
import bcrypt
from dotenv import load_dotenv


load_dotenv()


# Cloud PostgreSQL (Neon)
DATABASE_URL = os.getenv("DATABASE_URL")


# Local PostgreSQL
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}


def get_connection():
    # If DATABASE_URL exists, use Neon / cloud PostgreSQL
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)

    # Otherwise, use local PostgreSQL
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # Datasets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()


def create_user(username, email, password):

    conn = get_connection()
    cursor = conn.cursor()

    hashed_password = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    )

    try:
        cursor.execute(
            """
            INSERT INTO users (username, email, password)
            VALUES (%s, %s, %s)
            """,
            (
                username,
                email,
                hashed_password.decode("utf-8")
            )
        )

        conn.commit()
        return True

    except psycopg2.IntegrityError:
        conn.rollback()
        return False

    finally:
        cursor.close()
        conn.close()


def authenticate_user(username, password):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT password
        FROM users
        WHERE username = %s
        """,
        (username,)
    )

    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if user is None:
        return False

    stored_password = user[0].encode("utf-8")

    return bcrypt.checkpw(
        password.encode("utf-8"),
        stored_password
    )


def get_user_id(username):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id
        FROM users
        WHERE username = %s
        """,
        (username,)
    )

    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if user is None:
        return None

    return user[0]


def add_dataset(user_id, file_name, file_path):

    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Check if this dataset already exists for this user
        cursor.execute(
            """
            SELECT id
            FROM datasets
            WHERE user_id = %s
              AND file_name = %s
            """,
            (user_id, file_name)
        )

        existing_dataset = cursor.fetchone()

        if existing_dataset is not None:
            return True

        cursor.execute(
            """
            INSERT INTO datasets (user_id, file_name, file_path)
            VALUES (%s, %s, %s)
            """,
            (user_id, file_name, file_path)
        )

        conn.commit()
        return True

    except Exception:
        conn.rollback()
        return False

    finally:
        cursor.close()
        conn.close()