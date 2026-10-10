import os
import uuid
import psycopg2
import bcrypt
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}


def get_connection():
    """Connect to Supabase PostgreSQL or the configured local database."""
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)

    if not all(DB_CONFIG.values()):
        raise ValueError(
            "Configure DATABASE_URL or all DB_* variables in .env."
        )

    return psycopg2.connect(**DB_CONFIG)


def init_db():
    """Create the users table and ensure datasets has the required columns."""
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS public.users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS public.datasets (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL
                        REFERENCES public.users(id) ON DELETE CASCADE,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, file_name)
                )
            """)

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def create_user(username, email, password):
    """Register a user with a bcrypt-hashed password."""
    conn = get_connection()

    try:
        hashed_password = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt()
        ).decode("utf-8")

        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO public.users (username, email, password)
                VALUES (%s, %s, %s)
            """, (username, email, hashed_password))

        conn.commit()
        return True

    except psycopg2.IntegrityError:
        conn.rollback()
        return False

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def authenticate_user(username, password):
    """Authenticate a user."""
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT password
                FROM public.users
                WHERE username = %s
            """, (username,))
            user = cursor.fetchone()
    finally:
        conn.close()

    if not user:
        return False

    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            user[0].encode("utf-8")
        )
    except (ValueError, TypeError):
        return False


def get_user_id(username):
    """Return the UUID of a user, or None."""
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id
                FROM public.users
                WHERE username = %s
            """, (username,))
            user = cursor.fetchone()

        return str(user[0]) if user else None

    finally:
        conn.close()


def add_dataset(user_id, file_name, file_path):
    """Insert or update dataset metadata."""
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO public.datasets
                    (id, user_id, file_name, file_path, created_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id, file_name)
                DO UPDATE SET
                    file_path = EXCLUDED.file_path,
                    created_at = CURRENT_TIMESTAMP
            """, (str(uuid.uuid4()), user_id, file_name, file_path))

        conn.commit()
        return True

    except Exception as e:
        conn.rollback()
        print(f"Error saving dataset metadata: {e}")
        return False

    finally:
        conn.close()


def get_user_datasets(user_id):
    """Return datasets belonging to one user."""
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, file_name, file_path, created_at
                FROM public.datasets
                WHERE user_id = %s
                ORDER BY created_at DESC
            """, (user_id,))
            return cursor.fetchall()

    finally:
        conn.close()


def delete_dataset(user_id, file_name):
    """Delete metadata only, not the file in Supabase Storage."""
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                DELETE FROM public.datasets
                WHERE user_id = %s AND file_name = %s
            """, (user_id, file_name))

        conn.commit()
        return True

    except Exception as e:
        conn.rollback()
        print(f"Error deleting dataset metadata: {e}")
        return False

    finally:
        conn.close()


def rename_dataset(user_id, old_name, new_name, new_path):
    """Update metadata after renaming a file in Supabase Storage."""
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE public.datasets
                SET file_name = %s, file_path = %s
                WHERE user_id = %s AND file_name = %s
            """, (new_name, new_path, user_id, old_name))

            if cursor.rowcount == 0:
                conn.rollback()
                return False

        conn.commit()
        return True

    except Exception as e:
        conn.rollback()
        print(f"Error renaming dataset metadata: {e}")
        return False

    finally:
        conn.close()