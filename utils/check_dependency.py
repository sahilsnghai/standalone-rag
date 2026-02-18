import sys
from utils.logger import get_logger
from utils.config import Config

logger = get_logger()

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   logger.info(f"{GREEN} {msg}{RESET}")
def fail(msg): logger.error(f"{RED} {msg}{RESET}")
def hint(msg): logger.warning(f"{YELLOW} {msg}{RESET}")


def check_dependencies():
    """Check if required Python packages are installed"""
    logger.info("Checking Python dependencies...")
    try:
        import fastapi
        import langchain
        import langgraph
        import openai
        import sqlalchemy
        import streamlit
        import uvicorn
        ok("All required packages are installed.")
        return True
    except ImportError as e:
        fail(f"Missing dependency: {e}")
        hint("Run: pip install -r requirements.txt")
        return False


def check_env_file():
    """Check if .env file exists and has required variables"""
    import os
    logger.info("Checking .env file...")

    if not os.path.exists(".env"):
        fail(".env file not found.")
        hint("Copy .env.example to .env and fill in your details.")
        return False

    with open(".env", "r") as f:
        content = f.read()

    required_vars = ["DATABASE_URL", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if f"{var}=" not in content]

    if missing_vars:
        fail(f"Missing variables in .env: {', '.join(missing_vars)}")
        hint("Add the missing variables to your .env file.")
        return False

    ok("Environment setup looks fine.")
    return True


def check_redis():
    """Check Redis connection"""
    logger.info("Checking Redis...")
    try:
        import redis
        r = redis.Redis.from_url(Config.CELERY_BROKER_URL, socket_connect_timeout=3)
        r.ping()
        ok(f"Redis connected → {Config.CELERY_BROKER_URL}")
        return True
    except Exception as e:
        fail(f"Redis connection failed: {e}")
        hint("Start Redis: sudo service redis-server start")
        return False


def check_celery():
    """Check Celery broker reachability"""
    logger.info("Checking Celery...")
    try:
        from celery import Celery
        app = Celery(broker=Config.CELERY_BROKER_URL, backend=Config.CELERY_RESULT_BACKEND)
        conn = app.connection_for_write()
        conn.connect()
        conn.release()
        ok(f"Celery broker reachable → {Config.CELERY_BROKER_URL}")
        return True
    except Exception as e:
        fail(f"Celery broker connection failed: {e}")
        hint("Start Celery worker: celery -A your_app worker --loglevel=info")
        return False


def check_postgres():
    """Check PostgreSQL connection using SQLAlchemy"""
    logger.info("Checking PostgreSQL...")
    try:
        from sqlalchemy import create_engine, text

        db_url = Config.DATABASE_URL.replace("+asyncpg", "+psycopg2")
        engine = create_engine(db_url, connect_args={"connect_timeout": 5})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        ok(f"PostgreSQL connected → {Config.DATABASE_URL}")
        return True
    except Exception as e:
        fail(f"PostgreSQL connection failed: {e}")
        hint("Start PostgreSQL: sudo service postgresql start")
        return False


def check_qdrant():
    """Check Qdrant health"""
    logger.info("Checking Qdrant...")
    try:
        import requests
        response = requests.get("http://localhost:6333/healthz", timeout=5)
        if response.status_code == 200:
            ok("Qdrant is healthy → http://localhost:6333")
            return True
        else:
            fail(f"Qdrant responded with status {response.status_code}")
            return False
    except Exception as e:
        fail(f"Qdrant connection failed: {e}")
        hint("Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        return False


def run_all_checks():
    """
    Run all dependency and service checks.
    Returns True if everything passes, False otherwise.
    """
    logger.info("=" * 50)
    logger.info("  Running Dependency & Service Checks")
    logger.info("=" * 50)

    checks = {
        "Python Packages": check_dependencies,
        "Env File":        check_env_file,
        "Redis":           check_redis,
        "Celery":          check_celery,
        "PostgreSQL":      check_postgres,
        "Qdrant":          check_qdrant,
    }

    failed = []
    for name, check_fn in checks.items():
        if not check_fn():
            failed.append(name)

    logger.info("=" * 50)

    if failed:
        fail(f"Failed checks: {', '.join(failed)}")
        fail("Fix the above issues and try again.")
        logger.info("=" * 50)
        return False

    ok("All checks passed! Ready to start.")
    logger.info("=" * 50)
    return True


if __name__ == "__main__":
    if not run_all_checks():
        sys.exit(1)