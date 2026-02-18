import asyncio
import os

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text


async def test_connection():
    # Load .env file
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL not found in .env file")

    print(f"Connecting to: {database_url}")

    engine = create_async_engine(
        database_url,
        echo=True,
    )

    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            value = result.scalar()
            print("✅ Database connected successfully!")
            print("Test query result:", value)

    except Exception as e:
        print("❌ Connection failed:")
        print(e)

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_connection())
