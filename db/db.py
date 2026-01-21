from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from utils.config import Config
from utils.logger import get_logger

logger = get_logger()


engine = create_async_engine(
    Config.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    try:
        from db.models import Chat, ChatFile, Message

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Initialized database tables")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
