from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
)
from sqlalchemy.orm import sessionmaker, declarative_base

from utils.config import Config

 

engine = create_async_engine(
    Config.DATABASE_URL,
    pool_size=10,          
    max_overflow=20,       
    pool_timeout=30,       
    pool_recycle=1800,     pool_pre_ping=True,    
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
    from db.models import Chat, Message, ChatFile
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
