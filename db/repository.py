from typing import List, Dict
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from db.models import Chat, Message, ChatFile


async def save_chat_file(
    db: AsyncSession,
    chat_id: UUID,
    file_name: str,
    file_path: str,
    file_type: str | None = None,
    file_size: str | None = None,
) -> ChatFile:
    try:
        chat_file = ChatFile(
            chat_id=chat_id,
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            file_path=file_path,
        )
        db.add(chat_file)
        await db.commit()
        await db.refresh(chat_file)
        return chat_file
    except SQLAlchemyError:
        await db.rollback()
        raise


async def create_chat(
    db: AsyncSession,
    chat_name: str,
) -> Chat:
    try:
        chat = Chat(chat_name=chat_name)
        db.add(chat)
        await db.commit()
        await db.refresh(chat)
        return chat
    except SQLAlchemyError:
        await db.rollback()
        raise


async def list_chats(
    db: AsyncSession,
    limit: int = 20,
    offset: int = 0,
) -> List[Chat]:
    try:
        result = await db.execute(
            select(Chat)
            .order_by(Chat.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return result.scalars().all()
    except SQLAlchemyError:
        raise


async def delete_chat_by_id(
    db: AsyncSession,
    chat_id: UUID,
) -> bool:
    try:
        result = await db.execute(
            select(Chat).where(Chat.id == chat_id)
        )
        chat = result.scalar_one_or_none()
        if not chat:
            return chat, False

        await db.delete(chat)
        await db.commit()
        return chat, True,
    except SQLAlchemyError:
        await db.rollback()
        raise


async def add_message(
    db: AsyncSession,
    chat_id: UUID,
    role: str,
    content: str,
) -> Message:
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")

    try:
        message = Message(
            chat_id=chat_id,
            role=role,
            content=content,
        )
        db.add(message)
        await db.commit()
        await db.refresh(message)
        return message
    except SQLAlchemyError:
        await db.rollback()
        raise


async def get_chat_history(
    db: AsyncSession,
    chat_id: UUID,
) -> List[Dict]:
    try:
        result = await db.execute(
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at.asc())
        )
        messages = result.scalars().all()

        return [
            {
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() + "Z",
                "token_count": getattr(m, "token_count", None),
            }
            for m in messages
        ]
    except SQLAlchemyError:
        raise


async def get_chat_files(
    db: AsyncSession,
    chat_id: UUID,
) -> List[Dict]:
    try:
        result = await db.execute(
            select(ChatFile)
            .where(ChatFile.chat_id == chat_id)
            .order_by(ChatFile.uploaded_at.asc())
        )
        files = result.scalars().all()

        return [
            {
                "id": str(f.id),
                "chat_id": str(f.chat_id),
                "file_name": f.file_name,
                "file_type": f.file_type,
                "file_size": f.file_size,
                "file_path": f.file_path,
                "uploaded_at": f.uploaded_at.isoformat() + "Z",
            }
            for f in files
        ]
    except SQLAlchemyError:
        raise
