import asyncio
from pathlib import Path

from celery import states
from celery.exceptions import Ignore
from asgiref.sync import async_to_sync, sync_to_async

from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
)
from utils.logger import get_logger

from db.db import SyncSessionLocal
from src.vector import chunker
from db.repository import save_chat_file

from src.celery import celery_app


logger = get_logger()


def _set_progress(task, pct: int, stage: str, extra: dict | None = None):
    meta = {"progress": int(pct), "stage": stage}
    if extra:
        meta.update(extra)
    task.update_state(state="PROGRESS", meta=meta)


@celery_app.task(bind=True)
def process_upload_task(
    self,
    *,
    chat_id: str,
    file_name: str,
    file_bytes: bytes,
    suffix: str,
):
    """
    Runs heavy work in Celery worker with progress + logging.

    Key fixes:
      - Celery task stays sync (def), not async.
      - DB uses SyncSessionLocal (no AsyncSession/asyncpg in Celery).
      - Async vector store is created INSIDE the coroutine (no global async client).
      - No asyncio.run() inside Celery task; use run_async() helper.
    """
    try:
        logger.info("Task started | chat_id=%s file=%s", chat_id, file_name)
        _set_progress(self, 1, "starting", {"file": file_name})

        base_dir = Path("data/raw_data")
        base_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = base_dir / file_name

        logger.info("Writing file to disk | path=%s size=%d", tmp_path, len(file_bytes))
        _set_progress(self, 5, "writing_file")
        tmp_path.write_bytes(file_bytes)

        logger.info("Selecting loader | suffix=%s", suffix)
        _set_progress(self, 10, "selecting_loader")

        if suffix == ".txt":
            loader = TextLoader(tmp_path, encoding="utf-8")
        elif suffix == ".pdf":
            loader = UnstructuredPDFLoader(
                tmp_path,
                mode="elements",
                extract_images_in_pdf=True,
                infer_table_structure=True,
            )
        elif suffix == ".docx":
            loader = UnstructuredWordDocumentLoader(tmp_path, mode="elements")
        else:
            logger.error("Unsupported file type | suffix=%s", suffix)
            self.update_state(
                state=states.FAILURE, meta={"error": f"Unsupported file type: {suffix}"}
            )
            raise Ignore()

        logger.info("Loading document")
        _set_progress(self, 25, "loading_document")

        docs = async_to_sync(loader.aload)()


        if not docs:
            logger.error("No content extracted from document")
            self.update_state(
                state=states.FAILURE, meta={"error": "No content extracted"}
            )
            raise Ignore()

        logger.info("Extracting text | elements=%d", len(docs))
        _set_progress(self, 45, "extracting_text", {"elements": len(docs)})

        text = "\n".join(
            d.page_content for d in docs if getattr(d, "page_content", None)
        )
        if not text.strip():
            logger.error("Extracted text is empty")
            self.update_state(state=states.FAILURE, meta={"error": "Empty document"})
            raise Ignore()

        logger.info("Chunking text | length=%d chars", len(text))
        _set_progress(self, 60, "chunking")
        chunks = chunker.chunk(text)

        if not chunks:
            logger.error("Chunking produced no chunks")
            self.update_state(
                state=states.FAILURE, meta={"error": "No chunks generated"}
            )
            raise Ignore()

        logger.info("Chunking complete | chunks=%d", len(chunks))
        metadatas = [
            {"file_name": file_name, "chunk_id": idx} for idx in range(len(chunks))
        ]

        logger.info("Saving metadata & indexing")
        _set_progress(self, 70, "saving_metadata", {"chunks": len(chunks)})

        # ---- SYNC DB in Celery ----
        db = SyncSessionLocal()
        try:
            logger.info("Saving file metadata to DB")
            async_to_sync(save_chat_file(
                db,
                chat_id=chat_id,
                file_name=file_name,
                file_path=str(tmp_path),
                file_type=suffix[1:],
                file_size=str(len(file_bytes)),
            ))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

        _set_progress(self, 85, "indexing")

        async def _vector_ops():
            from src.vector import VectorStore
            from src.vector import embedder
            vector_store = VectorStore(embedder)

            logger.info("Ensuring vector collection | chat_id=%s", chat_id)
            await vector_store.ensure_collection(chat_id)

            logger.info("Adding chunks to vector store")
            await vector_store.add(
                collection_name=chat_id,
                texts=chunks,
                metadatas=metadatas,
            )

        async_to_sync(_vector_ops)()

        logger.info(
            "Task completed successfully | chat_id=%s file=%s chunks=%d",
            chat_id,
            file_name,
            len(chunks),
        )
        _set_progress(self, 100, "done", {"chunks": len(chunks)})

        return {
            "status": "uploaded",
            "chat_id": chat_id,
            "file": file_name,
            "chunks": len(chunks),
            "path": str(tmp_path),
        }

    except Ignore:
        logger.warning("Task ignored | chat_id=%s file=%s", chat_id, file_name)
        raise
    except Exception as e:
        logger.exception("process_upload_task failed")
        self.update_state(state=states.FAILURE, meta={"error": str(e)})
        raise Ignore()
