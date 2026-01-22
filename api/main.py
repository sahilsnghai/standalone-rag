import asyncio
import time
import json

from celery.result import AsyncResult
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List
from uuid import UUID

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Response,
    UploadFile,
    Request,
)
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import QueryRequest, QueryResponse
from db.db import get_db, init_db
from db.repository import (
    add_message,
    create_chat,
    delete_chat_by_id,
    get_chat_files,
    get_chat_history,
    list_chats,
)
from src.vector import evaluator, rag_graph, vector_store, RetrievalPipeline
from utils.logger import get_logger

from src.celery import process_upload_task , celery_app

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(lifespan=lifespan)

# chunker = Chunker()
# rag_graph = RAGGraph()
# evaluator = RAGEvaluator()
# embedder = EmbeddingModel()
# vector_store = VectorStore(embedder)


# async def get_vector_store() -> tuple[VectorStore, EmbeddingModel]:
#     global embedder

#     return vector_store, embedder


def docs_to_dicts(docs: List[Document]) -> List[Dict]:
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


@app.get("/chats")
async def fetch_chats(
    db: AsyncSession = Depends(get_db),
) -> List[Dict]:
    logger.info("Fetching all chats")
    chats = await list_chats(db)
    return [
        {
            "id": str(chat.id),
            "chat_name": chat.chat_name,
            "created_at": chat.created_at.isoformat() + "Z",
        }
        for chat in chats
    ]


@app.delete("/chats/{chat_id}", response_model=Dict)
async def delete_chat(
    chat_id: str,
    db: AsyncSession = Depends(get_db),
):
    logger.info(f"Deleting chat: {chat_id}")
    chat, deleted = await delete_chat_by_id(db, chat_id=chat_id)

    if not deleted:
        logger.error(f"Chat not found for deletion: {chat_id}")
        raise HTTPException(status_code=404, detail="Error deleting chat")
    # vector_store, _ = await get_vector_store()

    await vector_store.delete_collection(chat_id)

    return {
        "status": "deleted",
        "id": str(chat.id),
        "chat_name": chat.chat_name,
        "created_at": chat.created_at.isoformat() + "Z",
    }


@app.post("/chats")
async def create_new_chat(
    chat_name: str,
    db: AsyncSession = Depends(get_db),
) -> Dict:
    logger.info(f"Creating new chat: {chat_name}")
    chat = await create_chat(db, chat_name)
    return {
        "id": str(chat.id),
        "chat_name": chat.chat_name,
        "created_at": chat.created_at.isoformat() + "Z",
    }


# @app.post("/upload/{chat_id}")
# async def upload_file(
#     chat_id: str,
#     file: UploadFile,
#     db: AsyncSession = Depends(get_db),
# ):
#     logger.info(f"Starting upload process for file {file.filename} to chat {chat_id}")
#     suffix = Path(file.filename).suffix.lower()

#     logger.info(f"Reading file content for {file.filename}...")
#     raw = await file.read()
#     logger.info(f"File content read. Size: {len(raw)} bytes.")

#     base_dir = Path("data/raw_data")
#     base_dir.mkdir(parents=True, exist_ok=True)

#     tmp_path = base_dir / f"{file.filename}"

#     # Use aiofiles for async file writing
#     logger.info(f"Writing file to disk asynchronously...")
#     async with aiofiles.open(tmp_path, "wb") as f:
#         await f.write(raw)

#     logger.info(f"Stored the file in folder {tmp_path}")

#     if suffix == ".txt":
#         loader = TextLoader(tmp_path, encoding="utf-8")

#     elif suffix == ".pdf":
#         loader = UnstructuredPDFLoader(
#             tmp_path,
#             mode="elements",
#             extract_images_in_pdf=True,
#             infer_table_structure=True,
#         )

#     elif suffix == ".docx":
#         loader = UnstructuredWordDocumentLoader(
#             tmp_path,
#             mode="elements",
#         )

#     else:
#         logger.error(f"Unsupported file type: {suffix}")
#         raise HTTPException(status_code=400, detail="Unsupported file type")

#     logger.info("Loading document...")
#     docs = await loader.aload()
#     logger.info(f"Document loaded. Number of pages/elements: {len(docs)}")

#     if not docs:
#         logger.error("No content extracted from file")
#         raise HTTPException(status_code=400, detail="No content extracted")

#     # Run text extraction in thread pool to avoid blocking
#     logger.info("Extracting text from documents...")
#     text = await asyncio.to_thread(
#         lambda: "\n".join(d.page_content for d in docs if d.page_content)
#     )
#     logger.info(f"Extracted text length: {len(text)} characters")
#     logger.info(f"Extracted text preview: {text[:100]}...")
#     if not text.strip():
#         logger.error("Empty document")
#         raise HTTPException(status_code=400, detail="Empty document")

#     # Run chunking in thread pool (CPU-intensive operation)
#     logger.info("Chunking text asynchronously...")
#     chunks = await asyncio.to_thread(chunker.chunk, text)
#     logger.info(f"Text chunked into {len(chunks)} chunks.")
#     if not chunks:
#         logger.error("No chunks generated")
#         raise HTTPException(status_code=400, detail="No chunks generated")

#     metadatas = [
#         {
#             "file_name": file.filename,
#             "chunk_id": idx,
#         }
#         for idx in range(len(chunks))
#     ]

#     logger.info("Saving file metadata to DB...")
#     await save_chat_file(
#         db,
#         chat_id=chat_id,
#         file_name=file.filename,
#         file_path=str(tmp_path),
#         file_type=suffix[1:],
#         file_size=str(file.size),
#     )
#     logger.info("File metadata saved.")

#     # vector_store, _ = await get_vector_store()
#     logger.info(f"Ensuring collection {chat_id} exists...")
#     await vector_store.ensure_collection(chat_id)

#     logger.info(f"Adding {len(chunks)} chunks to vector store...")
#     await vector_store.add(
#         collection_name=chat_id,
#         texts=chunks,
#         metadatas=metadatas,
#     )
#     logger.info("Chunks added to vector store successfully.")

#     return {
#         "status": "uploaded",
#         "chat_id": chat_id,
#         "file": file.filename,
#         "chunks": len(chunks),
#     }


@app.post("/upload/{chat_id}")
async def upload_file(
    chat_id: str, file: UploadFile, db: AsyncSession = Depends(get_db)
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    logger.info("read the file and executing celery")

    task = process_upload_task.delay(
        chat_id=chat_id,
        file_name=file.filename,
        file_bytes=raw,
        suffix=suffix,
    )

    logger.info("Updated to the celery queue")

    return {
        "status": "queued",
        "chat_id": chat_id,
        "file": file.filename,
        "task_id": task.id,
        "sse_url": f"/upload/stream/{task.id}",
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/upload/stream/{task_id}")
async def upload_stream(task_id: str, request: Request):
    async def gen():
        # initial ping so UI connects immediately
        yield _sse("connected", {"task_id": task_id})

        last_payload: dict | None = None
        last_sent_at = 0.0
        KEEPALIVE_SECONDS = 5.0
        POLL_SECONDS = 0.5

        while True:
            if await request.is_disconnected():
                break

            res: AsyncResult = AsyncResult(task_id, app=celery_app)

            payload: dict = {"task_id": task_id, "state": res.state}

            if res.state == "PROGRESS":
                info = res.info or {}
                if isinstance(info, dict):
                    payload.update(info)
                else:
                    payload["info"] = str(info)

            elif res.state == "SUCCESS":
                result = res.result or {}
                if isinstance(result, dict):
                    payload.update(result)
                else:
                    payload["result"] = str(result)
                yield _sse("done", payload)
                break

            elif res.state in ("FAILURE", "REVOKED"):
                info = res.info
                payload["error"] = str(info) if info else "failed"
                yield _sse("error", payload)
                break

            now = time.monotonic()
            changed = (last_payload is None) or (payload != last_payload)

            if changed:
                yield _sse("progress", payload)
                last_payload = payload.copy() 
                last_sent_at = now

            elif (now - last_sent_at) >= KEEPALIVE_SECONDS:
                yield _sse("progress", last_payload)
                last_sent_at = now

            await asyncio.sleep(POLL_SECONDS)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    bgts: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """
    Handle a query for a specific chat.
    """
    logger.info(f"Received query for chat {req.chat_id}: {req.query}")
    start = time.time()

    # vector_store, _ = await get_vector_store()

    logger.info(f"Checking collection existence for chat {req.chat_id}...")
    if await vector_store.ensure_collection(req.chat_id):
        logger.info(
            f"Collection {req.chat_id} created during query (unexpected). Returning prompt to upload docs."
        )
        return Response(
            content="Newly created the chunk. kindly provide the Documents and rerun the query."
        )

    logger.info("Starting similarity search...")
    retriever = await vector_store.similarity_search(req.chat_id, req.query, 5)
    logger.info("Similarity search complete.")

    logger.info("Starting retrieval pipeline...")
    pipeline = RetrievalPipeline(retriever)
    docs = await pipeline.retrieve(req.query)
    logger.info(f"Retrieval pipeline complete. Retrieved {len(docs)} docs.")

    logger.info("Starting generation...")
    result = await rag_graph.generate(req.query, docs)
    logger.info(f"Generation complete. Assistant response: {result}")

    bgts.add_task(add_message, db, req.chat_id, "user", req.query)
    bgts.add_task(add_message, db, req.chat_id, "assistant", result["answer"])

    # add_message(db, req.chat_id, "user", req.query)
    # add_message(db, req.chat_id, "assistant", result["answer"])

    logger.info("Starting evaluation...")
    evaluation = evaluator.evaluate(
        req.query,
        docs,
        result["answer"],
        start,
    )
    logger.info(f"Evaluation complete: {evaluation}")

    response = QueryResponse(
        answer=result["answer"],
        retrieved_docs=docs_to_dicts(docs),
        evaluation=evaluation,
    )

    return response


@app.get("/chats/{chat_id}/history")
async def chat_history(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    logger.info(f"Fetching history for chat: {chat_id}")
    return await get_chat_history(db, chat_id)


@app.get("/chats/{chat_id}/files")
async def list_chat_files(chat_id: UUID, db: AsyncSession = Depends(get_db)):
    logger.info(f"Fetching files for chat: {chat_id}")
    return await get_chat_files(db, chat_id)


@app.get("/health")
def health():
    logger.info("Health check")
    return {"status": "ok"}
