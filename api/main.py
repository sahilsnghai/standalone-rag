from fastapi import FastAPI, UploadFile, Depends, HTTPException, Response, BackgroundTasks
from typing import List, Dict
from uuid import UUID
from pathlib import Path
from contextlib import asynccontextmanager
import time

from langchain_core.documents import Document
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import QueryRequest, QueryResponse
from db.db import init_db, get_db
from db.repository import (
    create_chat,
    list_chats,
    add_message,
    get_chat_history,
    delete_chat_by_id,
    save_chat_file,
    get_chat_files,
)
from src.chunking import Chunker
from src.retrieval import RetrievalPipeline
from src.rag_graph import RAGGraph
from src.evaluation import RAGEvaluator
from src.vector_store import VectorStore
from src.embedding import EmbeddingModel

from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(lifespan=lifespan)

chunker = Chunker()
rag_graph = RAGGraph()
evaluator = RAGEvaluator()


async def get_vector_store() -> tuple[VectorStore, EmbeddingModel]:

    embedder = EmbeddingModel()
    vector_store = VectorStore(embedder)

    return vector_store, embedder


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
    chat, deleted = await delete_chat_by_id(db, chat_id=chat_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Error deleting chat")
    vector_store, _ = await get_vector_store()
    
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
    chat = await create_chat(db, chat_name)
    return {
        "id": str(chat.id),
        "chat_name": chat.chat_name,
        "created_at": chat.created_at.isoformat() + "Z",
    }


@app.post("/upload/{chat_id}")
async def upload_file(
    chat_id: str,
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
):
    suffix = Path(file.filename).suffix.lower()
    raw = await file.read()

    base_dir = Path("data/raw_data")
    base_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = base_dir / f"{file.filename}"

    with open(tmp_path, "wb") as f:
        f.write(raw)

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
        loader = UnstructuredWordDocumentLoader(
            tmp_path,
            mode="elements",
        )

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    docs = loader.load()

    if not docs:
        raise HTTPException(status_code=400, detail="No content extracted")

    text = "\n".join(d.page_content for d in docs if d.page_content)
    print(f"{text = }")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty document")

    chunks = chunker.chunk(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks generated")

    metadatas = [
        {
            "file_name": file.filename,
            "chunk_id": idx,
        }
        for idx in range(len(chunks))
    ]

    await save_chat_file(
        db,
        chat_id=chat_id,
        file_name=file.filename,
        file_path=str(tmp_path),
        file_type=suffix[1:],
        file_size=str(file.size),
    )

    vector_store, _ = await get_vector_store()
    vector_store: VectorStore
    await vector_store.ensure_collection(chat_id)

    await vector_store.add(
        collection_name=chat_id,
        texts=chunks,
        metadatas=metadatas,
    )

    return {
        "status": "uploaded",
        "chat_id": chat_id,
        "file": file.filename,
        "chunks": len(chunks),
    }


@app.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    bgts: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """
    Handle a query for a specific chat.
    """
    start = time.time()

    vector_store, _ = await get_vector_store()

    if await vector_store.ensure_collection(req.chat_id):
        return Response(content="Newly created the chunk. kindly provide the Documents and rerun the query.")

    retriever = await vector_store.similarity_search(req.chat_id, req.query, 5)
    pipeline = RetrievalPipeline(retriever)
    docs = await pipeline.retrieve(req.query)

    result = await rag_graph.generate(req.query, docs)
    print("Assistant response: ", result)

    bgts.add_task(add_message, db, req.chat_id, "user", req.query)
    bgts.add_task(add_message, db, req.chat_id, "assistant", result["answer"])

    # add_message(db, req.chat_id, "user", req.query)
    # add_message(db, req.chat_id, "assistant", result["answer"])

    evaluation = evaluator.evaluate(
        req.query,
        docs,
        result["answer"],
        start,
    )
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
    return await get_chat_history(db, chat_id)


@app.get("/chats/{chat_id}/files")
async def list_chat_files(chat_id: UUID, db: AsyncSession = Depends(get_db)):
    return await get_chat_files(db, chat_id)


@app.get("/health")
def health():
    return {"status": "ok"}
