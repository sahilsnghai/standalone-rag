from __future__ import annotations

from typing import Any, Dict, List

from utils.logger import get_logger

logger = get_logger()

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class RAGGraph:
    """
    Improvements:
    - Lower temperature for less random + more complete answers
    - Stronger instructions: summarize + structure + cite sources (chunk/file) when available
    - Context is formatted (doc separators + metadata) so the model can reference it
    - Automatic chunking/truncation to stay within model limits
    - Explicit output format (bullet list) to avoid 1-liner answers
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 500,
        verbose: bool = False,
        max_context_chars: int = 14000,
    ):
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        )
        self.max_context_chars = max_context_chars

    def _format_docs(self, docs: List[Document]) -> str:
        """
        Format docs with metadata so the LLM can cite and group topics.
        """
        parts: List[str] = []
        for i, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            file_name = meta.get("file_name") or meta.get("source") or "unknown_file"
            chunk_id = meta.get("chunk_id", meta.get("page", meta.get("id", "")))
            score = meta.get("score", "")
            header_bits = [f"Doc {i}", f"file={file_name}"]
            if chunk_id != "":
                header_bits.append(f"chunk={chunk_id}")
            if score != "":
                header_bits.append(f"score={score}")
            header = " | ".join(header_bits)

            content = (d.page_content or "").strip()
            parts.append(f"--- {header} ---\n{content}")

        context = "\n\n".join(parts).strip()

        # Truncate from the end if too large (keeps earliest docs; adjust if you prefer newest/highest score)
        if len(context) > self.max_context_chars:
            context = context[: self.max_context_chars] + "\n\n[TRUNCATED]"
        return context

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""
You are a precise RAG assistant.

Rules:
- Use ONLY the information in the Context.
- If the answer is not in the Context, reply exactly: Not found in documents
- Do NOT guess or use outside knowledge.
- Be detailed and helpful: prefer a structured answer instead of one short line.
- Extract and organize content; do not copy huge blocks verbatim.

Task:
The user asks: "{query}"

Output requirements:
- Provide a clear structured answer.
- If the question is about "topics", list topics as bullet points.
- Group related topics into sections if possible.
- When helpful, add brief supporting detail per topic (1 sentence).
- If metadata exists in the context headers, cite sources like (file=..., chunk=...) after bullets.

Context:
{context}
""".strip()

    async def generate(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        context = self._format_docs(docs)

        # If nothing retrieved, enforce "Not found in documents"
        if not context.strip():
            logger.info(
                "No context available for generation. Returning 'Not found in documents'."
            )
            return {"answer": "Not found in documents", "prompt": ""}

        prompt = self._build_prompt(query=query, context=context)
        logger.info(f"Generating answer for query: {query}")

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        answer = (response.content or "").strip()

        # Hard guard: if model tries to answer without context
        if not answer:
            answer = "Not found in documents"
            logger.warning(
                "Empty response from LLM, defaulting to 'Not found in documents'"
            )

        logger.info(f"Generated answer length: {len(answer)}")
        return {"answer": answer, "prompt": prompt}

rag_graph = RAGGraph()
