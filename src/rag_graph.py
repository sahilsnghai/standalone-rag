from __future__ import annotations

from typing import Any, Dict, List, Optional, AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from langchain_core.tools import BaseTool, tool
from langchain.agents import create_agent


# class RAGGraph:
#     """
#     Improvements:
#     - Lower temperature for less random + more complete answers
#     - Stronger instructions: summarize + structure + cite sources (chunk/file) when available
#     - Context is formatted (doc separators + metadata) so the model can reference it
#     - Automatic chunking/truncation to stay within model limits
#     - Explicit output format (bullet list) to avoid 1-liner answers
#     """

#     def __init__(
#         self,
#         model: str = "gpt-4o-mini",
#         temperature: float = 0.2,
#         max_tokens: int = 500,
#         verbose: bool = False,
#         max_context_chars: int = 14000,
#         tools: List = None
#     ):
#         self.llm = ChatOpenAI(
#             model=model,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             verbose=verbose,
#         )
#         self.max_context_chars = max_context_chars

#     def _format_docs(self, docs: List[Document]) -> str:
#         """
#         Format docs with metadata so the LLM can cite and group topics.
#         """
#         parts: List[str] = []
#         for i, d in enumerate(docs, start=1):
#             meta = d.metadata or {}
#             file_name = meta.get("file_name") or meta.get("source") or "unknown_file"
#             chunk_id = meta.get("chunk_id", meta.get("page", meta.get("id", "")))
#             score = meta.get("score", "")
#             header_bits = [f"Doc {i}", f"file={file_name}"]
#             if chunk_id != "":
#                 header_bits.append(f"chunk={chunk_id}")
#             if score != "":
#                 header_bits.append(f"score={score}")
#             header = " | ".join(header_bits)

#             content = (d.page_content or "").strip()
#             parts.append(f"--- {header} ---\n{content}")

#         context = "\n\n".join(parts).strip()

#         # Truncate from the end if too large (keeps earliest docs; adjust if you prefer newest/highest score)
#         if len(context) > self.max_context_chars:
#             context = context[: self.max_context_chars] + "\n\n[TRUNCATED]"
#         return context

#     def _build_prompt(self, query: str, context: str) -> str:
#         return f"""
# You are a precise RAG assistant.

# Rules:
# - Use ONLY the information in the Context.
# - If the answer is not in the Context, reply exactly: Not found in documents
# - Do NOT guess or use outside knowledge.
# - Be detailed and helpful: prefer a structured answer instead of one short line.
# - Extract and organize content; do not copy huge blocks verbatim.

# Task:
# The user asks: "{query}"

# Output requirements:
# - Provide a clear structured answer.
# - If the question is about "topics", list topics as bullet points.
# - Group related topics into sections if possible.
# - When helpful, add brief supporting detail per topic (1 sentence).
# - If metadata exists in the context headers, cite sources like (file=..., chunk=...) after bullets.

# Context:
# {context}
# """.strip()

#     async def generate(self, query: str, docs: List[Document]) -> Dict[str, Any]:
#         context = self._format_docs(docs)

#         # If nothing retrieved, enforce "Not found in documents"
#         if not context.strip():
#             return {"answer": "Not found in documents", "prompt": ""}

#         prompt = self._build_prompt(query=query, context=context)

#         response = await self.llm.ainvoke([HumanMessage(content=prompt)])
#         answer = (response.content or "").strip()

#         # Hard guard: if model tries to answer without context
#         if not answer:
#             answer = "Not found in documents"

#         return {"answer": answer, "prompt": prompt}


class RAGGraph:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 500,
        verbose: bool = False,
        max_context_chars: int = 14000,
        tools: Optional[List[BaseTool]] = None,
        agentic: bool = True,
        streaming: bool = True,
    ):
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            streaming=streaming
        )


        self.max_context_chars = max_context_chars
        self.verbose = verbose
        self.tools = tools or []
        self.agentic = agentic

        if self.agentic:
            self._init_agent()

    def _format_docs(self, docs: List[Document]) -> str:
        parts: List[str] = []
        for i, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            file_name = meta.get("file_name") or meta.get("source") or "unknown_file"
            chunk_id = meta.get("chunk_id", meta.get("page", meta.get("id", "")))
            header = f"Doc {i} | file={file_name}" + (f" | chunk={chunk_id}" if chunk_id != "" else "")
            parts.append(f"--- {header} ---\n{(d.page_content or '').strip()}")

        context = "\n\n".join(parts).strip()
        if len(context) > self.max_context_chars:
            context = context[: self.max_context_chars] + "\n\n[TRUNCATED]"
        return context

    def _init_agent(self):
        @tool("get_context")
        def get_context(context: str) -> str:
            """Returns the document context."""
            return context

        self._context_tool = get_context
        all_tools = [self._context_tool, *self.tools]
        prompt = SystemMessage(
            """You are a precise RAG assistant.

                Rules:
                - Use ONLY the information in Context.
                - If the answer is not in Context, reply exactly: Not found in documents
                - Do NOT guess or use outside knowledge.
                - Be structured and detailed.

                Question: {input}

                Context:
                {context}

                {agent_scratchpad}
            """
        )
        self.agent_executor = create_agent(self.llm, all_tools, system_prompt=prompt)

    async def stream(self, query: str, docs: List[Document] = None) -> AsyncIterator[str]:
        """
        Yields text chunks as the LLM generates them.
        Works for both agentic and non-agentic paths.
        """
        context = self._format_docs(docs)

        if self.agentic:
            async for event in self.agent_executor.astream({"input": query, "context": context}):
                if isinstance(event, dict):
                    if "output" in event and isinstance(event["output"], str):
                        yield event["output"]
                    elif "chunk" in event and hasattr(event["chunk"], "content"):
                        yield event["chunk"].content
            return

        prompt = f"""
                You are a precise RAG assistant.

                Rules:
                - Use ONLY the information in the Context.
                - If the answer is not in the Context, reply exactly: Not found in documents
                - Do NOT guess or use outside knowledge.
                - Be structured and detailed.

                Question: "{query}"

                Context:
                {context}

            """.strip()

        async for chunk in self.llm.astream([HumanMessage(content=prompt)]):
            text = getattr(chunk, "content", "")
            if text:
                yield text

    async def generate(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        context = self._format_docs(docs)
        if not context.strip():
            return {"answer": "Not found in documents"}

        if self.agentic:
            result = await self.agent_executor.ainvoke({"input": query, "context": context})
            return {"answer": (result.get("output") or "").strip()}

        prompt = f"""
                    You are a precise RAG assistant.

                    Rules:
                    - Use ONLY the information in the Context.
                    - If the answer is not in the Context, reply exactly: Not found in documents

                    Question: "{query}"

                    Context:
                    {context}
            """.strip()

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return {"answer": (response.content or "").strip() or "Not found in documents"}
