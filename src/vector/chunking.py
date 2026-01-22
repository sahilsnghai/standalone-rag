import re
from typing import List, Optional

from utils.config import Config
from utils.logger import get_logger

logger = get_logger()

from dataclasses import dataclass
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


@dataclass
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    para_overlap: int = 1
    min_chunk_chars: int = 200
    force_split_threshold: int = 2


class Chunker:
    """
    Chunker with:
      - Table chunking
      - Code chunking
      - Markdown section chunking (header-aware)
      - Plain-text section/topic chunking
      - Paragraph-aware packing with optional paragraph overlap
    Returns List[str].
    """

    def __init__(self, cfg: Optional[ChunkingConfig] = None) -> None:
        self.cfg = cfg or ChunkingConfig(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

        self.recursive = RecursiveCharacterTextSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
        )

        self.character = CharacterTextSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
            separator="\n\n",
        )

        self.token = TokenTextSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
        )

        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
        )

        self.code_splitter = PythonCodeTextSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
        )

        self.section_boundary_re = re.compile(
            r"""
            ^(
                \s*(?:\d+(\.\d+)*[\)\.]|[A-Z][\)\.]|[IVXLCDM]+[\)\.])\s+\S.*  
                |
                \s*(?:chapter|section|part)\s+\d+\b.*                         
                |
                \s*(?:introduction|background|methods?|results?|discussion|
                     conclusion|summary|references|appendix)\b.*              
                |
                \s*topic\s*:\s*\S.*                                           
            )\s*$
            """,
            re.IGNORECASE | re.MULTILINE | re.VERBOSE,
        )

        self.code_fence_re = re.compile(r"```|~~~")
        self.table_pipe_re = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)


    def _is_table(self, text: str) -> bool:
        if self.table_pipe_re.search(text):
            return True
        if "\t" in text and any(line.count("\t") >= 2 for line in text.splitlines()):
            return True
        return False

    def _is_markdown(self, text: str) -> bool:
        return bool(re.search(r"^#{1,6}\s+\S+", text, re.MULTILINE))

    def _is_code(self, text: str) -> bool:
        if self.code_fence_re.search(text):
            return True

        code_signals = 0
        code_signals += 1 if re.search(r"^\s*def\s+\w+\(", text, re.MULTILINE) else 0
        code_signals += 1 if re.search(r"^\s*class\s+\w+", text, re.MULTILINE) else 0
        code_signals += (
            1 if re.search(r"^\s*(from|import)\s+\w+", text, re.MULTILINE) else 0
        )
        code_signals += (
            1 if re.search(r":\s*$", text, re.MULTILINE) else 0
        )  

        return code_signals >= 2

    def _is_long_text(self, text: str) -> bool:
        return len(text) > self.cfg.chunk_size * 3

    def _has_section_boundaries(self, text: str) -> bool:
        return bool(self.section_boundary_re.search(text))


    def _normalize_newlines(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_paragraphs(self, text: str) -> List[str]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return paras

    def _pack_units_into_chunks(self, units: List[str]) -> List[str]:
        """
        Pack units (paragraphs or sections) into chunks up to chunk_size.
        Adds paragraph-level overlap if configured.
        """
        chunks: List[str] = []
        buf: List[str] = []
        buf_len = 0

        def flush():
            nonlocal buf, buf_len
            if not buf:
                return
            chunk = "\n\n".join(buf).strip()
            if chunk:
                chunks.append(chunk)
            if self.cfg.para_overlap > 0:
                buf = buf[-self.cfg.para_overlap :]
                buf_len = len("\n\n".join(buf))
            else:
                buf = []
                buf_len = 0

        for u in units:
            if len(u) > self.cfg.chunk_size * self.cfg.force_split_threshold:
                flush()
                chunks.extend(self.recursive.split_text(u))
                continue

            candidate_len = buf_len + (2 if buf else 0) + len(u)
            if candidate_len <= self.cfg.chunk_size:
                buf.append(u)
                buf_len = candidate_len
            else:
                flush()
                buf.append(u)
                buf_len = len(u)

        flush()

        merged: List[str] = []
        for c in chunks:
            if (
                merged
                and len(c) < self.cfg.min_chunk_chars
                and (len(merged[-1]) + 2 + len(c)) <= self.cfg.chunk_size
            ):
                merged[-1] = (merged[-1] + "\n\n" + c).strip()
            else:
                merged.append(c)

        return merged


    def _chunk_table(self, text: str) -> List[str]:
        rows = text.splitlines()
        chunks, buffer = [], []
        for row in rows:
            buffer.append(row)
            if len("\n".join(buffer)) >= self.cfg.chunk_size:
                chunks.append("\n".join(buffer).strip())
                buffer = []
        if buffer:
            chunks.append("\n".join(buffer).strip())
        return [c for c in chunks if c]

    def _chunk_code(self, text: str) -> List[str]:
        return [c.strip() for c in self.code_splitter.split_text(text) if c.strip()]

    def _chunk_markdown(self, text: str) -> List[str]:
        """
        Split by headers, then pack header sections so you don't end up with tiny pieces.
        """
        docs = self.markdown_splitter.split_text(text)
        sections = [
            d.page_content.strip()
            for d in docs
            if d.page_content and d.page_content.strip()
        ]
        return self._pack_units_into_chunks(sections)

    def _chunk_by_sections_plaintext(self, text: str) -> List[str]:
        """
        Detect section headers in plain text and group content under them.
        Then pack sections into chunk_size.
        """
        lines = text.splitlines()
        sections: List[str] = []
        current: List[str] = []

        def flush():
            nonlocal current
            if current:
                s = "\n".join(current).strip()
                if s:
                    sections.append(s)
            current = []

        for line in lines:
            if self.section_boundary_re.match(line.strip()):
                flush()
                current.append(line.strip())
            else:
                current.append(line)

        flush()

        if len(sections) <= 1:
            return self._chunk_paragraphs(text)

        return self._pack_units_into_chunks(sections)

    def _chunk_paragraphs(self, text: str) -> List[str]:
        paras = self._split_paragraphs(text)
        if not paras:
            return []
        return self._pack_units_into_chunks(paras)

    def _chunk_long_text(self, text: str) -> List[str]:
        if self._has_section_boundaries(text):
            return self._chunk_by_sections_plaintext(text)
        return self._chunk_paragraphs(text)

    def _chunk_default(self, text: str) -> List[str]:
        if self._has_section_boundaries(text):
            return self._chunk_by_sections_plaintext(text)
        return self._chunk_paragraphs(text)


    def chunk(self, text: str) -> List[str]:
        text = self._normalize_newlines(text)
        if not text:
            return []

        if self._is_table(text):
            logger.info("Chunking strategy: Table")
            chunks = self._chunk_table(text)
            logger.info(f"Generated {len(chunks)} chunks")
            return chunks

        if self._is_markdown(text):
            logger.info("Chunking strategy: Markdown (section-packed)")
            chunks = self._chunk_markdown(text)
            logger.info(f"Generated {len(chunks)} chunks")
            return chunks

        if self._is_code(text):
            logger.info("Chunking strategy: Code")
            chunks = self._chunk_code(text)
            logger.info(f"Generated {len(chunks)} chunks")
            return chunks

        if self._is_long_text(text):
            logger.info("Chunking strategy: Long Text (para/section first)")
            chunks = self._chunk_long_text(text)
            logger.info(f"Generated {len(chunks)} chunks")
            return chunks

        logger.info("Chunking strategy: Default (para/section)")
        chunks = self._chunk_default(text)
        logger.info(f"Generated {len(chunks)} chunks")
        return chunks


chunker = Chunker()
