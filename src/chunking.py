import re
from typing import List

from utils.config import Config
from utils.logger import get_logger

logger = get_logger()

from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


class Chunker:
    """
    Smart chunker that automatically decides the best chunking strategy
    (text, table, markdown, code, long text) and returns only List[str].
    """

    def __init__(self) -> None:
        self.recursive = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

        self.character = CharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separator="\n\n",
        )

        self.token = TokenTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

        self.markdown = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )

        self.code = PythonCodeTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

    def _is_table(self, text: str) -> bool:
        return bool(re.search(r"\|\s*.+\s*\|", text)) or bool(
            re.search(r"\t.+\t", text)
        )

    def _is_markdown(self, text: str) -> bool:
        return bool(re.search(r"^#{1,6}\s+", text, re.MULTILINE))

    def _is_code(self, text: str) -> bool:
        return bool(re.search(r"(def |class |import |from )", text))

    def _is_long_text(self, text: str) -> bool:
        return len(text) > Config.CHUNK_SIZE * 3

    def _chunk_table(self, text: str) -> List[str]:
        rows = text.splitlines()
        chunks = []
        buffer = []

        for row in rows:
            buffer.append(row)
            if len("\n".join(buffer)) >= Config.CHUNK_SIZE:
                chunks.append("\n".join(buffer))
                buffer = []

        if buffer:
            chunks.append("\n".join(buffer))

        return chunks

    def _chunk_markdown(self, text: str) -> List[str]:
        docs = self.markdown.split_text(text)
        return [d.page_content for d in docs]

    def _chunk_code(self, text: str) -> List[str]:
        return self.code.split_text(text)

    def _chunk_long_text(self, text: str) -> List[str]:
        return self.recursive.split_text(text)

    def _chunk_default(self, text: str) -> List[str]:
        return self.character.split_text(text)

    def chunk(self, text: str) -> List[str]:
        """
        Automatically decide chunking strategy and return List[str] only.
        """

        text = text.strip()
        if not text:
            return []

        if self._is_table(text):
            logger.info("Chunking strategy: Table")
            chunks = self._chunk_table(text)
            logger.info(f"Generated {len(chunks)} chunks")
            return chunks

        if self._is_markdown(text):
            logger.info("Chunking strategy: Markdown")
            chunks = self._chunk_markdown(text)
            logger.info(f"Generated {len(chunks)} chunks")
            return chunks

        if self._is_code(text):
            logger.info("Chunking strategy: Code")
            chunks = self._chunk_code(text)
            logger.info(f"Generated {len(chunks)} chunks")
            return chunks

        if self._is_long_text(text):
            logger.info("Chunking strategy: Long Text")
            chunks = self._chunk_long_text(text)
            logger.info(f"Generated {len(chunks)} chunks")
            return chunks

        logger.info("Chunking strategy: Default")
        chunks = self._chunk_default(text)
        logger.info(f"Generated {len(chunks)} chunks")
        return chunks


chunker = Chunker()
