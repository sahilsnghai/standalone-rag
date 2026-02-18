import hashlib
import time
import requests
import streamlit as st
import json

from utils.logger import get_logger
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_incrementing


logger = get_logger(prefix="[UI]")

API_URL = "http://localhost:8000"


def file_fingerprint(uploaded) -> str:
    data = uploaded.getvalue()
    h = hashlib.sha256(data).hexdigest()
    fingerprint = f"{uploaded.name}:{len(data)}:{h}"
    logger.debug(f"Generated fingerprint for {uploaded.name}: {fingerprint}")
    return fingerprint


def format_file_size(size):
    if not size:
        return ""
    size = float(size)
    if size < 1024:
        return f"{size:.0f} B"
    elif size < 1024**2:
        return f"{size / 1024:.2f} KB"
    else:
        return f"{size / (1024 ** 2):.2f} MB"


@retry(
    stop=stop_after_attempt(5),
    wait=wait_incrementing(0.5),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def _health_request() -> bool:
    logger.debug("Attempting health check request")
    r = requests.get(f"{API_URL}/health", timeout=3)
    logger.debug(f"Health check response: {r.status_code}")
    return r.status_code == 200


@st.cache_data(ttl=0)
def cached_health() -> bool:
    try:
        logger.info("Running cached health check")
        result = _health_request()
        logger.info(f"Health check result: {result}")
        return result
    except Exception as e:
        logger.error(f"Health check failed after retries: {e}")
        return False


@st.cache_data(ttl=0)
def cached_fetch_chats():
    try:
        logger.info("Fetching chats from API")
        r = requests.get(f"{API_URL}/chats", timeout=5)
        r.raise_for_status()
        chats = r.json()
        logger.info(f"Successfully fetched {len(chats)} chats")
        return chats
    except Exception as e:
        logger.error(f"Error fetching chats: {e}")
        return []


@st.cache_data(ttl=0)
def cached_fetch_history(chat_id: str):
    try:
        logger.info(f"Fetching history for chat_id: {chat_id}")
        r = requests.get(f"{API_URL}/chats/{chat_id}/history", timeout=10)
        r.raise_for_status()
        history = r.json()
        logger.info(f"Successfully fetched {len(history)} messages for chat {chat_id}")
        return history
    except Exception as e:
        logger.error(f"Error fetching history for chat {chat_id}: {e}")
        return []


@st.cache_data(ttl=0)
def cached_fetch_uploaded_files(chat_id: str):
    try:
        logger.info(f"Fetching uploaded files for chat_id: {chat_id}")
        r = requests.get(f"{API_URL}/chats/{chat_id}/files", timeout=10)
        if r.status_code == 200:
            files = r.json()
            logger.info(f"Successfully fetched {len(files)} files for chat {chat_id}")
            return files
        logger.warning(f"Non-200 status code when fetching files: {r.status_code}")
        return []
    except Exception as e:
        logger.error(f"Error fetching files for chat {chat_id}: {e}")
        return []


def create_chat(chat_name: str):
    try:
        logger.info(f"Creating new chat with name: {chat_name}")
        result = requests.post(
            f"{API_URL}/chats",
            params={"chat_name": chat_name},
            timeout=10,
        )
        result.raise_for_status()
        chat_data = result.json()
        logger.info(f"Successfully created chat: {chat_name} (ID: {chat_data.get('id')})")
        st.cache_data.clear()
        return chat_data
    except Exception as e:
        logger.error(f"Backend error in create_chat: {e}")
        st.error(f"Backend error: {e}")
        return None


def delete_chat(chat_id: str):
    try:
        logger.info(f"Deleting chat with id: {chat_id}")
        result = requests.delete(f"{API_URL}/chats/{chat_id}", timeout=10)
        result.raise_for_status()
        logger.info(f"Successfully deleted chat: {chat_id}")
        st.cache_data.clear()
        st.success("Chat deleted")
    except Exception as e:
        logger.error(f"Error deleting chat {chat_id}: {e}")
        st.error(f"Error deleting chat: {e}")


def upload_file(chat_id: str, file):
    try:
        logger.info(f"Starting file upload - chat_id: {chat_id}, file: {file.name}, size: {format_file_size(file.size)}")
        files = {"file": (file.name, file.getvalue())}
        resp = requests.post(f"{API_URL}/upload/{chat_id}", files=files, timeout=300)
        resp.raise_for_status()
        result = resp.json()
        logger.info(f"File upload successful - {file.name} to chat {chat_id}")
        logger.debug(f"Upload response: {result}")
        st.cache_data.clear()
        return result
    except Exception as e:
        logger.error(f"Backend error in upload_file: {e}")
        st.error(f"Backend error: {e}")


def ask_question(chat_id: str, query: str):
    try:
        logger.info(f"Processing query for chat {chat_id}")
        logger.debug(f"Query text: {query[:100]}...")
        result = requests.post(
            f"{API_URL}/query",
            json={"chat_id": chat_id, "query": query},
            timeout=300,
        )
        result.raise_for_status()
        response = result.json()
        logger.info(f"Query processed successfully for chat {chat_id}")
        logger.debug(f"Response: {response}")
        st.cache_data.clear()
        return response
    except Exception as e:
        logger.error(f"Backend error in ask_question: {e}")
        st.error(f"Backend error: {e}")
        return {}


def typewriter(text: str, speed: float = 0.015):
    logger.debug(f"Starting typewriter animation with {len(text)} characters")
    box = st.empty()
    out = ""
    for ch in text:
        out += ch
        box.markdown(out)
        time.sleep(speed)
    logger.debug("Typewriter animation completed")


def init_state():
    logger.debug("Initializing session state")
    st.session_state.setdefault("chat_id", None)
    st.session_state.setdefault("uploaded_fingerprints", set())
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("messages_by_chat", {})
    st.session_state.setdefault("uploading_docs", False)
    logger.debug("Session state initialized")


def ensure_default_chat_selected(chats):
    if st.session_state.chat_id is None:
        if chats:
            st.session_state.chat_id = chats[0]["id"]
            logger.info(f"Default chat selected: {chats[0].get('chat_name', 'Unknown')}")
        else:
            logger.warning("No chats available for default selection")


def load_messages_once(chat_id: str):
    """
    Load backend history only once per chat into session state.
    Prevents duplicated rendering and removes the need for a separate 'messages' list.
    """
    if chat_id not in st.session_state.messages_by_chat:
        logger.info(f"Loading message history for chat {chat_id}")
        history = cached_fetch_history(chat_id) or []
        st.session_state.messages_by_chat[chat_id] = [
            {"role": m.get("role", "assistant"), "content": m.get("content", "")}
            for m in history
        ]
        logger.info(f"Loaded {len(history)} messages for chat {chat_id}")
    else:
        logger.debug(f"Messages already loaded for chat {chat_id}, using cached")


def render_sidebar(chats):
    logger.debug("Rendering sidebar")
    st.sidebar.title("💬 Chats")

    chat_name_input = st.sidebar.text_input(
        "Chat Name:",
        value=st.session_state.get("new_chat_name", "New Chat"),
        key="new_chat_name_input",
    )

    if st.sidebar.button("New Chat", icon="➕", use_container_width=True):
        logger.info(f"New Chat button clicked with name: {chat_name_input}")
        chat = create_chat(chat_name_input)
        if chat:
            st.session_state.chat_id = chat["id"]
            st.session_state.messages_by_chat.pop(chat["id"], None)
            logger.info(f"New chat created and selected: {chat['id']}")
            st.rerun()

    st.sidebar.divider()

    if not chats:
        logger.debug("No chats available to display")
        st.sidebar.caption("No chats yet.")
        return

    logger.debug(f"Rendering {len(chats)} chats in sidebar")
    for chat in chats:
        chat_id = chat["id"]
        is_selected = chat_id == st.session_state.chat_id

        col1, col2 = st.sidebar.columns([8, 1], vertical_alignment="center")

        label = chat.get("chat_name", "Chat")
        if chat.get("summary"):
            label += f" — {chat['summary'][:40]}…"

        with col1:
            if st.button(
                label,
                key=f"chat-{chat_id}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                logger.info(f"Chat selected: {chat_id}")
                st.session_state.chat_id = chat_id
                st.rerun()

        with col2:
            if st.button("", icon="🗑️", key=f"delete-{chat_id}"):
                logger.warning(f"Delete button clicked for chat: {chat_id}")
                delete_chat(chat_id)
                st.session_state.messages_by_chat.pop(chat_id, None)

                if st.session_state.chat_id == chat_id:
                    remaining = [c for c in chats if c["id"] != chat_id]
                    st.session_state.chat_id = remaining[0]["id"] if remaining else None
                    logger.info(f"Chat deleted: {chat_id}")

                st.rerun()


def show_upload_progress(result: dict, api_base_url: str = API_URL, *, timeout: int = 300):
    """
    Streamlit progress UI driven by your SSE format:

      event: <event_name>
      data: {"progress": 10, "message": "..."}   (JSON)
      (blank line)

    Notes:
    - `result` must contain `sse_url` (relative or absolute)
    - Progress keys supported in JSON data: progress|percent|pct
    - Message keys supported: message|status|detail
    - Auto-completes when progress>=100 or state/done flags indicate completion
    - UI vanishes once complete
    """
    sse_path = (result or {}).get("sse_url")
    if not sse_path:
        logger.info("No SSE URL found in the upload result — skipping progress UI.")
        st.session_state.uploading_docs = False
        return

    sse_url = f"{api_base_url.rstrip('/')}/{str(sse_path).lstrip('/')}"

    fname = (result or {}).get("file") or ""
    logger.info(
        "Starting upload progress stream%s. SSE: %s",
        f" for '{fname}'" if fname else "",
        sse_url,
    )

    container = st.container()
    with container:
        if fname:
            st.caption(f"Processing: {fname}")
        bar = st.progress(0, text="Processing...")

    def _clamp(x: int) -> int:
        return max(0, min(100, x))

    def _iter_sse(url: str):
        logger.debug("Opening SSE connection…")
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            logger.debug("SSE connection established (HTTP %s).", r.status_code)

            buf = ""
            event_name = None
            data_lines = []

            for chunk in r.iter_content(chunk_size=1024, decode_unicode=True):
                if not chunk:
                    continue
                buf += chunk

                while "\n\n" in buf:
                    raw_event, buf = buf.split("\n\n", 1)

                    event_name = None
                    data_lines = []

                    for line in raw_event.splitlines():
                        if line.startswith("event:"):
                            event_name = line[len("event:"):].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[len("data:"):].lstrip())

                    data_str = "\n".join(data_lines) if data_lines else ""
                    yield event_name, data_str

    def _extract_progress_and_msg(payload: dict):
        p = payload.get("progress", payload.get("percent", payload.get("pct")))
        msg = payload.get("message", payload.get("status", payload.get("detail")))

        prog = None
        if p is not None:
            try:
                prog = int(float(p))
            except Exception:
                prog = None

        message = msg if isinstance(msg, str) and msg.strip() else None

        state = str(payload.get("state", "")).lower()
        done_flag = payload.get("done", payload.get("completed", payload.get("finished")))
        done = bool(done_flag) or state in {"done", "completed", "finished", "success", "succeeded"}

        if prog is not None and prog >= 100:
            done = True

        return prog, message, done

    last_p = 0
    last_msg = "Processing..."
    st.session_state.uploading_docs = True

    try:
        for event_name, data_str in _iter_sse(sse_url):
            st.session_state.uploading_docs = True

            if not data_str:
                logger.debug("Received SSE event '%s' with no data — ignoring.", event_name)
                continue

            try:
                payload = json.loads(data_str)
            except Exception:
                logger.warning(
                    "Got a progress update, but it wasn't valid JSON. Event=%s Data=%r",
                    event_name,
                    data_str,
                )
                continue

            prog, message, done = _extract_progress_and_msg(payload)

            if prog is not None:
                last_p = prog
            if message is not None:
                last_msg = message
            elif event_name:
                last_msg = event_name

            if prog is not None:
                logger.debug("Progress update: %s%% — %s", _clamp(last_p), last_msg)
            else:
                logger.debug("Status update: %s", last_msg)

            bar.progress(_clamp(last_p), text=last_msg)

            if done:
                logger.info("Upload processing completed.")
                break

        bar.progress(100, text="Complete ✅")
        logger.info("Progress UI completed successfully.")
        time.sleep(0.35)

    except Exception as e:
        logger.error(f"Progress stream error: {e}" )
        st.error(f"Progress stream error: {e}")
        time.sleep(0.75)

    finally:
        st.session_state.uploading_docs = False
        container.empty()
        logger.debug("Progress UI cleared.")


def render_documents_panel(chat_id: str):
    """Fixed: Clean upload flow - hide uploader during upload"""
    logger.debug(f"Rendering documents panel for chat {chat_id}")
    st.subheader("📎 Documents")
    
    upload_container = st.container()
    
    with upload_container:
        if st.session_state.uploading_docs:
            logger.debug("Upload in progress, showing progress UI")
            st.info("⏳ Processing your document...")
            st.progress(100, text="Uploading & Vectorizing...")
        else:
            uploader_key = f"file_uploader_{st.session_state.uploader_key}"
            uploaded = st.file_uploader(
                "Upload document (TXT, PDF, DOCX)",
                type=["txt", "pdf", "docx"],
                key=uploader_key,
                label_visibility="collapsed"
            )
            
            if uploaded:
                logger.info(f"File selected for upload: {uploaded.name} ({format_file_size(uploaded.size)})")
                fp = file_fingerprint(uploaded)
                if fp not in st.session_state.uploaded_fingerprints:
                    logger.info(f"New file detected, starting upload process: {uploaded.name}")
                    st.session_state.uploaded_fingerprints.add(fp)
                    st.session_state.uploading_docs = True
                    
                    success_msg = st.success(f"✅ {uploaded.name} received")
                    
                    result = upload_file(chat_id, uploaded)
                    
                    if result:
                        show_upload_progress(result=result)
                    
                    success_msg.empty()
                    st.session_state.uploader_key += 1
                    st.cache_data.clear()
                    logger.info(f"File upload process completed for: {uploaded.name}")
                    st.rerun()
                else:
                    logger.warning(f"Duplicate file upload attempt: {uploaded.name}")
    
    st.divider()
    
    with st.expander("📂 View uploaded files", expanded=False):
        logger.debug(f"Fetching uploaded files for chat {chat_id}")
        files = cached_fetch_uploaded_files(chat_id) or []
        if not files:
            logger.debug(f"No files uploaded for chat {chat_id}")
            st.caption("No documents uploaded yet.")
        else:
            logger.info(f"Displaying {len(files)} uploaded files for chat {chat_id}")
            for f in files:
                st.markdown(
                    f"📄 **{f.get('file_name', '')}**  \n"
                    f"`{f.get('file_type', 'unknown')}` • {format_file_size(f.get('file_size'))}",
                )


def render_messages(messages):
    logger.debug(f"Rendering {len(messages)} messages")
    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


def main():
    logger.info("Starting RAG Chat application")
    st.set_page_config(page_title="RAG Chat", layout="wide")
    init_state()

    if not cached_health():
        logger.error("Backend health check failed")
        st.error("Backend not reachable")
        return

    logger.info("Backend is healthy, loading chats")
    chats = cached_fetch_chats() or []
    ensure_default_chat_selected(chats)

    render_sidebar(chats)

    chat_id = st.session_state.chat_id
    if not chat_id:
        logger.warning("No chat selected, showing create chat message")
        st.info("Create a chat to begin")
        return

    selected_chat = next((c for c in chats if c["id"] == chat_id), None)
    chat_name = selected_chat.get("chat_name", "Chat") if selected_chat else "Chat"
    logger.info(f"Active chat: {chat_name} (ID: {chat_id})")
    st.header(chat_name)

    load_messages_once(chat_id)

    st.divider()
    render_documents_panel(chat_id)
    st.divider()

    messages = st.session_state.messages_by_chat[chat_id]
    render_messages(messages)

    prompt = st.chat_input("Message")
    if not prompt:
        return

    logger.info(f"User submitted query to chat {chat_id}")
    logger.debug(f"Query: {prompt[:100]}...")
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            logger.info(f"Processing query for chat {chat_id}")
            response = ask_question(chat_id, prompt) or {}
            answer = response.get("answer") or "⚠️ No answer returned"

        typewriter(answer, speed=0.000012)

        docs = response.get("retrieved_docs", []) or []
        logger.info(f"Retrieved {len(docs)} documents for answer")
        if docs:
            with st.expander("Sources"):
                for d in docs:
                    meta = d.get("metadata", {}) or {}
                    st.markdown(
                        f"**{meta.get('file_name','')}** • chunk {meta.get('chunk_id')} • score {meta.get('score')}\n\n"
                        f"{d.get('content','')}"
                    )

    messages.append({"role": "assistant", "content": answer})
    logger.info(f"Message exchange completed for chat {chat_id}")


if __name__ == "__main__":
    main()