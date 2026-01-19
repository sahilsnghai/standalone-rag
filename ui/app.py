import streamlit as st
import hashlib
import time
import requests

from utils.logger import get_logger

logger = get_logger()
API_URL = "http://localhost:8000"


def file_fingerprint(uploaded) -> str:
    data = uploaded.getvalue()
    h = hashlib.sha256(data).hexdigest()
    return f"{uploaded.name}:{len(data)}:{h}"


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


@st.cache_data(ttl=1.5)
def cached_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


@st.cache_data(ttl=1.5)
def cached_fetch_chats():
    r = requests.get(f"{API_URL}/chats", timeout=5)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=1.5)
def cached_fetch_history(chat_id: str):
    r = requests.get(f"{API_URL}/chats/{chat_id}/history", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=1.5)
def cached_fetch_uploaded_files(chat_id: str):
    r = requests.get(f"{API_URL}/chats/{chat_id}/files", timeout=10)
    if r.status_code == 200:
        return r.json()
    return []


def create_chat(chat_name: str):
    try:
        result = requests.post(
            f"{API_URL}/chats",
            params={"chat_name": chat_name},
            timeout=10,
        )
        result.raise_for_status()
        st.cache_data.clear()
        return result.json()
    except Exception as e:
        st.error(f"Backend error: {e}")
        return None


def delete_chat(chat_id: str):
    try:
        result = requests.delete(f"{API_URL}/chats/{chat_id}", timeout=10)
        result.raise_for_status()
        st.cache_data.clear()
        st.success("Chat deleted")
    except Exception as e:
        st.error(f"Error deleting chat: {e}")


def upload_file(chat_id: str, file):
    try:
        files = {"file": (file.name, file.getvalue())}
        result = requests.post(f"{API_URL}/upload/{chat_id}", files=files, timeout=300)
        result.raise_for_status()
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Backend error: {e}")


def ask_question(chat_id: str, query: str):
    try:
        result = requests.post(
            f"{API_URL}/query",
            json={"chat_id": chat_id, "query": query},
            timeout=300,
        )
        result.raise_for_status()
        st.cache_data.clear()
        return result.json()

        # Demo / mock response:
        return 
    except Exception as e:
        st.error(f"Backend error: {e}")
        return {}


def typewriter(text: str, speed: float = 0.015):
    box = st.empty()
    out = ""
    for ch in text:
        out += ch
        box.markdown(out)
        time.sleep(speed)


def init_state():
    st.session_state.setdefault("chat_id", None)
    st.session_state.setdefault("uploaded_fingerprints", set())
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("messages_by_chat", {}) 


def ensure_default_chat_selected(chats):
    if st.session_state.chat_id is None:
        st.session_state.chat_id = chats[0]["id"] if chats else None


def load_messages_once(chat_id: str):
    """
    Load backend history only once per chat into session state.
    Prevents duplicated rendering and removes the need for a separate 'messages' list.
    """
    if chat_id not in st.session_state.messages_by_chat:
        history = cached_fetch_history(chat_id) or []
        st.session_state.messages_by_chat[chat_id] = [
            {"role": m.get("role", "assistant"), "content": m.get("content", "")}
            for m in history
        ]


def render_sidebar(chats):
    st.sidebar.title("💬 Chats")

    chat_name_input = st.sidebar.text_input(
        "Chat Name:",
        value=st.session_state.get("new_chat_name", "New Chat"),
        key="new_chat_name_input",
    )

    if st.sidebar.button("New Chat", icon="➕", use_container_width=True):
        chat = create_chat(chat_name_input)
        if chat:
            st.session_state.chat_id = chat["id"]
            st.session_state.messages_by_chat.pop(chat["id"], None)
            st.rerun()

    st.sidebar.divider()

    if not chats:
        st.sidebar.caption("No chats yet.")
        return

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
                st.session_state.chat_id = chat_id
                st.rerun()

        with col2:
            if st.button("", icon="🗑️", key=f"delete-{chat_id}"):
                delete_chat(chat_id)
                st.session_state.messages_by_chat.pop(chat_id, None)

                if st.session_state.chat_id == chat_id:
                    remaining = [c for c in chats if c["id"] != chat_id]
                    st.session_state.chat_id = remaining[0]["id"] if remaining else None

                st.rerun()


def render_documents_panel(chat_id: str):
    with st.container():
        left, right = st.columns([1, 3], vertical_alignment="center")

        with left:
            st.subheader("📎 Documents")

        with right:
            uploader_key = f"file_uploader_{st.session_state.uploader_key}"
            uploaded = st.file_uploader(
                "Upload document",
                label_visibility="collapsed",
                type=["txt", "pdf", "docx"],
                key=uploader_key,
            )

            if uploaded:
                fp = file_fingerprint(uploaded)
                if fp not in st.session_state.uploaded_fingerprints:
                    st.session_state.uploaded_fingerprints.add(fp)
                    upload_file(chat_id, uploaded)
                    st.success(f"Uploaded {uploaded.name}")

                st.session_state.uploader_key += 1
                st.rerun()

        with st.expander("View uploaded files", expanded=False):
            files = cached_fetch_uploaded_files(chat_id) or []
            if not files:
                st.caption("No documents uploaded yet.")
            else:
                for f in files:
                    st.markdown(
                        f"- **{f.get('file_name','')}**  \n"
                        f"<small>{f.get('file_type','unknown')} • {format_file_size(f.get('file_size'))}</small>",
                        unsafe_allow_html=True,
                    )


def render_messages(messages):
    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


def main():
    st.set_page_config(page_title="RAG Chat", layout="wide")
    init_state()

    if not cached_health():
        st.error("Backend not reachable")
        return

    chats = cached_fetch_chats() or []
    ensure_default_chat_selected(chats)

    render_sidebar(chats)

    chat_id = st.session_state.chat_id
    if not chat_id:
        st.info("Create a chat to begin")
        return

    selected_chat = next((c for c in chats if c["id"] == chat_id), None)
    st.header(selected_chat.get("chat_name", "Chat") if selected_chat else "Chat")

    load_messages_once(chat_id)

    st.divider()
    render_documents_panel(chat_id)
    st.divider()

    messages = st.session_state.messages_by_chat[chat_id]
    render_messages(messages)

    prompt = st.chat_input("Message")
    if not prompt:
        return

    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_question(chat_id, prompt) or {}
            answer = response.get("answer") or "⚠️ No answer returned"

        typewriter(answer, speed=0.000012)

        docs = response.get("retrieved_docs", []) or []
        if docs:
            with st.expander("Sources"):
                for d in docs:
                    meta = d.get("metadata", {}) or {}
                    st.markdown(
                        f"**{meta.get('file_name','')}** • chunk {meta.get('chunk_id')} • score {meta.get('score')}\n\n"
                        f"{d.get('content','')}"
                    )

    messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
