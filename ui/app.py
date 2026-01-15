# import streamlit as st
# import requests
# import hashlib
# import time

# API_URL = "http://localhost:8000"


# def file_fingerprint(uploaded) -> str:
#     data = uploaded.getvalue()
#     h = hashlib.sha256(data).hexdigest()
#     return f"{uploaded.name}:{len(data)}:{h}"


# def format_file_size(size):
#     if not size:
#         return ""
#     size = float(size)
#     if size < 1024:
#         return f"{size:.0f} B"
#     elif size < 1024**2:
#         return f"{size / 1024:.2f} KB"
#     else:
#         return f"{size / (1024 ** 2):.2f} MB"


# @st.cache_data(ttl=1.5)
# def cached_health() -> bool:
#     try:
#         r = requests.get(f"{API_URL}/health", timeout=3)
#         return r.status_code == 200
#     except requests.RequestException:
#         return False


# @st.cache_data(ttl=1.5)
# def cached_fetch_chats():
#     r = requests.get(f"{API_URL}/chats", timeout=5)
#     r.raise_for_status()
#     return r.json()


# @st.cache_data(ttl=1.5)
# def cached_fetch_history(chat_id: str):
#     r = requests.get(f"{API_URL}/chats/{chat_id}/history", timeout=10)
#     r.raise_for_status()
#     return r.json()


# @st.cache_data(ttl=1.5)
# def cached_fetch_uploaded_files(chat_id: str):
#     r = requests.get(f"{API_URL}/chats/{chat_id}/files", timeout=10)
#     if r.status_code == 200:
#         return r.json()
#     return []


# def create_chat(chat_name: str):
#     try:
#         result = requests.post(
#             f"{API_URL}/chats",
#             params={"chat_name": chat_name},
#             timeout=10,
#         )
#         result.raise_for_status()
#         st.cache_data.clear()
#         return result.json()
#     except Exception as e:
#         st.error(f"Backend error: {e}")


# def delete_chat(chat_id: str):
#     try:
#         result = requests.delete(f"{API_URL}/chats/{chat_id}", timeout=10)
#         result.raise_for_status()
#         st.cache_data.clear()
#         st.success("Chat deleted")
#     except Exception as e:
#         st.error(f"Error deleting chat: {e}")


# def upload_file(chat_id: str, file):
#     try:
#         files = {"file": (file.name, file.getvalue())}
#         result = requests.post(f"{API_URL}/upload/{chat_id}", files=files, timeout=300)
#         result.raise_for_status()
#         st.cache_data.clear()
#     except Exception as e:
#         st.error(f"Backend error: {e}")


# def ask_question(chat_id: str, query: str):
#     try:
#         # result = requests.post(
#         #     f"{API_URL}/query",
#         #     json={"chat_id": chat_id, "query": query},
#         #     timeout=300,
#         # )
#         # result.raise_for_status()
#         # st.cache_data.clear()
#         # r = result.json()
#         r = {
#             "answer": "Data Scientist",
#             "retrieved_docs": [
#                 {
#                     "content": "Associate System Engineer\n10/2022 – 01/2024 Pune, Maharashtra\nA leading consulting and services company\n•Engineered and preprocessed large retail datasets, improving model accuracy by 10%.\n¢\n©\n•Developed a customer churn prediction model with 92% accuracy and conducted exploratory data analysis\n(EDA) to inform promotional strategies.\n•Built RESTful APIs with FastAPI for seamless integration and model deployment.\nNETLINK SOFTWARE GROUP PVT.LTD\nIntern\n12/2021 – 05/2022 Bhopal, Madhya Pradesh\nA software group offering solutions in data science and AI\n¢\n•Developed REST APIs using the Falcon framework for CRUD applications\n¢\n•Explored concepts in Linear and Non-Linear optimization\n•Created Machine Learning models for predictive analysis and classification\n¢\nPROJECTS\nNLP For Ayurvedic Consultancy\n2\n•Extracted the knowledge graph on Ayurveda articles with multilingual & multimodal mediums leveraging Langchain integration\n*",
#                     "metadata": {
#                         "file_name": "SAURABH-RATHORE-DS-Resume  .pdf",
#                         "chunk_id": 3,
#                         "score": 0.7807477,
#                         "id": "46bb2995-777d-4fe6-98a8-a6e5715be8f6",
#                     },
#                 },
#                 {
#                     "content": "EXPERIENCE\nBOOTLABS TECHNOLOGY PVT.LTD\nAI/ML Engineer\n04/2025 – Present Hyderabad, Telangana\n•Delivered a GenAI-driven solution for ICICI Bank to automate bank guarantee and financial document processing, enhancing operational efficiency, legal accuracy, and contract management workflows.\n¢\n•Handled over 6,000 complex documents with highly unstructured formats, including structured/unstructured\ntables and embedded images, leveraging OCR-based text extraction for dynamic document understanding.\n•Developed a RAG-based chatbot with agentic flow using LangChain, LangGraph, LLAMA 3.3, BGE-Large, and Qdrant, achieving 80% accuracy in audit summarization and significantly reducing manual review effort.\nNETLINK SOFTWARE GROUP PVT.LTD\nData Scientist\n01/2024 – 03/2025 Bhopal, Madhya Pradesh\nA software group offering solutions in data science and AI\n•Specialized in Prompt Engineering and Model Fine-Tuning to enhance LLM performance, accuracy, and\ncontextual understanding.",
#                     "metadata": {
#                         "file_name": "SAURABH-RATHORE-DS-Resume  .pdf",
#                         "chunk_id": 1,
#                         "score": 0.76435024,
#                         "id": "0e6d375b-7bcc-4eab-92ce-bed46ca4eba3",
#                     },
#                 },
#                 {
#                     "content": "A software group offering solutions in data science and AI\n•Specialized in Prompt Engineering and Model Fine-Tuning to enhance LLM performance, accuracy, and\ncontextual understanding.\n•Developed Text-to-SQL and GenAI-driven backend solutions, ensuring robustness, scalability, and reliability\nacross production environments.\n•Built Lumenore AI, a system that transforms natural language queries into actionable insights by connecting to multiple data sources, interpreting user intent via LLMs, and generating SQL queries to deliver intelligent visual summaries through a microservice-based architecture.\n•Researched and implemented GenAI advancements, performing iterative testing and continuous model optimization based on user feedback and evolving requirements.\nTATA CONSULTANCY AND SERVICES (TCS)\nAssociate System Engineer\n10/2022 – 01/2024 Pune, Maharashtra\nA leading consulting and services company\n•Engineered and preprocessed large retail datasets, improving model accuracy by 10%.\n¢\n©",
#                     "metadata": {
#                         "file_name": "SAURABH-RATHORE-DS-Resume  .pdf",
#                         "chunk_id": 2,
#                         "score": 0.717936,
#                         "id": "96822bbd-e121-4152-a71a-a96530200daa",
#                     },
#                 },
#                 {
#                     "content": "SAURABH RATHORE AI/ML Engineer\nsaurabhrathore64@gmail.com\n.\n+91-7772832102\n@\nHyderabad, Telangana\n[J\nLinkedIn\n€\nLeetcode\n©)\nGitHub\nSUMMARY\nWorking with ICICI Bank on GenAI solutions, with 3.5+ years of experience in building scalable backend systems and developing chatbots using RAG and prompt engineering. Passionate about applying machine learning to solve complex problems and improve user experience.\nSKILLS\nLanguages: — Python Technical Skills / Concepts: — Machine Learning, Deep Learning, NLP, Generative AI, Prompt Engineering, Fine- tuning, Transformers, RAG , Agentic AI, Model Context Protocol (MCP), OCR Databases: — MySQL, Redis, Qdrant-client and FAISS Frameworks / Libraries: — FastAPI, LangChain, LangGraph, NLTK, spaCy, PyTorch, TensorFlow Others / Soft Skills: — Problem Solving, Data Analysis, AI/ML Pipeline Design Cloud / Platforms: — AWS (EC2, S3, Lambda, RDS, IAM, CloudWatch)\nEXPERIENCE\nBOOTLABS TECHNOLOGY PVT.LTD\nAI/ML Engineer\n04/2025 – Present Hyderabad, Telangana",
#                     "metadata": {
#                         "file_name": "SAURABH-RATHORE-DS-Resume  .pdf",
#                         "chunk_id": 0,
#                         "score": 0.6970748,
#                         "id": "dbea42b4-5752-4b44-aa72-31f5d960cf8e",
#                     },
#                 },
#                 {
#                     "content": "¢\nPROJECTS\nNLP For Ayurvedic Consultancy\n2\n•Extracted the knowledge graph on Ayurveda articles with multilingual & multimodal mediums leveraging Langchain integration\n*\n•Built a RAG system for Q&A agent of Ayurvedic research & disease with LLMs, FAISS, LightRAG achieving 83% Answer Relevancy\n*\nSmartSeg: Targeted Marketing Intelligence for Banking\n•Extracted key customer segments using clustering and dimensionality reduction on demographic and behavioral data.\n•Achieved 90% accuracy with the classification model, demonstrating strong predictive power for labeled marketing outcomes.\n*\n*\n•Uncovered actionable customer segments using clustering with an accuracy of 70%, supporting moderate structure for targeted strategies.\nEDUCATION\nMadhav Institute of Technology and Science B.Tech\n08/2018 – 06/2022 Gwalior, M.P",
#                     "metadata": {
#                         "file_name": "SAURABH-RATHORE-DS-Resume  .pdf",
#                         "chunk_id": 4,
#                         "score": 0.71580267,
#                         "id": "9df5151d-6946-41e7-b9fa-d5e51bb73d78",
#                     },
#                 },
#             ],
#             "evaluation": {
#                 "query": "what was my position in netlink software group",
#                 "retrieval_chunks": 5,
#                 "average_score": None,
#                 "latency": 6.519371509552002,
#                 "faithfulness": False,
#                 "completeness": False,
#             },
#         }
#         return r
#     except Exception as e:
#         st.error(f"Backend error: {e}")

# def typewriter(text: str, speed: float = 0.015):
#     box = st.empty()
#     out = ""
#     for ch in text:
#         out += ch
#         box.markdown(out)
#         time.sleep(speed)


# def main():
#     st.set_page_config(page_title="RAG Chat", layout="wide")

#     if "chat_id" not in st.session_state:
#         st.session_state.chat_id = None
#     if "uploaded_fingerprints" not in st.session_state:
#         st.session_state.uploaded_fingerprints = set()
#     if "uploader_key" not in st.session_state:
#         st.session_state.uploader_key = 0

#     if not cached_health():
#         st.error("Backend not reachable")
#         return

#     st.sidebar.title("💬 Chats")
#     chats = cached_fetch_chats()

#     if st.session_state.chat_id is None:
#         st.session_state.chat_id = chats[0]["id"] if chats else None

#     chat_name_input = st.sidebar.text_input(
#         "Chat Name:",
#         value=st.session_state.get("new_chat_name", "New Chat"),
#         key="new_chat_name_input",
#     )

#     if st.sidebar.button("New Chat", icon="➕", use_container_width=True):
#         chat = create_chat(chat_name_input)
#         if chat:
#             st.session_state.chat_id = chat["id"]
#             st.rerun()

#     st.sidebar.divider()

#     if not chats:
#         st.sidebar.caption("No chats yet.")
#     else:
#         for chat in chats:
#             is_selected = chat["id"] == st.session_state.chat_id

#             col1, col2 = st.sidebar.columns([8, 1], vertical_alignment="center")

#             label = chat["chat_name"]
#             if chat.get("summary"):
#                 label += f" — {chat['summary'][:40]}…"

#             with col1:
#                 if st.button(
#                     label,
#                     key=f"chat-{chat['id']}",
#                     use_container_width=True,
#                     type="primary" if is_selected else "secondary",
#                 ):
#                     st.session_state.chat_id = chat["id"]
#                     st.rerun()

#             with col2:
#                 if st.button("", icon="🗑️", key=f"delete-{chat['id']}"):
#                     delete_chat(chat["id"])
#                     if st.session_state.chat_id == chat["id"]:
#                         remaining = [c for c in chats if c["id"] != chat["id"]]
#                         st.session_state.chat_id = (
#                             remaining[0]["id"] if remaining else None
#                         )
#                     st.rerun()

#     chat_id = st.session_state.chat_id
#     if not chat_id:
#         st.info("Create a chat to begin")
#         return

#     selected_chat = next((c for c in chats if c["id"] == chat_id), None)
#     st.header(selected_chat["chat_name"] if selected_chat else "Chat")

#     history = cached_fetch_history(chat_id)
#     for msg in history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     st.divider()

#     with st.container():
#         left, right = st.columns([1, 3], vertical_alignment="center")

#         with left:
#             st.subheader("📎 Documents")

#         with right:
#             uploader_key = f"file_uploader_{st.session_state.uploader_key}"
#             uploaded = st.file_uploader(
#                 "Upload document",
#                 label_visibility="collapsed",
#                 type=["txt", "pdf", "docx"],
#                 key=uploader_key,
#             )

#             if uploaded:
#                 fp = file_fingerprint(uploaded)
#                 if fp not in st.session_state.uploaded_fingerprints:
#                     st.session_state.uploaded_fingerprints.add(fp)
#                     upload_file(chat_id, uploaded)
#                     st.success(f"Uploaded {uploaded.name}")

#                 st.session_state.uploader_key += 1
#                 st.rerun()

#         with st.expander("View uploaded files", expanded=False):
#             files = cached_fetch_uploaded_files(chat_id)
#             if not files:
#                 st.caption("No documents uploaded yet.")
#             else:
#                 for f in files:
#                     st.markdown(
#                         f"- **{f['file_name']}**  \n"
#                         f"<small>{f.get('file_type','unknown')} • {format_file_size(f.get('file_size'))}</small>",
#                         unsafe_allow_html=True,
#                     )

#     st.divider()

#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # render history
#     for m in st.session_state.messages:
#         with st.chat_message(m["role"]):
#             st.markdown(m["content"])

#     prompt = st.chat_input("Message")

#     if prompt:
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 response = ask_question(chat_id, prompt)
#                 answer = response.get("answer", "") or "⚠️ No answer returned"

#             # smooth animation
#             typewriter(answer, speed=0.01)

#             docs = response.get("retrieved_docs", [])
#             if docs:
#                 with st.expander("Sources"):
#                     for d in docs:
#                         meta = d.get("metadata", {})
#                         st.markdown(
#                             f"**{meta.get('file_name','')}** • chunk {meta.get('chunk_id')} • score {meta.get('score')}\n\n"
#                             f"{d.get('content','')}"
#                         )

#         st.session_state.messages.append({"role": "assistant", "content": answer})


# if __name__ == "__main__":
#     main()


import hashlib
import time

import requests
import streamlit as st

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
