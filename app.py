"""CiteCheck AI - Agentic RAG Citation Verification Assistant (Streamlit)."""
import os
import threading
import uuid
from datetime import date
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="CiteCheck AI", page_icon="🔎", layout="wide")

# Streamlit Secrets -> environment variables (used by the LangChain clients)
for _k in ("GOOGLE_API_KEY", "GROQ_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"):
    try:
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
    except Exception:
        pass

import core  # noqa: E402  (after env is populated)

BASE_INDEX_PATH = Path(__file__).parent / "faiss_index"
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "150"))   # chat turns per day, whole app
CHAT_LIMIT = int(os.getenv("CHAT_LIMIT", "20"))      # chat turns per session
MAX_REF_FILES = 3
MAX_SESSION_CHUNKS = int(os.getenv("MAX_SESSION_CHUNKS", "200"))
MAX_FILE_MB = 15
EXAMPLES = [
    "Apakah kalimat pada draft skripsi saya tentang tinggi gelombang itu relevan dengan paper yang ada?",
    "Apakah klaim ini relevan: kecepatan angin tidak berpengaruh sama sekali dalam keputusan melaut nelayan",
]


@st.cache_resource
def usage_counter():
    return {"lock": threading.Lock(), "day": date.today(), "count": 0}


def check_quota():
    u = usage_counter()
    with u["lock"]:
        if u["day"] != date.today():
            u["day"], u["count"] = date.today(), 0
        if u["count"] >= DAILY_LIMIT:
            raise RuntimeError("Kuota demo harian sudah habis. Silakan coba lagi besok.")
        u["count"] += 1


@st.cache_resource(show_spinner="Memuat knowledge base dan agen...")
def get_engine():
    embeddings = core.make_embeddings()
    base = core.load_base_store(BASE_INDEX_PATH, embeddings)
    retriever = core.Retriever(base, embeddings)
    graph = core.build_graph(retriever, core.make_llm(core.MAIN_MODEL))
    return retriever, graph, core.make_llm(core.METADATA_MODEL)


missing = [k for k in ("GOOGLE_API_KEY", "GROQ_API_KEY") if not os.environ.get(k)]
if missing:
    st.error(f"Secrets belum lengkap: {', '.join(missing)}. Isi di Settings → Secrets.")
    st.stop()

retriever, graph, metadata_llm = get_engine()

ss = st.session_state
ss.setdefault("sid", f"sesi_{uuid.uuid4().hex[:8]}")
ss.setdefault("messages", [])
ss.setdefault("chat_count", 0)
ss.setdefault("draft_name", None)
ss.setdefault("draft_file_text", "")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    st.caption(
        f"Knowledge base bawaan: **{retriever.base_size}** potongan teks"
        if retriever.base_size else "Knowledge base bawaan: belum ada (unggah paper referensi di bawah)."
    )

    st.subheader("📄 1. Draft skripsi")
    draft = st.file_uploader("Unggah draft (PDF)", type=["pdf"], key="draft_uploader")
    if draft is not None and ss.draft_name != draft.name:
        try:
            if draft.size > MAX_FILE_MB * 1024 * 1024:
                raise ValueError(f"Ukuran file maksimal {MAX_FILE_MB} MB.")
            ss.draft_file_text = core.pdf_text(draft.getvalue())[:200_000]
            ss.draft_name = draft.name
            if not ss.draft_file_text.strip():
                st.warning("PDF tidak berisi teks (kemungkinan hasil scan).")
        except Exception as e:
            st.error(f"Gagal membaca draft: {e}")
    pasted = st.text_area("Atau tempel teks draft", height=120, key="draft_paste")
    draft_text = pasted.strip() or ss.draft_file_text
    if draft_text:
        st.success(f"Draft siap ({len(draft_text):,} karakter).")

    st.subheader("📚 2. Tambah paper referensi")
    st.caption(f"Maks {MAX_REF_FILES} PDF. Hanya berlaku untuk sesi Anda dan hilang saat halaman ditutup.")
    refs = st.file_uploader("Unggah paper (PDF)", type=["pdf"], accept_multiple_files=True, key="ref_uploader")
    if st.button("Embed ke knowledge base sesi ini", disabled=not refs):
        files = [(f.name, f.getvalue()) for f in refs if f.size <= MAX_FILE_MB * 1024 * 1024]
        try:
            with st.status("Memproses paper...", expanded=True) as status:
                bar = st.progress(0.0)
                added, notes = core.ingest_reference_pdfs(
                    files, retriever, metadata_llm, ss.sid, MAX_REF_FILES, MAX_SESSION_CHUNKS,
                    on_progress=lambda i, n: bar.progress(i / n),
                )
                status.update(label=f"Selesai: {added} potongan teks ditambahkan", state="complete")
            for n in notes:
                st.warning(n)
        except Exception as e:
            st.error(f"Gagal memproses paper: {e}")
    if retriever.session_size(ss.sid):
        st.caption(f"Referensi sesi Anda: {retriever.session_size(ss.sid)} potongan teks")

    if st.button("🗑️ Mulai percakapan baru"):
        ss.messages, ss.chat_count = [], 0
        ss.sid = f"sesi_{uuid.uuid4().hex[:8]}"  # new thread = new memory + empty session KB
        st.rerun()

# ------------------------------------------------------------------ main
st.title("🔎 CiteCheck AI")
st.caption("Asisten verifikasi sitasi & klaim skripsi: cek apakah klaim Anda didukung paper referensi, "
           "dan cari paper alternatif lewat Semantic Scholar bila tidak.")
st.info("Demo portofolio. Jawaban AI dapat keliru, selalu periksa paper aslinya. "
        "Jangan unggah dokumen yang bersifat rahasia.")

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

picked = None
if not ss.messages:
    st.write("**Contoh pertanyaan:**")
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key=f"ex_{i}"):
            picked = ex

prompt = st.chat_input("Tanyakan: 'Apakah kalimat saya tentang [topik] relevan?'") or picked
if prompt:
    if ss.chat_count >= CHAT_LIMIT:
        st.warning("Batas pesan sesi ini tercapai. Klik 'Mulai percakapan baru' di sidebar.")
    else:
        try:
            check_quota()
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Mencari bukti di paper referensi..."):
                    result = graph.invoke(
                        {"draft_text": draft_text, "user_query": prompt},
                        config={"configurable": {"thread_id": ss.sid}},
                    )
                answer = result["final_response"]
                st.markdown(answer)
            ss.messages += [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
            ss.chat_count += 1
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")