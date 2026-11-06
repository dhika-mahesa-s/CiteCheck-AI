"""Bangun knowledge base FAISS dari folder PDF referensi (jalankan SEKALI di komputer Anda).

Pemakaian:
    set GOOGLE_API_KEY=...   (PowerShell: $env:GOOGLE_API_KEY="...")
    set GROQ_API_KEY=...
    python build_index.py "D:\\path\\folder_pdf_referensi"

Hasilnya folder `faiss_index/` (index.faiss + index.pkl). Commit folder itu ke GitHub.
Bisa dilanjutkan (resume) jika terhenti karena rate limit.
"""
import sys
import time
from pathlib import Path

import core

OUT = Path(__file__).parent / "faiss_index"


def main(folder):
    pdfs = sorted(Path(folder).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"Tidak ada PDF di {folder}")
    embeddings = core.make_embeddings()
    llm = core.make_llm(core.METADATA_MODEL)

    chunks = []
    for pdf in pdfs:
        data = pdf.read_bytes()
        meta = core.extract_metadata(llm, core.first_page_text(data), pdf.name)
        print(f"{pdf.name} -> {meta['author']} ({meta['year']})")
        chunks += core.chunk_reference(core.pdf_text(data), meta)
        time.sleep(3)  # aman dari rate limit Groq
    print(f"Total {len(chunks)} chunks")

    store = core.load_base_store(OUT, embeddings)
    if store is not None:
        done = store.index.ntotal
        print(f"Melanjutkan: {done} chunk sudah ada")
        chunks = chunks[done:]

    core.add_documents_in_batches(
        store, chunks, embeddings, batch_size=30, pause=15, backoff=60, max_retries=10,
        on_progress=lambda i, n: print(f"  embedded {i}/{n}"), save_path=OUT,
    )
    print(f"Selesai. Index tersimpan di {OUT}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])