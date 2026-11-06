# 🎓 CiteCheck AI: Asisten Verifikasi Sitasi & Klaim Skripsi

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-State_Agent-orange)](https://python.langchain.com/v0.1/docs/langgraph/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-red)](https://www.gradio.app/)
[![LLM](https://img.shields.io/badge/LLM-Llama_3.3_70B-green)](https://groq.com/)

**CiteCheck AI** adalah sistem kecerdasan buatan berbasis *Retrieval-Augmented Generation* (RAG) dan *Agentic Workflow* (LangGraph) yang dirancang untuk membantu mahasiswa dan akademisi memverifikasi keselarasan (konsistensi) antara klaim pada draf penelitian dengan sumber *paper* yang disitasi.

Sistem ini memecahkan masalah umum dalam penulisan akademis yaitu **Citation Mismatch**—di mana penulis sering kali salah mengutip atau menggunakan referensi yang sebenarnya tidak mendukung argumen mereka.

---

## ✨ Fitur Utama

- 🔍 **Verifikasi Klaim Ketat (Strict Fact-Checking):** Mengevaluasi apakah klaim pengguna benar-benar didukung, bertentangan, atau tidak relevan dengan literatur yang diunggah.
- ✍️ **Koreksi Berbasis Bukti:** Jika klaim terdeteksi salah/bertentangan, AI secara otomatis memberikan kalimat perbaikan berdasarkan data asli dari paper.
- 🌐 **Smart Fallback via Semantic Scholar API:** Jika klaim tidak ditemukan di database internal, sistem akan menerjemahkan klaim menjadi kata kunci akademis (LLM Query Rewriter) dan mencarikan jurnal alternatif pendukung secara *real-time*.
- 🧠 **Memori Kontekstual Berkelanjutan:** Dilengkapi `MemorySaver`, memungkinkan pengguna menanyakan pertanyaan lanjutan (contoh: *"Pada paragraf mana hal itu dibahas?"*) tanpa kehilangan konteks obrolan.
- 💾 **Persisten Knowledge Base (FAISS):** Otomatis mencadangkan dan memuat ulang database vektor langsung ke Google Drive agar data referensi tidak hilang saat *runtime* habis.

---

## 🏗️ Arsitektur Sistem

Aplikasi ini dibangun menggunakan **LangGraph** untuk mengatur *state* dan *conditional routing*:

1. **Retrieve Node:** Menyaring pertanyaan dan mencari *chunk* dokumen yang relevan di FAISS Vector Database (didukung Google Generative AI Embeddings).
2. **Evaluate Node:** Llama 3.3 (via Groq) mengevaluasi klaim dengan aturan *prompt* yang sangat ketat. Memiliki algoritma *Exact Match* untuk mencegah halusinasi *over-helpful*.
3. **External Search Node (Router):** Jika evaluator mengembalikan status `NOT_RELEVANT`, alur kerja otomatis berbelok (*route*) ke pencarian eksternal. LLM merumuskan kata kunci, dan sistem memanggil Semantic Scholar API untuk memberikan rekomendasi literatur baru.

---

## 🛠️ Teknologi yang Digunakan

- **Orkestrasi AI:** LangChain & LangGraph
- **Large Language Model (LLM):** Llama-3.3-70b-versatile (via API Groq)
- **Model Embedding:** models/embedding-001 (via Google Gemini API)
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Integrasi Eksternal:** Semantic Scholar API (Pro API Key)
- **Antarmuka Pengguna (UI):** Gradio
- **Pemrosesan PDF:** PyMuPDF (`fitz`) & Pydantic (Metadata Extraction)

---

## 🚀 Panduan Instalasi & Penggunaan (Google Colab)

Aplikasi ini dirancang untuk berjalan optimal di lingkungan Google Colab.

### 1. Persyaratan Dasar (API Keys)
Anda perlu menyiapkan tiga API Key gratis berikut dan menyimpannya di fitur **Secrets** (ikon kunci di sebelah kiri layar) Google Colab:
- `GROQ_API_KEY` (Dapatkan dari console.groq.com)
- `GOOGLE_API_KEY` (Dapatkan dari Google AI Studio)
- `SEMANTIC_SCHOLAR_API_KEY` (Dopsional, dapatkan dari situs Semantic Scholar untuk bypass rate-limit)

### 2. Cara Menjalankan
1. Unggah *file* `notebook.ipynb` ke Google Colab Anda.
2. Pastikan Google Drive Anda memiliki ruang kosong untuk menyimpan database FAISS (folder akan otomatis terbuat di `/MyDrive/SkripsiAssistant/faiss_index`).
3. Tekan **Runtime > Run All** di menu Colab.
4. Gulir ke bagian paling bawah *notebook*. Klik tautan publik Gradio (berakhiran `.gradio.live`) yang muncul.

### 3. Cara Penggunaan di Antarmuka Web (UI)
1. **Upload Referensi:** Tarik dan lepas file PDF jurnal/referensi Anda ke panel *Upload PDF Referensi*. Tekan tombol *Embed & Simpan*.
2. **Upload Draft:** Tarik dan lepas draft skripsi Anda ke panel *Upload PDF Draft*.
3. **Mulai Konsultasi:** Ketik klaim Anda di kotak obrolan. Contoh:
   > *"Apakah kalimat pada draft saya mengenai efektivitas model MobileNet relevan dengan literatur?"*

---

## 📝 Contoh Skenario (Use Cases)

| Skenario | Respons Sistem | Alur LangGraph |
| :--- | :--- | :--- |
| **Klaim Tepat** | ✅ Menyatakan klaim benar + menampilkan Nama Penulis & Bukti Kutipan. | `Retrieve -> Evaluate -> End` |
| **Klaim Salah** | ❌ Mengoreksi klaim + memberikan kalimat perbaikan sesuai data jurnal. | `Retrieve -> Evaluate -> End` |
| **Klaim Tak Relevan** | ⚠️ Memberikan peringatan + menyajikan daftar paper rujukan dari Semantic Scholar. | `Retrieve -> Evaluate -> External Search -> End` |

---

## 📄 Lisensi

Didistribusikan di bawah Lisensi MIT. Lihat file `LICENSE` untuk informasi lebih lanjut.