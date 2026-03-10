# 🕉️ GitaGPT — Ask Lord Krishna Anything

A spiritual AI chatbot powered by the Bhagavad Gita. Ask questions about life, dharma, career, or relationships and receive guidance in the voice of Lord Krishna — grounded in actual Gita verses using RAG.

---

## ✨ Features

- 🪷 **Krishna Persona** — Responses styled as Lord Krishna speaking directly to you
- 📖 **RAG-Powered** — Answers retrieved from 700+ Bhagavad Gita verses, not hallucinated
- 💬 **Conversational Chat** — Full multi-turn dialogue with chat history
- 📄 **PDF Transcript** — Download your entire conversation as a PDF
- 🎨 **Spiritual UI** — Dark, minimal design inspired by [gitagpt.in](https://www.gitagpt.in)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector DB | ChromaDB |
| RAG Framework | LangChain |
| PDF Generation | fpdf2 |

---

## 🚀 Local Setup

**1. Clone the repo**
```bash
git clone https://github.com/Saurya29/GitaGPT.git
cd GitaGPT
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
.\venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your API key**

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

**5. Ensure required files are present**
```
GitaGPT/
├── app.py
├── gita_book.pdf       ← Bhagavad Gita source text
├── krishna_ji.jpeg     ← Hero background image
├── requirements.txt
└── .env
```

**6. Run the app**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## ☁️ Deploy on Streamlit Cloud

1. Push your repo to GitHub (exclude `.env` and `gita_chroma/`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New App**
3. Select your repo → set main file as `app.py`
4. Go to **Settings → Secrets** and add:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```
5. Click **Deploy**

---

## 📦 Requirements
```txt
streamlit>=1.36.0
python-dotenv>=1.0.0
langchain>=0.2.11
langchain-community>=0.2.11
langchain-groq>=0.1.0
langchain-chroma>=0.2.2
sentence-transformers>=2.0.0
pypdf>=4.2.0
fpdf2>=2.7.9
pysqlite3-binary
```

---

## 🧠 How It Works
```
User Question
     ↓
Embed with MiniLM
     ↓
ChromaDB similarity search → Top 4 Gita passages
     ↓
Inject passages + Krishna persona into prompt
     ↓
Groq (Llama 3.1) generates response
     ↓
Displayed as Lord Krishna's guidance
```

---

## 📁 Project Structure
```
GitaGPT/
├── app.py              # Main Streamlit application
├── gita_book.pdf       # Bhagavad Gita source (not committed if large)
├── krishna_ji.jpeg     # UI background image
├── gita_chroma/        # Auto-generated vector store (gitignored)
├── requirements.txt
├── .env                # Local secrets (gitignored)
└── .gitignore
```

---

## ⚠️ .gitignore
```
.env
gita_chroma/
__pycache__/
*.pyc
venv/
```

---

## 🙏 Credits

Built by **Saurya Raj** · B.Tech AI/ML @ BIT Mesra  
Powered by Groq, LangChain & the eternal wisdom of the Bhagavad Gita

