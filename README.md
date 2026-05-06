# 📚 Bookiee AI - Prototype

> An AI-powered study tool that helps you understand, summarize, and study any document — built with Python, Streamlit and API.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Claude](https://img.shields.io/badge/Powered%20by-Claude%20AI-2ea043?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## ✨ Features

| Tab | What it does |
|-----|-------------|
| 📄 **Analyze** | Generates a summary, extracts key points, and identifies important concepts |
| 💬 **Ask Questions** | Chat with your document — ask anything, get context-aware answers |
| 🧠 **Study Mode** | Generate quizzes, interactive flashcards, or a simplified explanation |

**Supports:** `.pdf`, `.txt`, `.md` files — or paste text directly.

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/Bookiee-Prototype.git
cd Bookiee-Prototype
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key
Create a file at `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```
Get a key at [console.anthropic.com](https://console.anthropic.com).

### 4. Run the app
```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Point it to `app.py`
4. In **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Click **Deploy** — live in ~60 seconds

---

## 🏗️ Project Structure

```
Bookiee-Prototype/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore          # Keeps secrets out of git
└── README.md           # This file
```

---

## 🧠 How It Works

All features use the same core pattern: extract text from the document, then send it to Claude with a purpose-built prompt.

```
User uploads file / pastes text
        ↓
Text extracted (pdfplumber for PDFs)
        ↓
Stored in Streamlit session state
        ↓
User action → prompt built with context
        ↓
Claude API call → structured response
        ↓
Rendered in the UI
```

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key (required) |

---

## 📄 License

MIT — use it, fork it, build on it.

---

Built by **[Your Name]** · [Portfolio](https://yoursite.com) · [LinkedIn](https://linkedin.com)
