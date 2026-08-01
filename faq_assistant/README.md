# 🤖 FAQ AI Assistant

A minimal, clean, basic, and user-friendly FAQ AI Assistant application built from scratch with **Python Flask**, **Scikit-Learn (TF-IDF Vector Search)**, and **Google Gemini 2.0 Flash AI**.

---

## ✨ Features

- ⚡ **Instant Local FAQ Matching**: Uses TF-IDF vectorization and cosine similarity for sub-millisecond local search.
- 🧠 **Google Gemini 2.0 Flash Integration**: Generates answers for custom questions using Google's generative AI.
- 🛡️ **Offline & Quota-Resilient Fallback**: Automatically falls back to closest dataset matches if AI keys are missing or quota-limited.
- 🎨 **Minimal & User-Friendly UI**: Clean typography, crisp message bubbles, quick suggestion chips, and zero clutter.

---

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key (Optional)**:
   Add your free Gemini API key to `.env`:
   ```env
   GEMINI_API_KEY="AIzaSy..."
   ```

3. **Run the Assistant**:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your web browser!
