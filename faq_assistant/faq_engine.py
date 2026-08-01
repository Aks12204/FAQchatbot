import os
import re
import json
import urllib.request
import urllib.parse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import Config

def clean_text(text):
    """Normalizes input text for accurate matching."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\(Query ID:\s*\d+\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load FAQ dataset and build TF-IDF index
faq_df = None
vectorizer = None
faq_vectors = None
cleaned_questions = []

try:
    if os.path.exists(Config.DATA_PATH):
        faq_df = pd.read_csv(Config.DATA_PATH)
        cleaned_questions = faq_df["question"].apply(clean_text).tolist()
        
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True
        )
        faq_vectors = vectorizer.fit_transform(cleaned_questions)
        print(f"✅ Loaded FAQ index with {len(faq_df)} records.")
    else:
        print(f"⚠️ FAQ dataset not found at {Config.DATA_PATH}")
except Exception as e:
    print(f"❌ Error indexing FAQ dataset: {e}")

def get_gemini_client():
    """Dynamically fetches/initializes Gemini client if valid key present."""
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or Config.GEMINI_API_KEY or "").strip().strip('"').strip("'")
    if api_key and api_key not in ["your_gemini_api_key_here", ""]:
        try:
            from google import genai
            return genai.Client(api_key=api_key)
        except Exception:
            return None
    return None

def check_conversational_intent(text):
    """Handles general conversational greetings, identity questions, and thank-yous."""
    lower = text.lower().strip()
    words = re.findall(r'\b\w+\b', lower)
    
    # Greetings
    greetings = {"hi", "hii", "hiii", "hello", "hey", "heyy", "hie", "greetings", "good morning", "good afternoon", "good evening"}
    if any(w in greetings for w in words) or lower in greetings:
        return "Hello! 👋 I'm your AI Support & FAQ Assistant. How can I help you today? You can ask me questions about orders, returns, account settings, student portal, or AWS Cloud!"

    # Identity / Capability
    if any(phrase in lower for phrase in ["who are you", "what is your name", "what can you do", "help me", "what are you"]):
        return "I am an intelligent FAQ & Support Assistant! I can help you instantly with queries regarding account management, delivery, student portal, IT support, and cloud services."

    # Gratitude
    if any(w in lower for w in ["thank you", "thanks", "thx", "awesome", "great", "perfect"]):
        return "You're very welcome! 😊 Let me know if you have any other questions."

    # Farewell
    if any(w in lower for w in ["bye", "goodbye", "see you"]):
        return "Goodbye! Have a wonderful day ahead! 👋"

    return None

def fetch_free_instant_knowledge(query):
    """
    Fetches real-time instant definitions from DuckDuckGo's Free Knowledge API.
    Zero API keys required!
    """
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            abstract = data.get("AbstractText") or data.get("Definition")
            if abstract:
                return abstract
            # Check related topics
            related = data.get("RelatedTopics", [])
            for r in related:
                if isinstance(r, dict) and r.get("Text"):
                    return r.get("Text")
    except Exception as e:
        print(f"Free knowledge lookup notice: {e}")
    return None

def synthesize_multi_match(user_question, min_threshold=0.08, top_k=3):
    """
    Finds top-K matching knowledge base entries and synthesizes a clean, multi-point response.
    Zero API keys required!
    """
    if vectorizer is None or faq_vectors is None or faq_df is None:
        return None
    cleaned_input = clean_text(user_question)
    if not cleaned_input:
        return None
        
    user_vec = vectorizer.transform([cleaned_input])
    similarities = cosine_similarity(user_vec, faq_vectors).flatten()
    
    # Get indices sorted by similarity score descending
    top_indices = similarities.argsort()[::-1]
    matches = []
    
    for idx in top_indices:
        score = float(similarities[idx])
        if score >= min_threshold:
            matches.append({
                "question": cleaned_questions[idx],
                "answer": faq_df.iloc[idx]["answer"],
                "confidence": round(score * 100, 1)
            })
        if len(matches) >= top_k:
            break
            
    if not matches:
        return None
        
    if len(matches) == 1:
        return matches[0]["answer"]
        
    # Synthesize multi-point response
    response_lines = ["Here is what I found in our support knowledge base:"]
    for m in matches:
        response_lines.append(f"• **{m['question']}**: {m['answer']}")
        
    return "\n\n".join(response_lines)

def process_query(user_question, threshold=0.35):
    """
    Zero-API-Key Intelligent Query Pipeline:
    1. Conversational intent (Greetings, Gratitude, Identity)
    2. High-confidence TF-IDF local FAQ search
    3. Live Gemini 2.0 Flash AI generation (if valid API key present)
    4. Free Instant Web Knowledge API (DuckDuckGo 0-key search)
    5. Smart Multi-Match Knowledge Synthesis (Offline 0-key search)
    """
    cleaned_input = clean_text(user_question)
    if not cleaned_input:
        return {"answer": "Please ask a question.", "source": "error", "confidence": 0}

    # Step 1: Check Conversational Intent (Greetings, etc.)
    conv_response = check_conversational_intent(cleaned_input)
    if conv_response:
        return {
            "answer": conv_response,
            "source": "assistant",
            "confidence": 100.0
        }

    # Step 2: TF-IDF Cosine Similarity Search in Dataset
    if vectorizer is not None and faq_vectors is not None and faq_df is not None:
        user_vec = vectorizer.transform([cleaned_input])
        similarities = cosine_similarity(user_vec, faq_vectors).flatten()
        max_score = float(similarities.max())

        if max_score >= threshold:
            best_idx = similarities.argmax()
            return {
                "answer": faq_df.iloc[best_idx]["answer"],
                "matched_question": cleaned_questions[best_idx],
                "source": "faq",
                "confidence": round(max_score * 100, 1)
            }

    # Step 3: Try Gemini AI Generation (if valid key configured)
    gemini_client = get_gemini_client()
    if gemini_client is not None:
        try:
            response = gemini_client.models.generate_content(
                model='gemini-2.0-flash',
                contents=f"You are a helpful, friendly customer support AI assistant. Answer this question concisely and clearly: {user_question}"
            )
            return {
                "answer": response.text.strip(),
                "source": "ai (Gemini 2.0)",
                "confidence": 95.0
            }
        except Exception:
            pass  # Fallback to zero-key engines below

    # Step 4: Free Instant Knowledge Search (DuckDuckGo 0-Key API)
    instant_info = fetch_free_instant_knowledge(user_question)
    if instant_info:
        return {
            "answer": instant_info,
            "source": "instant-knowledge (0-key API)",
            "confidence": 90.0
        }

    # Step 5: Multi-Match Knowledge Base Synthesis (Offline 0-Key Search)
    synthesized_info = synthesize_multi_match(user_question)
    if synthesized_info:
        return {
            "answer": synthesized_info,
            "source": "faq (knowledge synthesis)",
            "confidence": 75.0
        }

    # Final polite fallback
    return {
        "answer": f"I couldn't find a direct answer for '{user_question}' in our knowledge base. Please contact our support team at support@example.com for further help.",
        "source": "fallback",
        "confidence": 0
    }

def get_suggested_queries(limit=6):
    """Returns sample FAQ questions for click chips."""
    if faq_df is None or len(cleaned_questions) == 0:
        return [
            "How do I reset my password?",
            "What is Amazon EC2?",
            "How can I track my order?",
            "What is the student portal?",
            "What payment methods are accepted?",
            "How do I create an AWS account?"
        ]
    unique_qs = []
    for q in cleaned_questions:
        if q and q not in unique_qs:
            unique_qs.append(q)
        if len(unique_qs) >= limit:
            break
    return unique_qs

def get_system_health():
    """System health inspection status."""
    return {
        "dataset_loaded": faq_df is not None,
        "total_records": len(faq_df) if faq_df is not None else 0,
        "zero_key_knowledge_engine": True,
        "gemini_ai_ready": get_gemini_client() is not None
    }
