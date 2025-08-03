import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

faq_df = pd.read_csv("faq_data.csv")
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq_df["question"])

def get_faq_answer(user_question, threshold=0.6):
    user_vec = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vec, faq_vectors).flatten()
    max_score = similarities.max()
    if max_score >= threshold:
        index = similarities.argmax()
        return faq_df.iloc[index]["answer"]
    return None

def get_openai_answer(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"