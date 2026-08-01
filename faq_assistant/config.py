import os
from dotenv import load_dotenv

# Load environment variables from local .env or parent .env
base_dir = os.path.dirname(__file__)
load_dotenv(os.path.join(base_dir, ".env"))
load_dotenv(os.path.join(os.path.dirname(base_dir), ".env"))
load_dotenv()

class Config:
    GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip().strip('"').strip("'")
    OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
    FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    DATA_PATH = os.path.join(base_dir, "faq_data.csv")
