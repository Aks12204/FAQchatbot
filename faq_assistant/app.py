from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from config import Config
from faq_engine import process_query, get_suggested_queries, get_system_health

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    
    if not message:
        return jsonify({
            "answer": "Please enter a valid question.",
            "source": "error",
            "confidence": 0
        }), 400
        
    result = process_query(message)
    return jsonify(result)

@app.route("/api/suggested", methods=["GET"])
def suggested():
    return jsonify({"suggested": get_suggested_queries(6)})

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "online", "health": get_system_health()})

if __name__ == "__main__":
    print(f"🚀 FAQ Assistant running on http://127.0.0.1:{Config.FLASK_PORT}")
    app.run(host="0.0.0.0", port=Config.FLASK_PORT, debug=Config.FLASK_DEBUG)
