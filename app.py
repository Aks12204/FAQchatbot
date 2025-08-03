from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from utils import get_faq_answer, get_openai_answer

app = Flask(__name__, static_folder=".")
CORS(app)

@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"response": "Please send a message."}), 400
    faq_response = get_faq_answer(user_input)
    if faq_response:
        return jsonify({"response": faq_response})
    ai_response = get_openai_answer(user_input)
    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(debug=True)