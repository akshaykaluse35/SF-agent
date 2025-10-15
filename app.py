from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import re

app = Flask(__name__)

# --- Initialize Gemini 2.5 Flash ---
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    genai.configure(api_key=GEMINI_API_KEY)
    generation_model = genai.GenerativeModel("gemini-2.5-flash")

    # Safety settings (optional, can be relaxed for internal analytics)
    safety_settings = {
        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
    }

    print("✅ Gemini 2.5 Flash initialized successfully")
except Exception as e:
    print(f"❌ Error initializing Gemini: {e}")
    raise

# --- Health check endpoint ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Lead Scoring Service is running"}), 200

# --- Lead scoring endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data or "candidates" not in data:
            return jsonify({"error": "Invalid JSON: payload must contain 'candidates' key"}), 400

        winners_json = data.get("winners", [])
        losers_json = data.get("losers", [])
        candidates_json = data.get("candidates", [])

        # Prepare prompt for Gemini
        prompt = f"""
You are an expert sales data analyst. Analyze historical data to score new leads.
1. Winners (successfully converted leads): {winners_json}
2. Losers (unconverted leads): {losers_json}
3. Identify patterns of a successful lead from the winners and losers.
4. Now, analyze these new leads: {candidates_json}

Based on the patterns, score each new lead from 1 to 10 on their likelihood to convert and provide a one-sentence justification.
VERY IMPORTANT: Format your response as a simple bulleted list. For each lead, use this exact format:
• Lead Name (Company Name) (Score: X/10): Justification text.
"""

        # Generate response using Gemini
        response = generation_model.generate_content(prompt, safety_settings=safety_settings)
        text = response.text

        # Parse the Gemini response
        leads = []
        pattern = re.compile(
            r"•\s(.*?)\s\((.*?)\)\s\(Score:\s(\d{1,2})/10\):\s(.*?)(?=\n•|\Z)", re.DOTALL
        )
        matches = pattern.findall(text)
        for match in matches:
            leads.append({
                "name": match[0].strip(),
                "company": match[1].strip(),
                "score": int(match[2].strip()),
                "justification": match[3].strip()
            })

        return jsonify(leads)

    except Exception as e:
        print(f"❌ Error in /predict: {e}")
        return jsonify({"error": "An internal error occurred"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
