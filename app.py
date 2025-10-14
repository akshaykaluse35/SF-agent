from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import re

app = Flask(__name__)

# --- Initialize clients ---
try:
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    genai.configure(api_key=GEMINI_API_KEY)
    generation_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("Services initialized successfully.")
except Exception as e:
    print(f"Error during initialization: {e}")

def parse_gemini_response(text):
    """Parses the text output from Gemini to extract structured lead data."""
    leads = []
    # Regex to find patterns like: • Lead Name (Company Name) (Score: X/10): Justification
    pattern = re.compile(r"•\s(.*?)\s\((.*?)\)\s\(Score:\s(\d{1,2})/10\):\s(.*?)(?=\n•|\Z)", re.DOTALL)
    
    matches = pattern.findall(text)
    
    for match in matches:
        leads.append({
            "name": match[0].strip(),
            "company": match[1].strip(),
            "score": match[2].strip(),
            "justification": match[3].strip()
        })
    return leads

@app.route('/predict', methods=['POST'])
def handle_predict():
    """Endpoint to handle lead prediction requests from Salesforce."""
    data = request.get_json(silent=True)
    if not data or 'candidates' not in data:
        return jsonify({"error": "Invalid JSON: payload must contain 'candidates' key."}), 400

    winners_json = data.get('winners', [])
    losers_json = data.get('losers', [])
    candidates_json = data.get('candidates', [])

    try:
        prompt = f"""
        You are an expert sales data analyst. Your goal is to identify which new leads are most likely to convert successfully based on historical data.

        1. First, here is a list of our successfully converted leads (winners):
           {winners_json}

        2. Next, here is a list of our unconverted leads (losers):
           {losers_json}

        3. Based on the differences between the winners and losers, identify the key patterns and characteristics of a successful lead for our business.

        4. Finally, analyze this list of new, open leads (candidates). Based on the patterns you identified, score each one from 1 to 10 on their likelihood to convert. Provide a one-sentence justification for each score.
           {candidates_json}
        
        VERY IMPORTANT: Format your response as a simple bulleted list. For each lead, use this exact format:
        • Lead Name (Company Name) (Score: X/10): Justification text.
        """
        
        response = generation_model.generate_content(prompt)
        
        # Parse the text response into structured JSON
        predicted_leads = parse_gemini_response(response.text)
        
        return jsonify(predicted_leads)

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == "__main__":
    app.run(port=int(os.environ.get('PORT', 8080)))