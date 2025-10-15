from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import re

app = Flask(__name__)

genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
generation_model = genai.GenerativeModel('gemini-2.5-flash')

@app.route('/predict', methods=['POST'])
def handle_predict():
    data = request.get_json(silent=True)
    candidates = data.get('candidates', [])
    winners = data.get('winners', [])
    losers = data.get('losers', [])

    prompt = f"""
    You are an expert sales data analyst. Analyze historical data to score new leads.
    Winners: {winners}
    Losers: {losers}
    New leads: {candidates}

    Score each new lead from 1 to 10 for likelihood to convert and provide one-line justification.
    Format:
    • Lead Name (Company) (Score: X/10): Justification
    """

    response = generation_model.generate_content(prompt)
    leads = []
    pattern = re.compile(r"•\s(.*?)\s\((.*?)\)\s\(Score:\s(\d{1,2})/10\):\s(.*?)(?=\n•|\Z)", re.DOTALL)
    matches = pattern.findall(response.text)
    for m in matches:
        leads.append({"name": m[0], "company": m[1], "score": int(m[2]), "justification": m[3]})
    return jsonify(leads)

if __name__ == "__main__":
    app.run(port=int(os.environ.get('PORT', 8080)))
