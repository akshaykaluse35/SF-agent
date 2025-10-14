from flask import Flask, request, jsonify
import google.generativeai as genai
from pinecone import Pinecone
import os
import re

app = Flask(__name__)

# --- Initialize clients ---
try:
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = "salesforce-metadata"

    genai.configure(api_key=GEMINI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    embedding_model = "models/text-embedding-004"
    generation_model = genai.GenerativeModel('gemini-2.5-flash') # Or your available model

    # Define safety settings to be less strict
    safety_settings = {
        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
    }

    print("Services initialized successfully.")
except Exception as e:
    print(f"FATAL: Error during initialization: {e}")

# --- ENDPOINT 1: For Metadata Q&A ---
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json(silent=True)
    if not data or 'question' not in data:
        return jsonify({"error": "Invalid JSON: payload must contain a 'question' key."}), 400

    question = data['question']

    try:
        question_embedding = genai.embed_content(model=embedding_model, content=question)['embedding']

        query_response = index.query(vector=question_embedding, top_k=5, include_metadata=True)
        context = "".join([match['metadata']['text'] + "\n\n" for match in query_response['matches']])

        prompt = f"""
        You are a factual database engine for Salesforce metadata.
        Your task is to answer the user's question based ONLY on the provided context.
        - Be as brief and direct as possible.
        - Do NOT add greetings or sales insights.

        CONTEXT:
        {context}
        QUESTION:
        {question}
        ANSWER:
        """
        
        response = generation_model.generate_content(prompt, safety_settings=safety_settings)
        return jsonify({"answer": response.text})

    except Exception as e:
        print(f"Error processing /query request: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

# --- ENDPOINT 2: For Lead Prediction ---
@app.route('/predict', methods=['POST'])
def handle_predict():
    data = request.get_json(silent=True)
    if not data or 'candidates' not in data:
        return jsonify({"error": "Invalid JSON: payload must contain 'candidates' key."}), 400

    winners_json = data.get('winners', [])
    losers_json = data.get('losers', [])
    candidates_json = data.get('candidates', [])

    try:
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
        
        response = generation_model.generate_content(prompt, safety_settings=safety_settings)
        
        leads = []
        pattern = re.compile(r"•\s(.*?)\s\((.*?)\)\s\(Score:\s(\d{1,2})/10\):\s(.*?)(?=\n•|\Z)", re.DOTALL)
        matches = pattern.findall(response.text)
        for match in matches:
            leads.append({
                "name": match[0].strip(),
                "company": match[1].strip(),
                "score": match[2].strip(),
                "justification": match[3].strip()
            })
        
        return jsonify(leads)

    except Exception as e:
        print(f"Error processing /predict request: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == "__main__":
    app.run(port=int(os.environ.get('PORT', 8080)))