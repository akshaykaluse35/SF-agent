from flask import Flask, request, jsonify
import google.generativeai as genai
from pinecone import Pinecone
import os

app = Flask(__name__)

# --- Initialize clients once when the application starts ---
try:
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = "salesforce-metadata"

    genai.configure(api_key=GEMINI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    # Pre-load the models for efficiency
    embedding_model = "models/text-embedding-004"
    generation_model = genai.GenerativeModel('gemini-2.5-flash') # Or your available model

    print("Services initialized successfully.")
except Exception as e:
    print(f"FATAL: Error during initialization: {e}")

@app.route('/query', methods=['POST'])
def handle_query():
    """Endpoint to handle questions from Salesforce."""
    data = request.get_json(silent=True)
    if not data or 'question' not in data:
        return jsonify({"error": "Invalid JSON: payload must contain a 'question' key."}), 400

    question = data['question']

    try:
        # 1. Embed the user's question
        question_embedding = genai.embed_content(
            model=embedding_model,
            content=question
        )['embedding']

        # 2. Search Pinecone for the most relevant context
        query_response = index.query(
            vector=question_embedding,
            top_k=5, # Get the top 5 most relevant metadata chunks
            include_metadata=True
        )
        context = "".join([match['metadata']['text'] + "\n\n" for match in query_response['matches']])

        # 3. Build an effective prompt and call Gemini for the final answer
        prompt = f"""
        You are a helpful Salesforce assistant. Answer the user's question directly and concisely based on the provided context.
        Do not add greetings or extra advice. Use lists for readability if needed.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        
        response = generation_model.generate_content(prompt)
        
        return jsonify({"answer": response.text})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == "__main__":
    app.run(port=int(os.environ.get('PORT', 8080)))