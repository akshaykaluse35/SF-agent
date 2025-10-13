from flask import Flask, request, jsonify
import google.generativeai as genai
import pinecone
import os

app = Flask(__name__)

# --- Initialize clients ---
try:
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = "salesforce-metadata"

    genai.configure(api_key=GEMINI_API_KEY)
    
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    print("Services initialized successfully.")
except Exception as e:
    print(f"Error during initialization: {e}")

@app.route('/query', methods=['POST'])
def handle_query():
    """Endpoint to handle questions from Salesforce."""
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Invalid JSON: payload must contain a 'question' key."}), 400

    question = data['question']

    try:
        # 1. Embed the user's question
        question_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=question
        )['embedding']

        # 2. Search Pinecone for context
        query_response = index.query(
            vector=question_embedding,
            top_k=5,
            include_metadata=True
        )
        context = "".join([match['metadata']['text'] + "\n\n" for match in query_response['matches']])

        # 3. Augment prompt and call Gemini
        prompt = f"""
        Based ONLY on the following Salesforce metadata context, answer the user's question. Answer should not be so big. Create bullet lists so user can get it easily. If the context doesn't contain the answer, say that you don't have information about that specific metadata.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        model = genai.GenerativeModel('gemini-2.5-flash') # Or your available model
        response = model.generate_content(prompt)
        
        return jsonify({"answer": response.text})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.O', port=int(os.environ.get('PORT', 8080)))