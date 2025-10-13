import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import os
import json
import time

# --- 1. CONFIGURE YOUR KEYS AND SETTINGS ---
GEMINI_API_KEY = "AIzaSyB2Dkt1nV-6ojN9-8PYQPoYp6bi9zH9n-E"
PINECONE_API_KEY = "pcsk_3Km89c_MhsUQse9XxNEY4VPbT52nhrCqKJEiH6eAuGyFwmw8wHxvT3QdcQR3vtqaeRW5ZK"
PINECONE_INDEX_NAME = "salesforce-metadata" # Make sure this matches your Pinecone index name


# --- 2. INITIALIZE SERVICES ---
print("Initializing services...")
genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- 3. CONNECT TO (OR CREATE) PINECONE INDEX ---
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768, 
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    time.sleep(1) 
index = pc.Index(PINECONE_INDEX_NAME)
print("Pinecone index is ready.")

# --- 4. PROCESS THE JSON FILES ---
chunks_to_upsert = []
files_to_process = ['Lead.json', 'Opportunity.json']

for filename in files_to_process:
    if os.path.exists(filename):
        print(f"Processing {filename}...")
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # **FIX IS HERE: Removed ['result']**
            object_name = data['name']
            
            # **FIX IS HERE: Removed ['result']**
            for field in data['fields']:
                field_name = field['name']
                field_type = field['type']
                field_label = field['label']
                
                chunk_id = f"{object_name}-{field_name}"
                chunk_text = f"In Salesforce, the object '{object_name}' has a field with the API name '{field_name}'. Its label is '{field_label}' and its data type is '{field_type}'."
                chunks_to_upsert.append({"id": chunk_id, "text": chunk_text})

# --- 5. EMBED AND UPLOAD IN BATCHES (No changes here) ---
print(f"Found {len(chunks_to_upsert)} chunks to process.")

def embed_content(text_chunk):
    # (Embedding function is the same)
    try:
        result = genai.embed_content(model="models/text-embedding-004", content=text_chunk)
        return result['embedding']
    except Exception as e:
        print(f"Could not embed content: {text_chunk[:50]}... Error: {e}")
        return None

batch_size = 100
for i in range(0, len(chunks_to_upsert), batch_size):
    batch = chunks_to_upsert[i:i+batch_size]
    vectors_to_upsert = []
    
    print(f"Processing batch {i//batch_size + 1}...")
    for chunk in batch:
        embedding = embed_content(chunk["text"])
        if embedding:
            vectors_to_upsert.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": {"text": chunk["text"]}
            })
            
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)
        print(f"Successfully upserted {len(vectors_to_upsert)} vectors.")

print("\nProcessing complete!")