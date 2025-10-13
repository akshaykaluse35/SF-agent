import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import os
from bs4 import BeautifulSoup
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

# --- 4. DEFINE METADATA PATH AND FIND OBJECTS ---
objects_path = 'force-app/main/default/objects/'
chunks_to_upsert = []

print("Processing Custom Objects...")
# Loop through each directory in the 'objects' folder (e.g., Account, Lead)
for object_name in os.listdir(objects_path):
    object_dir_path = os.path.join(objects_path, object_name)
    
    if os.path.isdir(object_dir_path):
        fields_dir_path = os.path.join(object_dir_path, 'fields')
        
        # Check if a 'fields' subdirectory exists
        if os.path.isdir(fields_dir_path):
            # Loop through each field's XML file
            for field_filename in os.listdir(fields_dir_path):
                if field_filename.endswith('.field-meta.xml'):
                    with open(os.path.join(fields_dir_path, field_filename), 'r', encoding='utf-8') as file:
                        soup = BeautifulSoup(file.read(), 'xml')
                        
                        field_name = soup.find('fullName').text if soup.find('fullName') else 'N/A'
                        field_type = soup.find('type').text if soup.find('type') else 'N/A'
                        field_desc = soup.find('description').text if soup.find('description') else 'No description available.'

                        chunk_id = f"{object_name}-{field_name}"
                        chunk_text = f"In Salesforce, on the '{object_name}' object, there is a field named '{field_name}'. Its type is '{field_type}' and its description is: {field_desc}"
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