#import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import os
import json
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

chunks_to_upsert = []
base_path = 'force-app/main/default/'

# --- 4. PROCESS OBJECT/FIELD METADATA (from JSON) ---
print("Processing Object and Field metadata...")
files_to_process = ['Lead.json', 'Opportunity.json']
for filename in files_to_process:
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            object_name = data['name']
            
            # Create summary chunk
            field_count = len(data['fields'])
            summary_id = f"{object_name}-summary"
            summary_text = f"The Salesforce object '{object_name}' has a total of {field_count} fields."
            chunks_to_upsert.append({"id": summary_id, "text": summary_text})

            # Process individual fields
            for field in data['fields']:
                field_name = field['name']
                field_type = field['type']
                field_label = field['label']
                
                chunk_id = f"{object_name}-{field_name}"
                chunk_text = f"In Salesforce, the object '{object_name}' has a field with the API name '{field_name}'. Its label is '{field_label}' and its data type is '{field_type}'."
                chunks_to_upsert.append({"id": chunk_id, "text": chunk_text})

# --- 5. PROCESS VALIDATION RULES (from Object XML) ---
print("Processing Validation Rules...")
objects_path = os.path.join(base_path, 'objects')
if os.path.exists(objects_path):
    for object_name in os.listdir(objects_path):
        object_dir_path = os.path.join(objects_path, object_name)
        object_meta_file = os.path.join(object_dir_path, f"{object_name}.object-meta.xml")

        if os.path.isdir(object_dir_path) and os.path.exists(object_meta_file):
            with open(object_meta_file, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'xml')
                for rule in soup.find_all('validationRules'):
                    rule_name = rule.find('fullName').text
                    rule_formula = rule.find('errorConditionFormula').text
                    chunk_id = f"{object_name}-vr-{rule_name}"
                    chunk_text = f"On the '{object_name}' object, there is a validation rule named '{rule_name}' with the formula: {rule_formula}"
                    chunks_to_upsert.append({"id": chunk_id, "text": chunk_text})

# --- 6. PROCESS FLOWS (from Flow XML) ---
flows_path = os.path.join(base_path, 'flows')
if os.path.exists(flows_path):
    print("Processing Flows...")
    for filename in os.listdir(flows_path):
        if filename.endswith('.flow-meta.xml'):
            flow_name = os.path.splitext(filename)[0]
            with open(os.path.join(flows_path, filename), 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'xml')
                trigger = soup.find('trigger')
                if trigger and trigger.find('object'):
                    object_name = trigger.find('object').text
                    chunk_id = f"flow-{flow_name}"
                    chunk_text = f"In Salesforce, there is a Flow named '{flow_name}' that is triggered to run on the '{object_name}' object."
                    chunks_to_upsert.append({"id": chunk_id, "text": chunk_text})

# --- 7. PROCESS APEX TRIGGERS (from Trigger XML) ---
triggers_path = os.path.join(base_path, 'triggers')
if os.path.exists(triggers_path):
    print("Processing Apex Triggers...")
    for filename in os.listdir(triggers_path):
        if filename.endswith('.trigger-meta.xml'):
            trigger_name = os.path.splitext(filename)[0]
            chunk_id = f"trigger-{trigger_name}"
            chunk_text = f"In Salesforce, there is an Apex Trigger named '{trigger_name}'."
            chunks_to_upsert.append({"id": chunk_id, "text": chunk_text})

# --- 8. EMBED AND UPLOAD IN BATCHES ---
print(f"Found {len(chunks_to_upsert)} total chunks to process.")

def embed_content(text_chunk):
    try:
        result = genai.embed_content(model="models/text-embedding-004", content=text_chunk, task_type="RETRIEVAL_DOCUMENT")
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