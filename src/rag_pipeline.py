# rag_pipeline.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 1. Load models and data
# -------------------------------
# ✅ Load your index and metadata from ABSOLUTE Windows paths
faiss_index_path = r"C:\Users\techin\Complaint_Analysis_Financial_Services\vector_store\directory\complaints_index.faiss"
metadata_path = r"C:\Users\techin\Complaint_Analysis_Financial_Services\vector_store\directory\complaints_metadata.pkl"

print("Loading embeddings and vector store...")
index = faiss.read_index(faiss_index_path)
with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

print(f"Loaded FAISS index with {index.ntotal} vectors.")
print(f"Loaded metadata with {len(metadata)} records.")

# ✅ Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --------------------------------------------------------------------------------
# ✅ Load local text generation model (flan-t5-large)
print("Loading local LLM model...")
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)  # -1 means CPU

print("All models loaded successfully.\n")

# 🔍 Function: Retrieve top-k chunks
def retrieve_top_k(query, k=5):
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding).astype("float32"), k)
    return [metadata[i] for i in I[0]]

# 📝 Build the structured prompt
def build_prompt(context_chunks, question):
    context_text = "\n".join([chunk.get("text", "") for chunk in context_chunks])
    prompt = (
        f"You are a financial analyst assistant for CrediTrust.\n"
        f"Your task is to answer questions about customer complaints.\n"
        f"Use ONLY the following retrieved complaint excerpts to answer. "
        f"If you don't have enough information, say so.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    return prompt

# 🚀 Run local generation
def generate_answer_local(prompt):
    result = generator(prompt, max_new_tokens=200)
    return result[0]['generated_text'].strip()

# 🚀 Full QA pipeline
def answer_question(question, k=5):
    retrieved_chunks = retrieve_top_k(question, k)
    prompt = build_prompt(retrieved_chunks, question)
    answer = generate_answer_local(prompt)
    return answer, retrieved_chunks

# 🔍 Qualitative Evaluation: test on representative questions
import time

questions = [
    "Why are people unhappy with BNPL?",
    "Are there complaints about hidden fees in personal loans?",
    "Why are money transfers delayed?",
    "What issues do users report with savings accounts?",
    "Why are credit card applications rejected frequently?"
]

# Write the header to the markdown file
with open("evaluation_report.md", "w", encoding="utf-8") as f:
    f.write("# 📊 Qualitative Evaluation Table (Markdown format)\n\n")
    f.write("| Question | Generated Answer | Retrieved Sources | Quality Score (1-5) | Comments |\n")
    f.write("|---|---|---|---|---|\n")

    # Iterate through your predefined questions
    for q in questions:
        start_time = time.time()
        answer, retrieved = answer_question(q)
        elapsed = time.time() - start_time
        
        # Format top 2 retrieved chunks
        top_sources = " | ".join([
            chunk.get("text", "").replace("\n", " ")[:60] + "..." 
            for chunk in retrieved[:2]
        ])
        
        # Write row to markdown
        f.write(f"| {q} | {answer[:80]}... | {top_sources} |  |  |\n")
        
        print(f"✅ Finished: '{q}' in {elapsed:.2f}s")

print("\n🎉 Done! Evaluation saved to 'evaluation_report.md'")