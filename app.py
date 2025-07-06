import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ✅ Load your vector DB and metadata
index = faiss.read_index(r"C:\Users\techin\Complaint_Analysis_Financial_Services\vector_store\directory\complaints_index.faiss")
with open(r"C:\Users\techin\Complaint_Analysis_Financial_Services\vector_store\directory\complaints_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# ✅ Load your models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline("text2text-generation", model="google/flan-t5-large", device=-1)

# ✅ Helper functions
def retrieve_top_k(question, k=5):
    question_embedding = embedder.encode([question])
    D, I = index.search(question_embedding, k)
    retrieved_chunks = [metadata[idx] for idx in I[0]]
    return retrieved_chunks

def build_prompt(retrieved_chunks, question):
    context = "\n".join([chunk.get('text', '') for chunk in retrieved_chunks])
    prompt = f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer. 
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}
Answer:
"""
    return prompt

def generate_answer(prompt):
    result = generator(prompt, max_length=200)[0]["generated_text"]
    return result

# ✅ Streamlit UI
st.set_page_config(page_title="CrediTrust Complaint Insights", layout="wide")
st.title("📊 CrediTrust Complaint Explorer")
st.markdown("Ask a question about customer complaints. The AI will search through complaint data and give you an evidence-based answer.")

# Input box & buttons
question = st.text_input("❓ Type your question here:", "")
submit = st.button("Ask")
clear = st.button("Clear")

if submit and question.strip():
    with st.spinner("Thinking..."):
        retrieved_chunks = retrieve_top_k(question)
        prompt = build_prompt(retrieved_chunks, question)
        answer = generate_answer(prompt)

        # Display the answer
        st.success(f"💬 **Answer:** {answer}")

        # Display retrieved sources
        st.markdown("#### 🔎 Retrieved Sources:")
        for i, chunk in enumerate(retrieved_chunks):
            st.markdown(f"**Source {i+1}:** {chunk.get('text', '')[:300]}...")

if clear:
    st.experimental_rerun()