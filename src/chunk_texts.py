# src/chunk_texts.py

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import csv

# 1. Load your cleaned data
df = pd.read_csv("C:/Users/techin/Complaint_Analysis_Financial_Services/data/filtered_complaints.csv")

# 2. Initialize the chunk splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

# 3. Prepare a list to hold chunks
chunk_records = []

# 4. Loop over each complaint narrative and chunk it
for idx, row in df.iterrows():
    narrative = row['Cleaned narrative']
    complaint_id = row.get('Complaint ID', idx)  # fallback to idx if no ID
    product = row['Product']

    # split text into chunks
    chunks = splitter.split_text(narrative)
    for chunk in chunks:
        chunk_records.append({
            "complaint_id": complaint_id,
            "product": product,
            "chunk_text": chunk
        })

# 5. Convert to a DataFrame
chunk_df = pd.DataFrame(chunk_records)

# 6. Save to a new CSV
chunk_df.to_csv("C:/Users/techin/Complaint_Analysis_Financial_Services/data/chunked_complaints.csv", index=False)

print(f"✅ Done! Created {len(chunk_df)} chunks across {len(df)} original complaints.")
print("Saved as data/chunked_complaints.csv")