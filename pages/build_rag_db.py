"""
build_rag_db.py
Run this ONCE before starting the app to build the RAG database.
Usage: python build_rag_db.py

It reads news.csv, calls OpenAI to generate embeddings for each article,
and saves the resulting index to rag_db.pkl so the app loads instantly.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from openai import OpenAI

DB_PATH = "rag_db.pkl"
CSV_PATH = "news.csv"

def build_db():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)

    texts = []
    for _, row in df.iterrows():
        text = (
            f"Company: {row['company_name']}\n"
            f"Date: {row['Date'].strftime('%Y-%m-%d')}\n"
            f"Article: {str(row['Document'])[:800]}"
        )
        texts.append(text)

    print(f"Generating embeddings for {len(texts)} articles...")

    # Batch in groups of 100 to stay within rate limits
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        vecs = [item.embedding for item in resp.data]
        all_embeddings.extend(vecs)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    embeddings = np.array(all_embeddings, dtype=np.float32)


    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)

    db = {
        "df": df,
        "texts": texts,
        "embeddings": embeddings,
        "companies": sorted(df["company_name"].unique().tolist()),
    }

    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

    print(f"\nSaved to {DB_PATH}")
    print(f"  Articles: {len(df)}")
    print(f"  Companies: {len(db['companies'])}")


if __name__ == "__main__":
    build_db()
