from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from werkzeug.utils import secure_filename
import os


app = Flask(__name__)

# Load model and documents
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [
    "Legal contract about property rights",
    "Court ruling on employment disputes",
    "Patent law regarding intellectual property",
    "Legal framework for data protection"
]

# Encode and index documents
vectors = model.encode(documents)
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors))

@app.route("/", methods=["GET", "POST"])
def search():
    results = []
    query = ""
    if request.method == "POST":
        query = request.form["query"]
        query_vector = model.encode([query])
        D, I = index.search(np.array(query_vector), k=3)
        results = [documents[i] for i in I[0]]

    return render_template("index.html", query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True)