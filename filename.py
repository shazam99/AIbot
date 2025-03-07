from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, File, Query
import fitz  # PyMuPDF for PDF extraction
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load extractive question-answering model (no API key needed)
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0 if torch.cuda.is_available() else -1)

# Load Mistral 7B (Hugging Face Model)
# model_name = "tiiuae/falcon-7b-instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# FAISS index setup
d = 384  # Dimension of embeddings
index = faiss.IndexFlatL2(d)
paragraphs = []  # To store paragraphs
metadata = []  # To track source documents

FAISS_DISTANCE_THRESHOLD = 1.2  # Adjust as needed
QA_CONFIDENCE_THRESHOLD = 0.5 

# Function to extract text from PDF
def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n\n".join([page.get_text("text") for page in doc])
    print(text)
    return text

# Function to add documents
def add_document(doc_name, text):
    global paragraphs, metadata
    chunks = text.split("\n\n")  # Split into paragraphs
    for para in chunks:
        vec = embed_model.encode([para])[0]
        index.add(np.array([vec]))
        paragraphs.append(para)
        metadata.append(doc_name)

# Function to get answer
def answer_query(query):
    query_vec = embed_model.encode([query])[0]
    D, I = index.search(np.array([query_vec]), k=1)  # Retrieve top paragraph

    best_distance = D[0][0]  # Get the FAISS distance
    best_para_index = I[0][0]
    print(best_distance)
    # Check if the retrieved paragraph is too far (i.e., not relevant)
    
    best_para = paragraphs[I[0][0]]
    doc_name = metadata[I[0][0]]
    
    # Use Mistral 7B for answer extraction
    # prompt = f"Extract answer for: {query} from: {best_para}"
    # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # output = llm.generate(**inputs, max_new_tokens=100)
    # response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(best_para)

    result = qa_model(question=query, context=best_para)
    
    return {"answer": result["answer"], "source_paragraph": best_para, "document": doc_name}
# add_document("Advisory-on-Independent-Due-Diligence by-Operating-Units.pdf", extract_text_from_pdf("Advisory-on-Independent-Due-Diligence by-Operating-Units.pdf"))
# print(answer_query("Give me circular date?"))
# FastAPI Setup
app = FastAPI()
# Allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","*"],  # Change this to match your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.filename.endswith(".pdf"):
        pdf_bytes = await file.read()
        text = extract_text_from_pdf(pdf_bytes)
    else:
        text = file.file.read().decode("utf-8")
    print(text)
    print(file.filename)
    add_document(file.filename, text)
    return {"status": "Document added successfully"}

@app.get("/query")
def query_document(q: str = Query(..., description="Enter your query")):
    result = answer_query(q)
    return result

# Run with: uvicorn filename:app --reload
