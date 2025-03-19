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
# qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0 if torch.cuda.is_available() else -1)
qa_model = pipeline("text2text-generation", model="google/flan-t5-large", device=0 if torch.cuda.is_available() else -1)
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

    response = qa_model(f"question: {query} context: {best_para}", max_new_tokens=200)
    
    return {"answer": response[0]['generated_text'], "source_paragraph": best_para, "document": doc_name}
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


import os
import cv2
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path
import shutil


POPPLER_PATH = "/opt/homebrew/bin"

def convert_pdf_to_images(pdf_path, output_folder):
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    image_paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_folder, f"page_{i + 1}.jpg")
        img.save(path, "JPEG")
        image_paths.append(path)
    return image_paths

def crop_top_right(image, height_percent=0.10, width_percent=0.30):
    h, w = image.shape[:2]
    crop_h = int(h * height_percent)
    crop_w = int(w * width_percent)
    return image[0:crop_h, w - crop_w:w]

def match_logo_sift(logo_img, target_img, min_matches=10):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(logo_img, None)
    kp2, des2 = sift.detectAndCompute(target_img, None)

    if des1 is None or des2 is None:
        return False, 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return len(good) >= min_matches, len(good)

@app.post("/check-logo/")
async def check_logo(pdf: UploadFile = File(...), logo: UploadFile = File(...)):
    # Create a temp working directory
    session_id = str(uuid.uuid4())
    work_dir = f"temp_{session_id}"
    os.makedirs(work_dir, exist_ok=True)

    # Save uploaded files
    pdf_path = os.path.join(work_dir, "input.pdf")
    logo_path = os.path.join(work_dir, "logo.png")
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(pdf.file, f)
    with open(logo_path, "wb") as f:
        shutil.copyfileobj(logo.file, f)

    logo_img = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
    image_paths = convert_pdf_to_images(pdf_path, work_dir)
    results = []

    for i, img_path in enumerate(image_paths):
        page_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        cropped = crop_top_right(page_img)

        found, match_count = match_logo_sift(logo_img, cropped, min_matches=10)
        if match_count < 80:
            found = False

        results.append({
            "page": i + 1,
            "logo_found": found,
            "match_score": match_count
        })

        os.remove(img_path)

    # Cleanup
    os.remove(pdf_path)
    os.remove(logo_path)
    os.rmdir(work_dir)

    return JSONResponse(content={"result": results})


# **Ensure app binds to correct port for Render**
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)




# Run with: uvicorn filename:app --reload
