from fastapi import FastAPI, File, UploadFile
from paddleocr import PaddleOCR
import re
import pdf2image
import shutil
import os
from PIL import Image

app = FastAPI()

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

def extract_pan_and_dob(image_path):
    
    
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    results = ocr.ocr("temp_pan.jpeg")
    print(results)
    
    print("inside method")
    """ Extract PAN Number and DOB from the given image """
    print("inside method")
    results = ocr.ocr(image_path)
    print("inside method")
    # Extract text from OCR result
    text_list = [line[1][0] for line in results[0]]
    extracted_text = ' '.join(text_list)

    # Regex for PAN Number
    pan_match = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", extracted_text)
    pan_number = pan_match.group() if pan_match else "Not Found"

    # Regex for DOB
    dob_match = re.search(r"\b\d{2}[/\-]\d{2}[/\-]\d{4}\b", extracted_text)
    dob = dob_match.group() if dob_match else "Not Found"

    return {"PAN Number": pan_number, "DOB": dob}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """ API Endpoint to upload a PDF, PNG, or JPEG file and extract PAN & DOB """

    file_extension = file.filename.split(".")[-1].lower()
    print("working")

    if file_extension in ["pdf"]:
        # Save the uploaded PDF
        pdf_path = f"temp_{file.filename}"
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print("working2")

        # Convert PDF to images
        images = pdf2image.convert_from_path(pdf_path)

        # Save the first page as an image
        image_path = "temp_image.jpg"
        images[0].save(image_path, "JPEG")

        # Remove PDF after processing
        os.remove(pdf_path)

    elif file_extension in ["png", "jpeg", "jpg"]:
        # Save the uploaded image
        image_path = f"temp_{file.filename}"
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print("working3")

    else:
        print("working4")
        
        return {"error": "Unsupported file format. Please upload a PDF, PNG, or JPEG file."}

    # Extract PAN & DOB
    extracted_data = extract_pan_and_dob(image_path)
    print("working6")
    # Cleanup temporary image file
    os.remove(image_path)
    print("working5")

    return {"filename": file.filename, "extracted_data": extracted_data}

# uvicorn ocr:app --reload