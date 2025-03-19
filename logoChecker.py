import os
import cv2
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path
import shutil

app = FastAPI()

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