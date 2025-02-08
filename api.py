from fastapi import FastAPI, UploadFile, File
import shutil
import model
import os
from typing import List

app = FastAPI()

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure temp folder exists

@app.post("/predict/")
async def predict_images(files: List[UploadFile] = File(...)):
    image_paths = []

    # Save uploaded files
    for file in files:
        if file.filename is None:
            continue
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_paths.append(file_path)

    # Process all images in batch
    predictions = model.predict(image_paths)

    # Remove temporary files
    for file_path in image_paths:
        os.remove(file_path)

    return {"predictions": predictions}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)