import face_recognition
import requests
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from datetime import datetime
from typing import List, Union
import uvicorn
import pytz
import asyncio
import io



app = FastAPI()

class Item(BaseModel):
    img1: str
    img2: str

def get_bd_time():
    bd_timezone = pytz.timezone("Asia/Dhaka")
    time_now = datetime.now(bd_timezone)
    current_time = time_now.strftime("%I:%M:%S %p")
    return current_time

def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        image_data = response.content  # Binary image data
        return image_data
    else:
        return None

def is_same_person(image1_data, image2_data):
    image1 = face_recognition.load_image_file(io.BytesIO(image1_data))
    image2 = face_recognition.load_image_file(io.BytesIO(image2_data))
    face_encodings1 = face_recognition.face_encodings(image1)
    face_encodings2 = face_recognition.face_encodings(image2)
    if not face_encodings1 or not face_encodings2:
        return False
    for encoding1 in face_encodings1:
        for encoding2 in face_encodings2:
            distance = face_recognition.face_distance([encoding1], encoding2)[0]
            threshold = 0.6
            if distance < threshold:
                return True
    return False

def analyse(r):
    if r == True:
        det = "Same Person"
    else:
        det = "Different Person"
    return det

async def process_item(item: Item):
    try:
        img1_data = download_image(item.img1)
        img2_data = download_image(item.img2)
        
        if img1_data is not None and img2_data is not None:
            result = is_same_person(img1_data, img2_data)
            report = analyse(result)
            return {"AI": report}
        else:
            return {"AI": "Image download failed"}
    finally:
        pass

async def process_items(items: Union[Item, List[Item]]):
    if isinstance(items, list):
        coroutines = [process_item(item) for item in items]
        results = await asyncio.gather(*coroutines)
        print("multi: ", results)
    else:
        results = await process_item(items)
        print("single: ", results)
    return results

@app.get("/status")
async def status():
    return "AI Server is running"

@app.post("/face")
async def create_items(items: Union[Item, List[Item]]):
    try:
        results = await process_items(items)
        print("###################################################################################################")
        print(items)
        print("Last Execution Time: ", get_bd_time())
        return results
    finally:
        pass


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="127.0.0.1", port=8060)
    finally:
        pass
