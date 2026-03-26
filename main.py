from fastapi import FastAPI, Request, File, UploadFile, Form,HTTPException,Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
import os
import base64
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
     CORSMiddleware,
     allow_origins=["*"],
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TextRequest(BaseModel):
     text: str

class ImageRequest(BaseModel):
     image: str

@app.post("/generate-text")
async def generate_text(request: TextRequest):
     try:
          response = client.chat.completions.create(
               model="gpt-4o",
               messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": request.text}
               ]
          )
          return JSONResponse({"text": response.choices[0].message.content})
     except Exception as e:
          raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
     try:
          response = client.images.generate(
               model="dall-e-3",
               prompt=request.image,
               n=1,
               size="1024x1024",
               response_format="b64_json"
          )
          return JSONResponse({"image": response.data[0].b64_json})
     except Exception as e:
          raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
     return {
          "status": "ok",
          "message": "Spandoek API is running"
     }

@app.get("/")
async def root():
     return {"message": "Spandoek API is running"}

if __name__ == "__main__":
     import uvicorn
     uvicorn.run(app, host="[IP_ADDRESS]", port=8080)