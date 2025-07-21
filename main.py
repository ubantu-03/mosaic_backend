from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import speech_recognition as sr
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from dotenv import load_dotenv
import spacy
from transformers import pipeline
import requests
import uuid

# --- Environment and Model Setup ---

load_dotenv()
gemini_api_key = os.getenv("gemini_api_key")
hugging_token = os.getenv("hugging_face_token")

if not gemini_api_key: raise Exception("Missing Gemini API key")
if not hugging_token: raise Exception("Missing HuggingFace key")

genai.configure(api_key=gemini_api_key)
nlp = spacy.load("en_core_web_sm")
sent_analyzer = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# --- Helper Functions ---

def process_audio(audio_file: UploadFile):
    # Save temp audio
    temp_path = os.path.join(STATIC_DIR, f"audio_{uuid.uuid4()}.wav")
    with open(temp_path, "wb") as f:
        f.write(audio_file.file.read())
    # Speech-to-text
    r = sr.Recognizer()
    with sr.AudioFile(temp_path) as source:
        audio = r.record(source)
        text = r.recognize_google(audio)
    os.remove(temp_path)
    return text

def process_image(image_file: UploadFile):
    image = Image.open(BytesIO(image_file.file.read()))
    # Use Gemini to caption
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = (
        "Analyze the image and generate a detailed, vivid description that captures visual elements, mood, atmosphere, "
        "and possible stories or emotions present."
    )
    response = model.generate_content([prompt, image])
    return response.text

def nlp_extraction(memories):
    structured = []
    for mem in memories:
        doc = nlp(mem)
        events = [sent.text for sent in doc.sents]
        entities = [ent.text for ent in doc.ents]
        keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        structured.append({"original": mem, "events": events, "entities": entities, "keywords": keywords})
    return structured

def sentiment_analysis(structured):
    out = []
    for item in structured:
        result = sent_analyzer(
            item["original"],
            candidate_labels=["positive", "neutral", "negative"]
        )
        item["sentiment"] = result["labels"][0]
        item["sentiment_score"] = float(result["scores"][0])
        out.append(item)
    return out

def prepare_llm_prompt(structured):
    prompt = "Here are some personal memories:\n\n"
    for idx, mem in enumerate(structured, 1):
        prompt += f"{idx}. {mem['original']}\n"
        prompt += f"   Events: {mem['events']}\n"
        prompt += f"   Entities: {mem['entities']}\n"
        prompt += f"   Keywords: {mem['keywords']}\n"
        prompt += f"   Sentiment: {mem.get('sentiment','')} ({mem.get('sentiment_score',0):.2f})\n\n"
    return prompt

def run_llm_story(prompt):
    model = genai.GenerativeModel("gemini-2.5-flash")
    full_prompt = f"{prompt}\n\nPlease create a mosaic story or synthesis that weaves together these memories."
    response = model.generate_content(full_prompt)
    return response.text

def generate_title(story):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = (
        f"{story}\n\n"
        "Based on the story above, generate a concise, creative title for this memory mosaic. "
        "Respond with only the title, no extra text."
    )
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_image_prompt(story):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = (
        f"{story}\n\n"
        "Based on the story above, create ONE detailed, vivid, and visually clear image prompt for AI image generation. "
        "Respond with only a single image prompt in one complete sentence, describing the scene to be depicted in the image. "
        "Do not include any titles, explanations, or extra text—output only the image prompt."
    )
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_image_from_prompt(image_prompt, img_save_path):
    url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {
        "Authorization": f"Bearer {hugging_token}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json={"inputs": image_prompt})
    if response.ok:
        img = Image.open(BytesIO(response.content))
        img.save(img_save_path)
    else:
        raise Exception(f"Image gen error: {response.text}")

# --- Main POST Endpoint ---

@app.post("/api/memory")
async def create_memory(
    description: str = Form(...),
    audioFile: Optional[UploadFile] = File(None),
    imageFile: Optional[UploadFile] = File(None),
):
    memories = []
    if description.strip():
        memories.append(description.strip())

    # --- Audio
    if audioFile is not None and audioFile.filename != "":
        try:
            audio_text = process_audio(audioFile)
            if audio_text.strip(): memories.append(audio_text.strip())
        except Exception as e:
            return {"error": f"Audio processing failed: {str(e)}"}

    # --- Image
    if imageFile is not None and imageFile.filename != "":
        try:
            image_text = process_image(imageFile)
            if image_text.strip(): memories.append(image_text.strip())
        except Exception as e:
            return {"error": f"Image analysis failed: {str(e)}"}

    # --- NLP/AI pipeline:
    try:
        structured = nlp_extraction(memories)
        sentimental = sentiment_analysis(structured)
        llm_prompt = prepare_llm_prompt(sentimental)
        story = run_llm_story(llm_prompt)
        title = generate_title(story)
        image_prompt = generate_image_prompt(story)
        image_filename = f"memory_{uuid.uuid4().hex}.png"
        image_save_path = os.path.join(STATIC_DIR, image_filename)
        generate_image_from_prompt(image_prompt, image_save_path)
        image_url = f"/static/{image_filename}"
        return {
            "title": title,
            "story": story,
            "generated_image_url": image_url
        }

    except Exception as e:
        return {"error": f"Pipeline failed: {str(e)}"}

# --- Serve images (static files) ---
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

