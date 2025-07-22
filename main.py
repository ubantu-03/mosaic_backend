import os
import uuid
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import speech_recognition as sr
import google.generativeai as genai
from dotenv import load_dotenv
import spacy
from transformers import pipeline
import requests

# --- Environment Setup ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
hugging_token = os.getenv("HUGGING_FACE_TOKEN")

if not gemini_api_key: 
    raise Exception("Missing GEMINI_API_KEY environment variable")
if not hugging_token: 
    raise Exception("Missing HUGGING_FACE_TOKEN environment variable")

genai.configure(api_key=gemini_api_key)

# --- Load Models Once ---
try:
    gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
    nlp = spacy.load("en_core_web_sm")
    sent_analyzer = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli",
        device=-1  # Force CPU usage for free tier
    )
    print("All models loaded successfully")
except Exception as e:
    print(f"Model loading error: {e}")
    raise

# --- App Setup ---
app = FastAPI(title="Memory Mosaic API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Helper Functions ---

def process_audio(audio_file: UploadFile):
    """Process audio file and convert to text"""
    temp_path = os.path.join(STATIC_DIR, f"audio_{uuid.uuid4()}.wav")
    try:
        with open(temp_path, "wb") as f:
            f.write(audio_file.file.read())
        
        r = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio = r.record(source)
            text = r.recognize_google(audio)
        return text
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_image(image_file: UploadFile):
    """Analyze image using Gemini Vision"""
    image = Image.open(BytesIO(image_file.file.read()))
    prompt = (
        "Analyze the image and generate a detailed, vivid description that captures visual elements, mood, atmosphere, "
        "and possible stories or emotions present."
    )
    response = gemini_model.generate_content([prompt, image])
    return response.text

def nlp_extraction(memories):
    """Extract structured information from memories"""
    structured = []
    for mem in memories:
        doc = nlp(mem)
        events = [sent.text for sent in doc.sents]
        entities = [ent.text for ent in doc.ents]
        keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        structured.append({
            "original": mem, 
            "events": events, 
            "entities": entities, 
            "keywords": keywords
        })
    return structured

def sentiment_analysis(structured):
    """Analyze sentiment of structured memories"""
    out = []
    for item in structured:
        try:
            result = sent_analyzer(
                item["original"],
                candidate_labels=["positive", "neutral", "negative"]
            )
            item["sentiment"] = result["labels"][0]
            item["sentiment_score"] = float(result["scores"][0])
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            # Fallback to neutral if analysis fails
            item["sentiment"] = "neutral"
            item["sentiment_score"] = 0.5
        out.append(item)
    return out

def prepare_llm_prompt(structured):
    """Prepare prompt for LLM story generation"""
    prompt = "Here are some personal memories:\n\n"
    for idx, mem in enumerate(structured, 1):
        prompt += f"{idx}. {mem['original']}\n"
        prompt += f"   Events: {mem['events']}\n"
        prompt += f"   Entities: {mem['entities']}\n"
        prompt += f"   Keywords: {mem['keywords']}\n"
        prompt += f"   Sentiment: {mem.get('sentiment','')} ({mem.get('sentiment_score',0):.2f})\n\n"
    return prompt

def run_llm_story(prompt):
    """Generate story using Gemini LLM"""
    full_prompt = f"{prompt}\n\nPlease create a mosaic story or synthesis that weaves together these memories."
    response = gemini_model.generate_content(full_prompt)
    return response.text

def generate_title(story):
    """Generate title for the story"""
    prompt = (
        f"{story}\n\nBased on the story above, generate a concise, creative title for this memory mosaic. "
        "Respond with only the title, no extra text."
    )
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def generate_image_prompt(story):
    """Generate image prompt from story"""
    prompt = (
        f"{story}\n\nBased on the story above, create ONE detailed, vivid, and visually clear image prompt "
        "for AI image generation. Respond with only the prompt."
    )
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def generate_image_from_prompt(image_prompt, img_save_path):
    """Generate image using HuggingFace API"""
    url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {
        "Authorization": f"Bearer {hugging_token}",
        "Content-Type": "application/json"
    }
    
    # Add retry logic for HuggingFace model loading
    max_retries = 3
    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json={"inputs": image_prompt})
        if response.ok:
            img = Image.open(BytesIO(response.content))
            img.save(img_save_path)
            return
        elif "loading" in response.text.lower():
            print(f"Model loading, attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                import time
                time.sleep(10)  # Wait for model to load
            continue
        else:
            raise Exception(f"Image generation error: {response.text}")
    
    raise Exception("Failed to generate image after retries")

# --- Health Check Endpoint ---
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Memory Mosaic API is running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

# --- Main POST Endpoint ---
@app.post("/api/memory")
async def create_memory(
    description: str = Form(...),
    audioFile: Optional[UploadFile] = File(None),
    imageFile: Optional[UploadFile] = File(None),
):
    """Main endpoint to process memories and generate story"""
    memories = []
    
    # Process text description
    if description.strip():
        memories.append(description.strip())

    # Process audio file
    if audioFile and audioFile.filename:
        try:
            audio_text = process_audio(audioFile)
            if audio_text.strip():
                memories.append(audio_text.strip())
        except Exception as e:
            return {"error": f"Audio processing failed: {str(e)}"}

    # Process image file
    if imageFile and imageFile.filename:
        try:
            image_text = process_image(imageFile)
            if image_text.strip():
                memories.append(image_text.strip())
        except Exception as e:
            return {"error": f"Image analysis failed: {str(e)}"}

    if not memories:
        return {"error": "No valid memories provided"}

    try:
        # Process memories through the pipeline
        structured = nlp_extraction(memories)
        sentimental = sentiment_analysis(structured)
        llm_prompt = prepare_llm_prompt(sentimental)
        story = run_llm_story(llm_prompt)
        title = generate_title(story)
        image_prompt = generate_image_prompt(story)
        
        # Generate and save image
        image_filename = f"memory_{uuid.uuid4().hex}.png"
        image_save_path = os.path.join(STATIC_DIR, image_filename)
        generate_image_from_prompt(image_prompt, image_save_path)
        image_url = f"/static/{image_filename}"
        
        return {
            "title": title,
            "story": story,
            "generated_image_url": image_url,
            "image_prompt": image_prompt
        }

    except Exception as e:
        return {"error": f"Pipeline failed: {str(e)}"}

# For local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
