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
import requests

# --- Environment Setup ---
load_dotenv()
gemini_api_key = os.getenv("gemini_api_key")
hugging_token = os.getenv("hugging_face_token")
if not gemini_api_key: raise Exception("Missing Gemini API key")
if not hugging_token: raise Exception("Missing HuggingFace key")
genai.configure(api_key=gemini_api_key)

# --- Load Models Once ---
# Load models on request to save memory

# --- App Setup ---
app = FastAPI()
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
    temp_path = os.path.join(STATIC_DIR, f"audio_{uuid.uuid4()}.wav")
    with open(temp_path, "wb") as f:
        f.write(audio_file.file.read())
    r = sr.Recognizer()
    with sr.AudioFile(temp_path) as source:
        audio = r.record(source)
        text = r.recognize_google(audio)
    os.remove(temp_path)
    return text


def process_image(image_file: UploadFile):
    image = Image.open(BytesIO(image_file.file.read()))
    prompt = (
        "Analyze the image and generate a detailed, vivid description that captures visual elements, mood, atmosphere, "
        "and possible stories or emotions present."
    )
    response = gemini_model.generate_content([prompt, image])
    return response.text


def nlp_extraction(memories):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    structured = []
    for mem in memories:
        doc = nlp(mem)
        events = [sent.text for sent in doc.sents]
        entities = [ent.text for ent in doc.ents]
        keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        structured.append({"original": mem, "events": events, "entities": entities, "keywords": keywords})
    return structured


def sentiment_analysis(structured):
    from transformers import pipeline
    sent_analyzer = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
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
    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(f"{prompt}\n\nPlease create a mosaic story or synthesis that weaves together these memories.")
    return response.text


def generate_title(story):
    prompt = (
        f"{story}\n\nBased on the story above, generate a concise, creative title for this memory mosaic. "
        "Respond with only the title, no extra text."
    )
    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
    return response.text.strip()


def generate_image_prompt(story):
    prompt = (
        f"{story}\n\nBased on the story above, create ONE detailed, vivid, and visually clear image prompt "
        "for AI image generation. Respond with only the prompt."
    )
    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
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


    if audioFile and audioFile.filename:
        try:
            audio_text = process_audio(audioFile)
            if audio_text.strip():
                memories.append(audio_text.strip())
        except Exception as e:
            return {"error": f"Audio processing failed: {str(e)}"}


    if imageFile and imageFile.filename:
        try:
            image_text = process_image(imageFile)
            if image_text.strip():
                memories.append(image_text.strip())
        except Exception as e:
            return {"error": f"Image analysis failed: {str(e)}"}


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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

# Reduced memory usage version with on-demand model loading etc.
