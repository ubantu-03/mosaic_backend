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
import re
import nltk
from collections import Counter

# Download NLTK data on startup
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass  # Continue without NLTK if download fails

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
    print("Gemini model loaded successfully")
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

def simple_nlp_extraction(memories):
    """Enhanced NLP extraction using NLTK"""
    structured = []
    
    try:
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        stop_words = set(stopwords.words('english'))
        sia = SentimentIntensityAnalyzer()
        
        for mem in memories:
            # Sentence tokenization
            sentences = sent_tokenize(mem)
            events = [s.strip() for s in sentences if s.strip()]
            
            # Entity extraction using regex patterns
            entities = []
            entities.extend(re.findall(r'\b[A-Z][a-z]+\b', mem))
            entities.extend(re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', mem))
            entities.extend(re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b', mem))
            
            # Keyword extraction
            words = word_tokenize(mem.lower())
            keywords = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
            keyword_freq = Counter(keywords)
            top_keywords = [word for word, count in keyword_freq.most_common(10)]
            
            # Sentiment analysis
            sentiment_scores = sia.polarity_scores(mem)
            if sentiment_scores['compound'] >= 0.05:
                sentiment = 'positive'
            elif sentiment_scores['compound'] <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            structured.append({
                "original": mem,
                "events": events,
                "entities": list(set(entities)),
                "keywords": top_keywords,
                "sentiment": sentiment,
                "sentiment_score": sentiment_scores['compound']
            })
    
    except ImportError:
        # Fallback to simple regex-based extraction
        for mem in memories:
            sentences = re.split(r'[.!?]+', mem)
            events = [s.strip() for s in sentences if s.strip()]
            
            entities = []
            entities.extend(re.findall(r'\b[A-Z][a-z]+\b', mem))
            entities.extend(re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', mem))
            entities.extend(re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b', mem))
            
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            words = re.findall(r'\b\w+\b', mem.lower())
            keywords = [word for word in words if word not in stop_words and len(word) > 2]
            
            structured.append({
                "original": mem,
                "events": events,
                "entities": list(set(entities)),
                "keywords": list(set(keywords)),
                "sentiment": "neutral",
                "sentiment_score": 0.5
            })
    
    return structured

def simple_sentiment_analysis(structured):
    """Use Gemini for sentiment analysis as fallback"""
    for item in structured:
        if 'sentiment' not in item or item['sentiment'] == 'neutral':
            try:
                prompt = f"Analyze the sentiment of this text and respond with only one word: 'positive', 'negative', or 'neutral'. Text: {item['original']}"
                response = gemini_model.generate_content(prompt)
                sentiment = response.text.strip().lower()
                
                if sentiment not in ['positive', 'negative', 'neutral']:
                    sentiment = 'neutral'
                    
                item["sentiment"] = sentiment
                item["sentiment_score"] = 0.8 if sentiment != 'neutral' else 0.5
            except Exception as e:
                print(f"Sentiment analysis error: {e}")
                item["sentiment"] = "neutral"
                item["sentiment_score"] = 0.5
    
    return structured

def prepare_llm_prompt(structured):
    """Prepare prompt for LLM story generation"""
    prompt = "Here are some personal memories:\n\n"
    for idx, mem in enumerate(structured, 1):
        prompt += f"{idx}. {mem['original']}\n"
        prompt += f"   Key Events: {', '.join(mem['events'])}\n"
        prompt += f"   Notable Entities: {', '.join(mem['entities'])}\n"
        prompt += f"   Keywords: {', '.join(mem['keywords'][:10])}\n"
        prompt += f"   Sentiment: {mem.get('sentiment','')} ({mem.get('sentiment_score',0):.2f})\n\n"
    return prompt

def run_llm_story(prompt):
    """Generate story using Gemini LLM"""
    full_prompt = f"{prompt}\n\nPlease create a beautiful, cohesive story or narrative that weaves together these memories in a meaningful way. Make it engaging and emotionally resonant."
    response = gemini_model.generate_content(full_prompt)
    return response.text

def generate_title(story):
    """Generate title for the story"""
    prompt = (
        f"Based on this story, generate a short, creative, and memorable title (maximum 8 words):\n\n{story}\n\n"
        "Respond with only the title, no quotes or extra text."
    )
    response = gemini_model.generate_content(prompt)
    return response.text.strip().replace('"', '').replace("'", '')

def generate_image_prompt(story):
    """Generate image prompt from story"""
    prompt = (
        f"Based on this story, create a detailed, artistic image prompt for AI generation that captures the essence and mood:\n\n{story}\n\n"
        "Make it visually rich and specific. Respond with only the image prompt."
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
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json={"inputs": image_prompt}, timeout=60)
            if response.ok:
                img = Image.open(BytesIO(response.content))
                img.save(img_save_path)
                return
            elif "loading" in response.text.lower():
                print(f"Model loading, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(15)
                continue
            else:
                print(f"Image generation error: {response.text}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(10)
                    continue
                else:
                    raise Exception(f"Image generation failed: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(10)
                continue
            else:
                raise Exception(f"Failed to generate image: {e}")
    
    raise Exception("Failed to generate image after retries")

# --- Health Check Endpoints ---
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
            print(f"Audio processing error: {e}")
            return {"error": f"Audio processing failed: {str(e)}"}

    # Process image file
    if imageFile and imageFile.filename:
        try:
            image_text = process_image(imageFile)
            if image_text.strip():
                memories.append(image_text.strip())
        except Exception as e:
            print(f"Image processing error: {e}")
            return {"error": f"Image analysis failed: {str(e)}"}

    if not memories:
        return {"error": "No valid memories provided"}

    try:
        print("Processing memories through NLP pipeline...")
        structured = simple_nlp_extraction(memories)
        sentimental = simple_sentiment_analysis(structured)
        llm_prompt = prepare_llm_prompt(sentimental)
        
        print("Generating story...")
        story = run_llm_story(llm_prompt)
        
        print("Generating title...")
        title = generate_title(story)
        
        print("Generating image prompt...")
        image_prompt = generate_image_prompt(story)
        
        print("Generating image...")
        image_filename = f"memory_{uuid.uuid4().hex}.png"
        image_save_path = os.path.join(STATIC_DIR, image_filename)
        generate_image_from_prompt(image_prompt, image_save_path)
        image_url = f"/static/{image_filename}"
        
        return {
            "title": title,
            "story": story,
            "generated_image_url": image_url,
            "image_prompt": image_prompt,
            "memories_processed": len(memories)
        }

    except Exception as e:
        print(f"Pipeline error: {e}")
        return {"error": f"Processing failed: {str(e)}"}

# For local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
