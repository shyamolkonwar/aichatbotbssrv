import os
import requests
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from typing import List, Optional, Dict

# Initialize FastAPI app
app = FastAPI(title="DeepSeek RAG Chatbot API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://*.netlify.app"  # Allow all Netlify subdomains
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Verify API key
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in .env file. Please set it and try again.")

# Initialize vector store
def initialize_vector_store():
    try:
        document_path = "knowledge_base1.pdf"
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Knowledge base file not found: {document_path}")
            
        loader = PyMuPDFLoader(document_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
        return vector_store
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        raise

# Initialize vector store at startup
try:
    vector_store = initialize_vector_store()
except Exception as e:
    print(f"Failed to initialize vector store: {str(e)}")
    vector_store = None

# Pydantic models for request/response
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    links: Optional[List[Dict[str, str]]] = None
    error: Optional[str] = None

# Query DeepSeek API
def query_deepseek(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": """You are an AI assistant for BSSRV College admissions. Your role is to provide accurate and helpful information about the college's admission process, courses, and facilities.

When responding:
1. If the message starts with [User: Name], use the provided name to personalize your response
2. Keep responses concise and clear
3. If the query is about admissions, end your response with an invitation to join the WhatsApp helpdesk
4. Format WhatsApp links in a special card format with the WhatsApp logo
5. Remove any markdown formatting or special characters from your response

Example response format:
For a user named "John" asking about admission process:
The admission process at BSSRV College involves submitting an online application form, followed by document verification and fee payment. The process typically takes 2-3 working days to complete.

[WhatsApp Card]
Join our 24/7 Admission Support Group
Click here to get instant help with your admission queries."""
            },
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error querying DeepSeek API: {str(e)}")

def is_admission_related(query: str) -> bool:
    admission_keywords = [
        'admission', 'apply', 'application', 'enroll', 'register', 'registration',
        'course', 'program', 'fee', 'deadline', 'eligibility', 'requirement',
        'document', 'entrance', 'exam', 'test', 'selection', 'criteria'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in admission_keywords)

def clean_text(text: str) -> tuple[str, List[Dict[str, str]]]:
    # Extract links before cleaning
    links = []
    link_pattern = r'\[(.*?)\]\((.*?)\)'
    
    def replace_link(match):
        link_text = match.group(1)
        link_url = match.group(2)
        # Clean the link text from any markdown
        link_text = re.sub(r'[*#_`]', '', link_text)
        links.append({"text": link_text, "url": link_url})
        return f"[LINK_{len(links)-1}]"
    
    # Replace markdown links with placeholders
    text = re.sub(link_pattern, replace_link, text)
    
    # Remove all markdown formatting and special characters
    text = re.sub(r'###\s*', '', text)            # Remove headers
    text = re.sub(r'\*\*|\*|__|_|`', '', text)    # Remove bold, italic, underline, code
    text = re.sub(r'^\s*[-•]\s*', '• ', text, flags=re.MULTILINE)  # Standardize bullet points
    text = re.sub(r'^\s*\d+\.\s*', '• ', text, flags=re.MULTILINE)  # Convert numbered lists to bullet points
    text = re.sub(r'[\n]{3,}', '\n\n', text)     # Remove extra newlines
    text = re.sub(r'📧|😊|[^\w\s.,;:?!()\[\]{}|/<>@#$%^&*+=\-\'\"•\n]', '', text)  # Remove emojis and other special characters
    text = re.sub(r'\s+', ' ', text)              # Remove extra spaces
    text = text.strip()                           # Remove leading/trailing whitespace
    
    # Replace link placeholders with link text
    for i, link in enumerate(links):
        text = text.replace(f"[LINK_{i}]", link["text"])
    
    return text, links

# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not vector_store:
            raise HTTPException(
                status_code=500,
                detail="Vector store not initialized. Please check the server logs."
            )

        # Get relevant documents from vector store
        docs = vector_store.similarity_search(request.query, k=3)
        
        # Clean each document's content before joining
        cleaned_contents = []
        for doc in docs:
            cleaned_text, _ = clean_text(doc.page_content)
            cleaned_contents.append(cleaned_text)
        
        # Join the cleaned contents
        context = "\n".join(cleaned_contents)
        
        # Add instruction about WhatsApp card if query is admission-related
        if is_admission_related(request.query):
            context += "\n\nRemember to include the WhatsApp helpdesk card at the end of your response."
        
        # Construct prompt with cleaned context
        prompt = f"Context:\n{context}\n\nQuery:\n{request.query}\n\nAnswer based on the context provided."
        
        # Get response from DeepSeek
        response = query_deepseek(prompt)
        
        # Clean the response text and extract links
        cleaned_response, links = clean_text(response)
        
        # If response contains WhatsApp card markers, format it properly
        if "---whatsapp-card---" in cleaned_response:
            parts = cleaned_response.split("---whatsapp-card---")
            main_response = parts[0].strip()
            card_content = parts[1].split("---end-card---")[0].strip()
            
            # Add WhatsApp link to links array if not already present
            whatsapp_link = next((link for link in links if "whatsapp.com" in link["url"]), None)
            if not whatsapp_link:
                links.append({
                    "text": "Join BSSRV Admission Helpdesk",
                    "url": "https://chat.whatsapp.com/your-group-link",
                    "isWhatsApp": True
                })
            
            cleaned_response = main_response
        
        return ChatResponse(response=cleaned_response, links=links)
        
    except Exception as e:
        return ChatResponse(
            response="",
            error=f"An error occurred while processing your request: {str(e)}"
        )

# Health check endpoint
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "vector_store_initialized": vector_store is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)