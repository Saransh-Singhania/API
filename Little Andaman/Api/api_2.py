from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import httpx
import json
from typing import List

# Initialize FastAPI app
app = FastAPI(title="RAG API with Local Vector Store")

# Define request model for the prompt
class PromptRequest(BaseModel):
    prompt: str
    top_k: int = 3  # Number of context chunks to retrieve

# Global variables for vector store and embeddings
vector_store = None
embeddings = None

@app.on_event("startup")
async def load_rag_components():
    global vector_store, embeddings
    try:
        # Load your trained embeddings (must match your RAG model)
        embeddings = OllamaEmbeddings(model="qwen3:0.6b")
        
        # Load your pre-built vector store (update the path)
        vector_store = FAISS.load_local(
            folder_path=r"C:\Users\ACER\Documents\NIC_intern\Little Andaman\Api\V_set",
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load RAG components: {str(e)}")

# Endpoint to generate response with RAG
@app.post("/generate")
async def generate_response(request: PromptRequest):
    try:
        # 1. Retrieve relevant context from vector store
        docs = vector_store.similarity_search(
            query=request.prompt,
            k=request.top_k
        )
        
        # 2. Format context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 3. Create RAG prompt with context
        rag_prompt = f"""Create custom prompt template for GNIDP-focused responses"""
    
        template = """You are an expert assistant specialized in Little Andaman Island, its development projects, infrastructure, demographics, and all matters related to the Andaman & Nicobar Islands administration.

        Gather information only from the provided context and documents to give a proper structured answer to the queries. 
        Context from Knowledge Base: {context}

        User Question: {request.promt}
        Answer:"""
        
        # 4. Prepare payload for Ollama API
        payload = {
            "model": "qwen3:0.6b",
            "prompt": rag_prompt,
            "stream": False
        }

        # 5. Send request to Ollama
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://127.0.0.1:11434/api/generate",
                json=payload,
                timeout=60.0  # Increased timeout for RAG
            )

        # 6. Check and parse response
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, 
                              detail="Error contacting Ollama API")

        result = response.json()
        if "response" not in result:
            raise HTTPException(status_code=500, 
                              detail="Invalid response from Ollama API")

        return {
            "response": result["response"],
            "context_sources": [doc.page_content for doc in docs]  # For debugging
        }

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, 
                          detail=f"Failed to connect to Ollama: {str(e)}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, 
                          detail="Error parsing Ollama response")
    except Exception as e:
        raise HTTPException(status_code=500, 
                          detail=f"RAG processing error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Verify vector store is loaded
        if vector_store is None:
            return {"status": "unhealthy", "reason": "Vector store not loaded"}
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}