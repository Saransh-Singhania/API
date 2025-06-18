from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import json

# Initialize FastAPI app
app = FastAPI(title="Ollama Deepseek API")

# Define request model for the prompt
class PromptRequest(BaseModel):
    prompt: str

# Endpoint to generate response from deepseek-r1:1.5b model
@app.post("/generate")
async def generate_response(request: PromptRequest):
    try:
        # Prepare payload for Ollama API
        payload = {
            "model": "qwen3:0.6b",
            "prompt": request.prompt,
            "stream": False  # Non-streaming response for simplicity
        }

        # Send request to Ollama's /api/generate endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://127.0.0.1:11434/api/generate",
                json=payload,
                timeout=30.0
            )

        # Check if request was successful
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error contacting Ollama API")

        # Parse response
        result = response.json()
        if "response" not in result:
            raise HTTPException(status_code=500, detail="Invalid response from Ollama API")

        return {"response": result["response"]}

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to Ollama: {str(e)}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing Ollama response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}