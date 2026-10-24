"""OpenAI-compatible FastAPI server for Ghana-LLM."""

import os
import sys
import time
import uuid
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from fastapi.responses import StreamingResponse

MODEL_NAME = os.getenv("MODEL_PATH", "ghanaopenai/ghana-llm-base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

print(f"Loading {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True).to(DEVICE)
model.eval()
print("Model ready!")

app = FastAPI(title="Ghana-LLM API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "ghana-llm"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "ghana-llm", "object": "model", "owned_by": "ghanaopenai"}]
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    conversation = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt_str = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_str, return_tensors="pt").to(DEVICE)

    gen_kwargs = {
        "max_new_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "do_sample": req.temperature > 0,
        "repetition_penalty": 1.1,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if req.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(target=model.generate, kwargs={**inputs, **gen_kwargs, "streamer": streamer})
        thread.start()

        def stream_generator():
            resp_id = f"chatcmpl-{uuid.uuid4()}"
            created_time = int(time.time())
            for text_chunk in streamer:
                data = {
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text_chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        import json
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    new_ids = out[0][inputs.input_ids.shape[1]:]
    resp_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": resp_text},
            "finish_reason": "stop"
        }]
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
