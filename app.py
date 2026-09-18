"""FastAPI web application for the Darukaa.Earth TF-IDF RAG assistant."""
from __future__ import annotations
import uuid
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
import ingest, memory
from rag_engine import resources, stream_answer

app = FastAPI(title="Darukaa.Earth")

class Question(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    context: dict[str, Any] | None = None
    session_id: str | None = Field(default=None, max_length=100)

@app.on_event("startup")
def startup() -> None: ingest.ensure_index()

@app.get("/health")
def health() -> dict[str, Any]:
    return {"status":"ok", "chunks":len(resources()["documents"]), "retrieval":"local-tfidf"}

@app.post("/api/chat")
def chat(payload: Question) -> StreamingResponse:
    session_id = payload.session_id or str(uuid.uuid4())
    try: generator, sources = stream_answer(payload.question.strip(), payload.context, session_id)
    except Exception as error: raise HTTPException(400, str(error)) from error
    def events():
        try:
            memory.add(session_id, "user", payload.question.strip())
            yield from generator
            memory.add(session_id, "assistant", "Answer delivered to browser.")
        except Exception: yield "\n\n[Unable to answer. Both configured providers were unavailable; please retry shortly.]"
    return StreamingResponse(events(), media_type="text/plain; charset=utf-8", headers={"X-Session-Id":session_id,"X-Sources":" | ".join(sources)})

@app.get("/", response_class=HTMLResponse)
def home() -> str: return HTML

HTML = '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Terra</title><style>body{margin:auto;max-width:850px;background:#f5f8f3;color:#17351f;font:16px system-ui;padding:26px}h1{margin-bottom:3px}.sub{color:#57705d}.chat{min-height:300px;background:#fff;border:1px solid #d9e5d9;border-radius:12px;padding:18px;margin:20px 0}.msg{white-space:pre-wrap;padding:12px;margin:9px 0;border-radius:9px}.user{background:#e5f2e3}.assistant{background:#f6f6f6}textarea{box-sizing:border-box;width:100%;height:105px;padding:12px;border:1px solid #b6cbb7;border-radius:8px;font:inherit}button{margin-top:9px;background:#216e39;color:white;border:0;border-radius:8px;padding:11px 22px;font-weight:600;cursor:pointer}small{color:#557}</style></head><body><h1>🌱 Terra</h1><div class="sub">Grounded biodiversity advice from your environmental PDFs.</div><div id="chat" class="chat"><div class="msg assistant">Ask in plain text, or paste JSON with a question and land details.</div></div><textarea id="input" placeholder='Ask a question… or paste JSON: {"question":"What should I do?","rainfall":"low","crop":"wheat"}'></textarea><button onclick="ask()">Send</button><p><small id="status"></small></p><script>let sid=localStorage.terraSession||crypto.randomUUID();localStorage.terraSession=sid;const chat=document.querySelector('#chat'),status=document.querySelector('#status'),input=document.querySelector('#input');function add(role,text){let x=document.createElement('div');x.className='msg '+role;x.textContent=text;chat.append(x);chat.scrollTop=chat.scrollHeight;return x}async function ask(){let raw=input.value.trim(),q=raw,ctx=null;if(!raw)return;if(raw.startsWith('{'))try{let data=JSON.parse(raw);q=data.question;if(typeof q!=='string'||!q.trim())throw Error('JSON needs a non-empty "question" field.');delete data.question;ctx=data}catch(e){status.textContent='Invalid JSON: '+e.message;return}add('user',q);input.value='';let out=add('assistant','');status.textContent='Retrieving evidence…';try{let r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,context:ctx,session_id:sid})});sid=r.headers.get('X-Session-Id')||sid;localStorage.terraSession=sid;if(!r.ok)throw Error((await r.json()).detail);let reader=r.body.getReader(),dec=new TextDecoder();while(true){let z=await reader.read();if(z.done)break;out.textContent+=dec.decode(z.value,{stream:true});chat.scrollTop=chat.scrollHeight}status.textContent=r.headers.get('X-Sources')?'Sources: '+r.headers.get('X-Sources'):'Response complete.'}catch(e){out.textContent='Error: '+e.message;status.textContent='Check keys or retry.'}}</script></body></html>'''
