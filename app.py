import os
import asyncio
from typing import Any

import gunicorn
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from queue import Queue
from pydantic import BaseModel

from langchain.agents import AgentType, initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.schema import LLMResult

# -------------------

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
import os
import openai
import json
import re
import shutil

import tiktoken

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# --------------------

app = FastAPI()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize the agent (we need to do this for the callbacks)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k",
    streaming=True,  # ! important
    callbacks=[]  # ! important (but we will add them later)
)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
    output_key="output"
)
agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=[],
    llm=llm,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory,
    return_intermediate_steps=False
)

# embeddings = OpenAIEmbeddings()
# docs_db = FAISS.load_local("data/merged_vector", embeddings, allow_dangerous_deserialization=True)

class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    content: str = ""
    final_answer: bool = False
    
    def __init__(self) -> None:
        super().__init__()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content += token
        # if we passed the final answer, we put tokens in queue
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ['"', "}"]:
                    self.queue.put_nowait(token)
        elif "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""
    
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.final_answer:
            self.content = ""
            self.final_answer = False
            self.done.set()
        else:
            self.content = ""

embeddings = OpenAIEmbeddings()
global docs_db
docs_db = FAISS.load_local("data/merged_vector", embeddings)

async def run_call(query: str, stream_it: AsyncCallbackHandler):
    prompt = ""
    docs = docs_db.similarity_search(query)
    prompt += "\n\n"
    prompt += query
    prompt += "\n\n"
    prompt += str(docs)
    # assign callback handler
    agent.agent.llm_chain.llm.callbacks = [stream_it]
    # now query
    # await agent.acall(inputs={"input": query})
    await agent.acall(inputs={"input": prompt})

# request input format

class Query(BaseModel):
    query: str

async def create_gen(query: str, stream_it: AsyncCallbackHandler):
    task = asyncio.create_task(run_call(query, stream_it))
    async for token in stream_it.aiter():
        yield token
    await task

@app.post("/get_response_from_ai")
async def chat(
    # query: Query = Body(...),
    # query: str | None = None
    query: Query
):
    stream_it = AsyncCallbackHandler()
    # gen = create_gen(query.text, stream_it)
    gen = create_gen(query.query, stream_it)
    print(f"query.query --> {query.query}")
    return StreamingResponse(gen, media_type="text/event-stream")

@app.get("/health")
async def health():
    """Check the api is running"""
    return {"status": "😃"}

# ----------------------------------------


if __name__ == "__main__":
    import multiprocessing
    import subprocess
    import uvicorn
    
    # workers = multiprocessing.cpu_count() * 2 + 1

    workers = multiprocessing.cpu_count()
    
    uvicorn_cmd = [
        "uvicorn",
        "app:app",
        "--host=0.0.0.0",
        "--port=8000",
        f"--workers={workers}",
        # "--reload"
    ]
    
    # uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, workers=workers)
    subprocess.run(uvicorn_cmd)
