import os
import requests as remote_requests
import pandas as pd
import numpy as np
from fastapi import *
from typing import *
from pydantic import BaseModel
import time

##
from uuid import UUID, uuid4
from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters

from openai import OpenAI

from fastapi_session import *

os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI()

class chat_input(BaseModel):
    messages: List[dict[str, str]]

############

qa_pairs = pd.read_json(
    'qa_pairs_embeddings.json', 
    lines = True, 
    orient = 'records',
    ).to_dict('records')

############


system_prompt = f'You are a large language model named The Line Safety Chatbot developed by TONOMUS to answer general questions regarding construction safety, such as "What should be ensured before work commences?". Your response should be short and abstract, less than 64 words. Conversations should flow, and be designed in a way to not reach a dead end by ending responses with "Do you have any further questions?"'

app = FastAPI()



@app.post(
    "/chat_complete",
    #dependencies=[Depends(cookie)],
    tags = ["the-line-safety-training-chatbot"],
    summary = "The LINE construction safety chatbot API backed by GPT-3.5 Turbo",
    )
async def chat_complete(
    input:chat_input,
    #session_data: SessionData = Depends(verifier),
    ):

    start_time = time.time()
    
    user_input = input.messages[-1]["content"]

    # embedding of the input
    input_embedding = remote_requests.post(
        'http://37.224.68.132:27329/text_embedding/all_mpnet_base_v2',
        json = {
        "text": user_input
        }
        ).json()['embedding_vector']


    # score the qa pairs
    similar_qas = []

    for r in qa_pairs:  

        question_score = np.dot(
            np.array(input_embedding),
            np.array(r['Question_embedding']),
            )

        # if the question matches the qa, return the answer
        if question_score >= 0.9:
            return {
            "response": r['Answer'],
            "response_source":"semantic_search",
            "response_score":question_score,
            "running_time":time.time() - start_time,
            }

        answer_score = np.dot(
            np.array(input_embedding),
            np.array(r['Answer_embedding']),
            )

        overall_score = np.max([question_score,answer_score])
        if overall_score >= 0.8:
            similar_qas.append({
                'Question':r['Question'],
                'Answer':r['Answer'],
                'question_score':question_score,
                'answer_score':answer_score,
                'overall_score':overall_score
                })

    similar_qas = sorted(similar_qas, key=lambda x: x['overall_score'],)

    # prompt engineering

    prompt_conversation = [{"role": "system", "content": system_prompt}]

    for r in similar_qas[-4:]:
        prompt_conversation.append({"role": "user", "content": r['Question'].strip()})
        prompt_conversation.append({"role": "assistant", "content":f"{r['Answer'].strip()}"})

    for m in input.messages[-10:]:
        prompt_conversation.append({"role": m["role"], "content": m["content"].strip()})

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=prompt_conversation
    )

    answer = response.choices[0].message.content

    return {
    "response": str(answer),
    "response_source":"generative_model",
    "running_time":time.time() - start_time,
    }
