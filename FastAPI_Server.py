from fastapi import FastAPI, HTTPException
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain_teddynote import logging
import faiss
import numpy as np
import json
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 OpenAI API 키 가져오기
project_api_key = os.getenv("OPENAI_API_KEY")
organization_id = os.getenv("ORGANIZATION_ID")
project_id = os.getenv("PROJECT_ID")
langsmith_project = os.getenv("LANGCHAIN_PROJECT")

# langchain 프로젝트명
logging.langsmith(langsmith_project)

# OpenAI API 키 설정
client = OpenAI(
    temperature=0.1,
    model_name='gpt-4o-mini'
)

# FastAPI 앱 생성
app = FastAPI()

# GPT-4o-mini 모델을 사용할 OpenAI 클라이언트 설정
llm = OpenAI(model="gpt-4o-mini", api_key=project_api_key)

# 미리 생성한 FAISS 인덱스 및 법률 데이터 로드
index = faiss.read_index("law_embeddings.index")
with open("law_embeddings.json", "r", encoding="utf-8") as f:
    law_embeddings = json.load(f)

# 법률 임베딩에서 가장 유사한 텍스트를 검색하는 함수
def search_law_embeddings(query_embedding, top_k=3):
    D, I = index.search(np.array([query_embedding], dtype=np.float32), top_k)
    return [law_embeddings[i] for i in I[0]]

# 사용자 질문에 대한 답변 생성 함수
def generate_answer(question, law_texts):
    # Langchain을 통해 GPT-4o-mini 모델에 질문을 프롬프트
    prompt_template = """
    사용자가 제시한 질문: {question}
    관련된 법률 정보:
    {law_texts}
    
    위 법률에 기반하여 질문에 대한 답변을 작성하세요.
    """
    prompt = PromptTemplate(
        input_variables=["question", "law_texts"],
        template=prompt_template,
    )
    formatted_prompt = prompt.format(question=question, law_texts=law_texts)

    # GPT-4o-mini 모델에 프롬프트 전달
    return llm(formatted_prompt)

