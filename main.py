from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from labor_law_chatbot import setup_faiss_vectorstore, setup_rag_chain, ask_labor_law_question

# FastAPI 앱 생성
app = FastAPI()

# 정적 파일 경로 설정 (CSS 파일 포함)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 템플릿 디렉토리 설정
templates = Jinja2Templates(directory="templates")

# 벡터스토어 및 체인 초기화
vectorstore = setup_faiss_vectorstore()
rag_chain = setup_rag_chain(vectorstore)

# 메인 페이지 설정
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 질문 모델
class QuestionRequest(BaseModel):
    query: str
    history: List[str] = []

# 질문 처리 API
@app.post("/ask")
async def ask_question(question: QuestionRequest):
    answer, references = ask_labor_law_question(rag_chain, question.query)

    # 참조 문서들을 포맷팅
    references_formatted = [
        {"law": ref.metadata["law"], "chapter": ref.metadata["chapter"], "title": ref.metadata["title"]}
        for ref in references
    ]

    return JSONResponse(content={"answer": answer, "references": references_formatted})