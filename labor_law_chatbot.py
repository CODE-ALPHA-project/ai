import faiss
import json
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
import os

# .env 파일에서 환경 변수 로드
load_dotenv()
project_api_key = os.getenv("OPENAI_API_KEY")
organization_id = os.getenv("ORGANIZATION_ID")
project_id = os.getenv("PROJECT_ID")
langsmith_project = os.getenv("LANGCHAIN_PROJECT")

memory = ConversationBufferMemory()

# OpenAI LLM 설정 (GPT-4o-mini)
llm = ChatOpenAI(
    temperature=0.1,            # 창의성 (0.0 ~ 2.0)
    model_name="gpt-4o-mini",   # 모델명
)

# 벡터 스토어 설정
def setup_faiss_vectorstore():
    # FAISS 인덱스 로드
    index = faiss.read_index("law_embeddings.index")

    # JSON 파일에서 문서 로드
    with open("law_embeddings.json", "r", encoding="utf-8") as f:
        docs_data = json.load(f)

    # OpenAI 임베딩 모델 설정
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # JSON 구조에 따라 Document 객체로 변환
    docs = []
    for item in docs_data:
        page_content = item.get('Body', '')  # 'Body' 필드 사용
        metadata = {
            "law": item.get('Law', ''),      # 'Law' 필드 사용
            "chapter": item.get('Chapter', ''),  # 'Chapter' 필드 사용
            "title": item.get('Title', '')   # 'Title' 필드 사용
        }
        doc = Document(page_content=page_content, metadata=metadata)
        docs.append(doc)

    # InMemoryDocstore에 문서 저장
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})

    # FAISS 벡터스토어 초기화
    vectorstore = FAISS(
        embedding_function=embeddings,      # 임베딩 생성기
        index=index,                        # 검색 인덱스
        docstore=docstore,                  # 문서 저장소
        index_to_docstore_id={i: str(i) for i in range(len(docs))}  # 인덱스와 문서 ID 매핑
    )
    return vectorstore


# 질의응답 체인 설정
def setup_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 시스템 프롬프트 정의
    system_prompt = (
        "You are an AI chatbot specialized in Korean labor law. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Please explain in as much detail as possible."
        "Please mention exactly which 'law' and 'title' you referenced in the retrieved context."
        "If possible, please also provide the number or link of the relevant department."
        "If it seems like you need to consult with a professional labor attorney, "
        "Please explain it to me as easily as possible, like a friend."
        # "please add the comment at the end, '전문 노무사와 매칭 서비스를 신청하시겠습니까?'"
        "\n\n"
        "{context}"
    )

    # 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    # 문서 조각을 결합하는 체인 생성
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # 검색과 QA를 결합하는 체인 생성
    chain = create_retrieval_chain(retriever, question_answer_chain)

    return chain

def ask_labor_law_question(rag_chain, question):
    # 이전 대화 히스토리를 불러옴, 없으면 빈 문자열 사용
    previous_context = memory.load_memory_variables({"input": question})
    chat_history = previous_context.get('history', '')  # 'chat_history'가 없으면 빈 문자열 사용
    print(chat_history)
    # 이전 대화와 새로운 질문 결합
    full_input = f"{chat_history}\nuser: {question}\nAI:"

    # RAG 체인을 사용하여 질문에 대한 답변을 생성 (invoke 사용)
    result = rag_chain.invoke({"input": full_input})
    answer = result.get('answer', '답변을 찾을 수 없습니다.')  # 'answer' 키로 답변 가져오기

    # 새로운 질문과 답변을 메모리에 저장
    memory.save_context({"input": question}, {"output": answer})

    # 참조된 문서 가져오기
    source_docs = result.get('source_documents', [])

    return answer, source_docs