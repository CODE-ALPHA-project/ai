# uvicorn main:app --reload

import asyncio
from confluent_kafka import Consumer, KafkaError, Producer
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from labor_law_chatbot import setup_faiss_vectorstore, setup_rag_chain, ask_labor_law_question
import json

# FastAPI 앱 생성
app = FastAPI()

# 벡터스토어 및 체인 초기화
vectorstore = setup_faiss_vectorstore()
rag_chain = setup_rag_chain(vectorstore)

# Kafka Consumer 설정
consumer_config = {
    'bootstrap.servers': '13.209.21.155:9092',  # Kafka 브로커 주소
    'group.id': 'my-group',                        # Consumer 그룹 ID
    'auto.offset.reset': 'earliest'                # 메시지를 어디서부터 읽을지 설정 (earliest, latest)
}

# Kafka Producer 설정
producer_config = {
    'bootstrap.servers': '13.209.21.155:9092'  # Kafka 브로커 주소
}

# Consumer 객체 생성
consumer = Consumer(consumer_config)

# Producer 객체 생성
producer = Producer(producer_config)

# 구독할 토픽 지정
consumer.subscribe(['ai-messages'])

# 질문 모델
# class QuestionRequest(BaseModel):
#     query: str
#     history: List[str] = []

# 비동기 메시지 폴링 함수
async def consume_messages():
    loop = asyncio.get_event_loop()

    # ThreadPoolExecutor를 사용해 블로킹 함수인 poll을 비동기처럼 처리
    with ThreadPoolExecutor() as pool:
        while True:
            # poll 함수가 블로킹이므로 ThreadPoolExecutor에서 실행
            msg = await loop.run_in_executor(pool, consumer.poll, 1.0)

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # 파티션 끝에 도달한 경우, 경고 메시지 출력
                    print(f"Reached end of partition: {msg.partition()}")
                else:
                    # 그 외의 에러 처리
                    print(f"Error occurred: {msg.error()}")
            else:
                # 메시지 처리 (비동기 처리 가능)
                await handle_message(msg)

# 메시지 처리 함수 (비동기)
async def handle_message(msg):
    # 메시지를 utf-8로 디코딩하여 처리
    message_value = msg.value().decode('utf-8')

    print(f"Received message: {message_value}")

    # 질문을 처리하기 위해 ask_question 함수 호출
    answer, references = ask_labor_law_question(rag_chain, message_value)
    print(answer)
    print(references)

    # 응답 데이터를 Kafka로 전송할 데이터 형식으로 준비
    result = {
        "answer": answer,
        "references": [
            {"law": ref.metadata["law"], "chapter": ref.metadata["chapter"], "title": ref.metadata["title"]}
            for ref in references
        ]
    }

    # 결과를 JSON 형식으로 직렬화
    result_message = json.dumps(result)

    # 결과를 Kafka로 전송 (예: 'ai-responses' 토픽에전송)
    producer.produce('ai-responses', key=msg.key(),value=result_message.encode('utf-8'))
    producer.flush()

    print(f"Sent response to ai-responses: {result_message}")


# 이벤트 루프 실행
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(consume_messages())

# FastAPI 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


