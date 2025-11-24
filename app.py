import os
import json
import datetime
from typing import List, Dict, Optional
import re 

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===== 0. Gemini 설정 (분석 및 임베딩용) =====
import google.generativeai as genai

# [주의] API 키는 환경 변수로 설정하거나 안전하게 관리해야 합니다.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyB83EARkSHNRarWsAubDWiihNywP93iawQ") 

try:
    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
        genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Gemini 설정 실패: {e}")

# ===== 1. Gemini Embedding (SBERT 대체) =====

def get_embedding(text: str) -> List[float]:
    """Gemini API를 사용하여 텍스트 임베딩 벡터를 생성합니다."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        raise RuntimeError("Gemini API 키가 없어 임베딩을 생성할 수 없습니다.")
    
    try:
        response = genai.embed_content(
            model='models/text-embedding-004',
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return response["embedding"]
    except Exception as e:
        raise RuntimeError(f"Gemini 임베딩 API 호출 오류: {e}")


# ===== 2. Firestore 초기화 (유지) =====
import firebase_admin
from firebase_admin import credentials, firestore

# Firebase Admin SDK 파일 경로 설정
SERVICE_ACCOUNT_FILE =  "stellink-b94ac-firebase-adminsdk-fbsvc-c86ad07b09.json"

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        print(f"Firebase 초기화 오류: {e}")

db = firestore.client()
COLLECTION_NAME = "posts" 


# ===== 3. Pydantic 모델 (유지) =====

class PostIn(BaseModel):
    content: str

class PostOut(BaseModel):
    id: str
    emotion: str 
    topics: List[str] 
    similar: List[Dict] 


# ===== 4. Gemini 분석 유틸리티 (기본값 제거) =====

def classify_text_with_gemini(text: str):
    """일기 텍스트를 Gemini를 이용해 감정 및 주제로 분류합니다."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        # [수정] 기본값 반환 대신 오류 발생
        raise RuntimeError("Gemini API 키가 설정되지 않아 분석할 수 없습니다.")
        
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        system_prompt = (
            """
            너는 한국어 일기 텍스트를 감정/주제로 분류하는 어시스턴트다.
            반드시 아래 JSON 포맷만 출력해라 (설명/코드블록 금지).
            {
              "emotion": "<하나의 대표 감정: 행복|기쁨|뿌듯함|평온|불안|스트레스|우울|분노|슬픔|짜증|외로움|설렘 중 하나>",
              "topics": ["주제1","주제2","주제3"]
            }
            - topics는 1~3개, 한글 단어로 간결하게.
            - 일기 텍스트의 핵심만 반영.
            """
        )
        resp = model.generate_content([system_prompt, f"일기:\n{text}"])
        t = (resp.text or "").strip()
        
        try:
            data = json.loads(t)
        except Exception:
            m = re.search(r"\{.*\}", t, flags=re.S)
            if not m:
                # [수정] 파싱 실패 시 오류 발생
                raise RuntimeError("Gemini 분류 응답 파싱 실패")
            data = json.loads(m.group(0))

        # 응답이 유효하지 않아도 기본값 대신 오류 발생 (제거된 부분)
        emotion = data.get("emotion") 
        topics = data.get("topics")
        
        if not emotion or not topics:
            raise RuntimeError("Gemini 분석 결과에 emotion 또는 topics가 누락되었습니다.")
        
        if isinstance(topics, str):
            topics = [topics]
            
        return emotion, topics
        
    except Exception as e:
        print(f"🚨 Gemini 분석 오류 발생: {e}")
        # [수정] 오류 발생 시 기본값 반환 대신 다시 예외 발생
        raise RuntimeError(f"Gemini API 호출 중 오류 발생: {e}")


# ===== 5. Firestore 유틸리티 및 유사도 계산 (유지) =====

def ensure_embedding_for_doc(doc_ref, data: dict) -> Optional[np.ndarray]:
    # ... (유지) ...
    emb_list = data.get("embedding")
    if emb_list:
        return np.array(emb_list, dtype=np.float32)

    content = data.get("generated_content") or data.get("original_content")
    if not content:
        return None

    try:
        # [수정]: Gemini API 호출로 임베딩 생성
        emb = get_embedding(content)
    except RuntimeError as e:
        print(f"임베딩 생성 중 오류: {e}")
        return None
        
    emb_np = np.array(emb, dtype=np.float32)
    doc_ref.update({"embedding": emb}) 
    return emb_np


def fetch_corpus_embeddings(exclude_id: Optional[str] = None):
    # ... (유지) ...
    try:
        docs = db.collection(COLLECTION_NAME).stream()
    except Exception as e:
        print(f"Firestore 데이터 로드 오류: {e}")
        return []

    items = []
    for doc in docs:
        doc_id = doc.id
        if exclude_id and doc_id == exclude_id:
            continue

        data = doc.to_dict() or {}
        doc_ref = db.collection(COLLECTION_NAME).document(doc_id)
        emb = ensure_embedding_for_doc(doc_ref, data) 
        if emb is None:
            continue

        content = (
            data.get("generated_content")
            or data.get("original_content")
            or ""
        )
        preview = content[:120]
        emotion = data.get("emotion_tag", "")
        archive_tags = data.get("archive_tags") or []
        if isinstance(archive_tags, str):
            archive_tags = [archive_tags]

        items.append(
            (doc_id, preview, emb, emotion, archive_tags)
        )
    return items


def topk_similar(query_emb: np.ndarray, items, k: int = 5):
    # ... (유지) ...
    q = query_emb.astype(np.float32)
    qn = np.linalg.norm(q) + 1e-9

    sims = []
    for _id, preview, emb, emotion, tags in items:
        s = float(np.dot(q, emb) / (qn * (np.linalg.norm(emb) + 1e-9)))
        sims.append((_id, s, preview, emotion, tags))

    sims.sort(key=lambda x: x[1], reverse=True)
    top = sims[:k]

    similar_for_response = [
        {"id": _id, "score": round(score, 4), "preview": preview}
        for _id, score, preview, _, _ in top
    ]
    
    return similar_for_response


# ===== 6. FastAPI 앱 및 엔드포인트 수정 =====

app = FastAPI(title="Diary Constellation (Firestore + Gemini Embeddings)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.post("/posts", response_model=PostOut)
def create_post(post: PostIn):
    """
    1) Gemini로 감정/주제 분석, 2) Gemini로 임베딩, 3) 유사도 비교 결과를 반환합니다.
    """
    txt = post.content.strip()
    if not txt:
        raise HTTPException(400, "content is empty")

    try:
        # 1. Gemini로 감정/주제 분석
        emotion, topics = classify_text_with_gemini(txt)
        
        # 2. 쿼리 텍스트 임베딩 (Gemini 사용)
        q_emb_list = get_embedding(txt)
        q_emb = np.array(q_emb_list, dtype=np.float32)
        
    except RuntimeError as e:
        # [수정] Gemini API 호출/분석 실패 시 500 에러 발생
        raise HTTPException(500, detail=f"Analysis/Embedding failed: {e}")

    # 3. 기존 코퍼스 불러오기
    items = fetch_corpus_embeddings(exclude_id=None)
    
    # 4. 유사도 계산
    if not items:
        return {"id": "query", "emotion": emotion, "topics": topics, "similar": []}

    similar = topk_similar(q_emb, items, k=5)
    
    # id를 'query'로 반환 (Flutter 앱에서 저장 시 새로운 ID를 부여해야 함)
    return {
        "id": "query", 
        "emotion": emotion,
        "topics": topics,
        "similar": similar,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
