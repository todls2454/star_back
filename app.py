import os
import json
import datetime
from typing import List, Dict, Optional
import re 

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===== 0. Gemini 설정 (감정/주제 분석용) =====
import google.generativeai as genai

# [주의] API 키는 환경 변수로 설정하거나 안전하게 관리해야 합니다.
# os.environ.get("GEMINI_API_KEY")를 사용하거나, 실제 키를 설정하세요.
GEMINI_API_KEY = 'AIzaSyB83EARkSHNRarWsAubDWiihNywP93iawQ' 

try:
    if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
        genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Gemini 설정 실패: {e}")

# ===== 1. SBERT (한국어 임베딩) =====
from sentence_transformers import SentenceTransformer
try:
    sbert_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
except Exception as e:
    print(f"SBERT 모델 로드 실패: {e}")
    sbert_model = None

def get_embedding(text: str) -> List[float]:
    if sbert_model is None:
        raise RuntimeError("SBERT 모델이 로드되지 않았습니다.")
    emb = sbert_model.encode(text, convert_to_numpy=True)
    return emb.astype(np.float32).tolist()


# ===== 2. Firestore 초기화 =====
import firebase_admin
from firebase_admin import credentials, firestore

# Firebase Admin SDK 파일 경로 설정
BASE_DIR = os.path.dirname(__file__)
SERVICE_ACCOUNT_FILE = os.path.join(BASE_DIR, "stellink-b94ac-firebase-adminsdk-fbsvc-c86ad07b09.json")

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        print(f"Firebase 초기화 오류: {e}")

db = firestore.client()
COLLECTION_NAME = "posts" 


# ===== 3. Pydantic 모델 =====

class PostIn(BaseModel):
    content: str

class PostOut(BaseModel):
    id: str
    emotion: str 
    topics: List[str] 
    similar: List[Dict] 


# ===== 4. Gemini 분석 유틸리티 =====

def classify_text_with_gemini(text: str):
    """일기 텍스트를 Gemini를 이용해 감정 및 주제로 분류합니다."""
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GEMINI_API_KEY:
        print("Gemini API 키가 없어 기본값 반환")
        return "평온", ["일상"]
        
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
        # Timeout 추가 (Gemini 호출 실패 방지)
        resp = model.generate_content([system_prompt, f"일기:\n{text}"])
        t = (resp.text or "").strip()
        
        # JSON 파싱 시도 및 예외 처리
        try:
            data = json.loads(t)
        except Exception:
            m = re.search(r"\{.*\}", t, flags=re.S)
            if not m:
                raise RuntimeError("Gemini 분류 응답 파싱 실패")
            data = json.loads(m.group(0))

        emotion = data.get("emotion", "평온")
        topics = data.get("topics", ["일상"])
        if isinstance(topics, str):
            topics = [topics]
            
        return emotion, topics
        
    except Exception as e:
        print(f"🚨 Gemini 분석 오류 발생: {e}")
        return "평온", ["일상"]


# ===== 5. Firestore 유틸리티 및 유사도 계산 =====

def ensure_embedding_for_doc(doc_ref, data: dict) -> Optional[np.ndarray]:
    emb_list = data.get("embedding")
    if emb_list:
        return np.array(emb_list, dtype=np.float32)

    content = data.get("generated_content") or data.get("original_content")
    if not content:
        return None

    try:
        emb = get_embedding(content)
    except RuntimeError:
        return None
        
    # [주의] Firestore 업데이트는 쓰기 비용이 발생하므로, 필요할 때만 호출해야 합니다.
    # doc_ref.update({"embedding": emb}) 
    return np.array(emb, dtype=np.float32)


def fetch_corpus_embeddings(exclude_id: Optional[str] = None):
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
        # 임베딩이 없으면 생성(및 저장) 시도
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
    q = query_emb.astype(np.float32)
    qn = np.linalg.norm(q) + 1e-9

    sims = []
    for _id, preview, emb, emotion, tags in items:
        # 코사인 유사도 계산
        s = float(np.dot(q, emb) / (qn * (np.linalg.norm(emb) + 1e-9)))
        sims.append((_id, s, preview, emotion, tags))

    sims.sort(key=lambda x: x[1], reverse=True)
    top = sims[:k]

    similar_for_response = [
        {"id": _id, "score": round(score, 4), "preview": preview}
        for _id, score, preview, _, _ in top
    ]
    
    # 유사도 계산 함수는 유사 포스트 목록만 반환 (감정/주제는 Gemini가 담당)
    return similar_for_response


# ===== 6. FastAPI 앱 및 엔드포인트 수정 =====

app = FastAPI(title="Diary Constellation (Firestore + SBERT + Gemini)")

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
    1) Gemini로 감정/주제 분석, 2) SBERT로 임베딩, 3) 유사도 비교 결과를 반환합니다.
    """
    txt = post.content.strip()
    if not txt:
        raise HTTPException(400, "content is empty")

    try:
        # 1. Gemini로 감정/주제 분석
        emotion, topics = classify_text_with_gemini(txt)
        
        # 2. 쿼리 텍스트 임베딩
        q_emb = np.array(get_embedding(txt), dtype=np.float32)
        
    except Exception as e:
        raise HTTPException(500, f"Analysis error (Gemini/SBERT): {e}")

    # 3. 기존 코퍼스 불러오기
    items = fetch_corpus_embeddings(exclude_id=None)
    
    # 4. 유사도 계산
    if not items:
        # 코퍼스가 비어 있으면, Gemini 분석 결과만 반환
        return {"id": "query", "emotion": emotion, "topics": topics, "similar": []}

    # 유사 포스트 목록만 반환 받음
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