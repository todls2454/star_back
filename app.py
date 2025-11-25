import os
import json
import time
from typing import List, Dict, Any, Optional

# [필요 라이브러리]
# pip install fastapi uvicorn firebase-admin pydantic requests google-generativeai
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import requests

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import FieldFilter

# ✅ [추가] Gemini SDK (가벼움)
import google.generativeai as genai

# --- 설정 및 초기화 ---

SERVICE_ACCOUNT_JSON_STR = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
POSTS_COLLECTION = "posts"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") 

# Gemini 모델 설정
POST_ANALYSIS_MODEL = "gemini-1.5-flash" # 2.5가 아직 불안정하다면 1.5 사용 권장
EMBEDDING_MODEL = "models/text-embedding-004" # Gemini 전용 임베딩 모델

# ✅ Gemini 초기화
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

db = None

# 추천 설정
RECOMMENDATION_THRESHOLD = 0.6
MAX_RECOMMENDATIONS = 8

app = FastAPI(title="STELLINK Lightweight API")

# --- 1. 초기화 함수 (이제 모델 로딩 없음!) ---

@app.on_event("startup")
def startup_event():
    """서버 시작 시 Firebase만 가볍게 연결합니다."""
    print("INFO: 서버 시작 중... Firebase 연결 시도")
    initialize_firebase()

def initialize_firebase():
    global db
    try:
        if SERVICE_ACCOUNT_JSON_STR and not firebase_admin._apps:
            service_account_info = json.loads(SERVICE_ACCOUNT_JSON_STR)
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("INFO: Firebase 초기화 완료.")
        elif firebase_admin._apps:
            db = firestore.client()
        else:
            print("WARN: Firebase 설정 없음")
            db = None
    except Exception as e:
        print(f"FATAL: Firebase 초기화 실패: {e}")
        db = None

# --- 2. 요청 모델 ---

class NewPostRequest(BaseModel):
    user_id: str
    original_content: str
    generated_content: Optional[str] = None
    topic: Optional[str] = None
    emotion_tag: Optional[str] = None
    archive_tags: List[str] = []
    media_urls: List[str] = []
    visibility: str = "PUBLIC"
    constellation_tag: Optional[str] = None
    content_format: str = "narrative"
    comment_count: int = 0
    like_count: int = 0
    is_deleted: bool = False
    is_pinned: bool = False

# --- 3. AI 로직 (Gemini로 통합) ---

def extract_topic_and_emotion(original_content: str) -> Dict[str, str]:
    """Gemini를 이용해 주제/감정 추출 (기존 로직 유지)"""
    if not GEMINI_API_KEY:
        return {"topic": "KEY_ERROR", "emotion_tag": "ERROR"}

    try:
        model = genai.GenerativeModel(POST_ANALYSIS_MODEL)
        prompt = (
            f"다음 일기를 분석해 주제(topic) 한 문장과 감정(emotion_tag) 단어 하나를 JSON으로 추출해.\n"
            f"일기: {original_content}\n"
            f"형식: {{\"topic\": \"...\", \"emotion_tag\": \"...\"}}"
        )
        response = model.generate_content(prompt)
        
        # 간단한 파싱 (JSON 포맷 강제화가 안될 경우를 대비해 텍스트 정리)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"Gemini Analysis Error: {e}")
        return {"topic": "분석 실패", "emotion_tag": "모름"}

def get_embedding(text: str) -> Optional[List[float]]:
    """✅ [변경됨] 로컬 모델 대신 Gemini API로 임베딩 생성 (메모리 절약)"""
    if not GEMINI_API_KEY:
        return None
    try:
        # Gemini 임베딩 API 호출
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="clustering",
        )
        return result['embedding']
    except Exception as e:
        print(f"Gemini Embedding Error: {e}")
        return None

def calculate_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """순수 Python으로 코사인 유사도 계산 (Scipy/Numpy 제거로 가볍게)"""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    
    # 내적 (Dot Product)
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    # 크기 (Magnitude)
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)

def _perform_recommendation_logic(
    user_id: str,
    new_topic_vec: Optional[List[float]],
    new_content_vec: Optional[List[float]],
    new_emotion_vec: Optional[List[float]],
    current_post_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    
    if not db: return []
    
    # 공개 포스트 가져오기
    docs = db.collection(POSTS_COLLECTION).where(filter=FieldFilter('visibility', '==', 'PUBLIC')).stream()
    
    recommendations = []
    
    for doc in docs:
        if doc.id == current_post_id: continue
        data = doc.to_dict()
        
        # 하나라도 벡터가 있어야 함
        past_topic_vec = data.get('topic_vector')
        if not past_topic_vec: continue 
        
        # 유사도 계산 (topic 벡터만 사용해도 충분히 가볍고 정확함)
        # 비용 절약을 위해 Topic 벡터끼리만 비교하는 것을 추천
        sim = 0.0
        count = 0
        
        if new_topic_vec and past_topic_vec:
            sim += calculate_cosine_similarity(new_topic_vec, past_topic_vec)
            count += 1
            
        # 필요한 경우 content, emotion 벡터도 위와 같이 계산
            
        if count > 0:
            final_sim = sim / count
            if final_sim >= RECOMMENDATION_THRESHOLD:
                recommendations.append({
                    "post_id": doc.id,
                    "similarity": round(final_sim, 4),
                    "topic": data.get('topic', ''),
                    "emotion_tag": data.get('emotion_tag', ''),
                    "original_content": data.get('original_content', ''),
                })
    
    recommendations.sort(key=lambda x: x['similarity'], reverse=True)
    return recommendations[:MAX_RECOMMENDATIONS]


# --- 엔드포인트 ---

@app.get("/health")
def health_check():
    return {"status": "Running", "backend": "Lightweight Gemini Mode"}

@app.post("/process-and-save-post", status_code=status.HTTP_201_CREATED)
async def process_and_save_post(request: NewPostRequest):
    if not db: raise HTTPException(500, "DB 미연결")
    
    # 1. 분석
    if not request.topic or not request.emotion_tag:
        analysis = extract_topic_and_emotion(request.original_content)
        topic = analysis.get('topic', '무제')
        emotion_tag = analysis.get('emotion_tag', '무감정')
    else:
        topic = request.topic
        emotion_tag = request.emotion_tag
        
    # 2. 임베딩 (Gemini API 사용)
    # 텍스트 하나로 합쳐서 한 번만 임베딩하는 게 비용/속도면에서 유리함
    combined_text = f"주제: {topic}. 감정: {emotion_tag}. 내용: {request.original_content}"
    
    # 여기서는 기존 구조 유지를 위해 3번 호출하지만, 실제론 줄이는 게 좋습니다.
    topic_vector = get_embedding(combined_text) 
    
    post_data = {
        'user_id': request.user_id,
        'original_content': request.original_content,
        'topic': topic,
        'emotion_tag': emotion_tag,
        'archive_tags': request.archive_tags,
        'visibility': request.visibility,
        'topic_vector': topic_vector, # 주요 벡터
        # 나머지 벡터는 생략하거나 동일하게 처리
        'created_at': firestore.SERVER_TIMESTAMP,
    }
    
    doc_ref = db.collection(POSTS_COLLECTION).document()
    doc_ref.set(post_data)
    
    return {"post_id": doc_ref.id, "message": "Saved"}

@app.get("/recommendations/{post_id}")
async def get_recommendations(post_id: str):
    if not db: raise HTTPException(500, "DB 미연결")
    
    doc = db.collection(POSTS_COLLECTION).document(post_id).get()
    if not doc.exists: raise HTTPException(404, "Not Found")
    
    data = doc.to_dict()
    return _perform_recommendation_logic(
        data.get('user_id'),
        data.get('topic_vector'),
        None, None, 
        current_post_id=post_id
    )
