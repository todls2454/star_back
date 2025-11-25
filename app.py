import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional

# [필요 라이브러리]
# pip install fastapi uvicorn firebase-admin numpy scipy pydantic requests sentence-transformers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import requests

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import FieldFilter 

# --- 설정 및 초기화 ---

# Firebase Admin SDK 초기화 (JSON 환경 변수 사용)
SERVICE_ACCOUNT_JSON_STR = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
POSTS_COLLECTION = "posts"

# Gemini API 설정
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") 
POST_ANALYSIS_MODEL = "gemini-2.5-flash" 

# Hugging Face 모델 설정
HUGGINGFACE_MODEL_NAME = "snunlp/KR-ELECTRA-discriminator" 
embedding_model_instance = None
db = None

# 추천 설정
RECOMMENDATION_THRESHOLD = 0.6  # 코사인 유사도 임계값
MAX_RECOMMENDATIONS = 8        # 최대 추천 개수


app = FastAPI(title="STELLINK Post Processor and Recommendation API")


# --- 1. 모델 및 DB 초기화 함수 ---

def load_embedding_model() -> Optional[SentenceTransformer]:
    """SentenceTransformer 모델을 전역적으로 로드하여 재사용합니다."""
    global embedding_model_instance
    if embedding_model_instance is None:
        try:
            print(f"INFO: Loading Hugging Face Embedding Model: {HUGGINGFACE_MODEL_NAME}...")
            embedding_model_instance = SentenceTransformer(HUGGINGFACE_MODEL_NAME)
            print("INFO: Model loaded successfully.")
        except Exception as e:
            print(f"FATAL: Hugging Face model 로드 실패: {e}")
            return None
    return embedding_model_instance

def initialize_firebase():
    """Firebase Admin SDK 초기화 및 Firestore 클라이언트 설정."""
    global db
    try:
        if SERVICE_ACCOUNT_JSON_STR and not firebase_admin._apps:
            service_account_info = json.loads(SERVICE_ACCOUNT_JSON_STR)
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("INFO: Firebase Admin SDK 초기화 완료.")
        elif firebase_admin._apps:
            db = firestore.client()
        else:
            print("WARN: FIREBASE_SERVICE_ACCOUNT_JSON 환경 변수가 설정되지 않았습니다.")
            db = None
    except Exception as e:
        print(f"FATAL: Firebase 초기화 실패. 오류: {e}")
        db = None

# 서버 시작 시 모델과 DB를 초기화합니다.
@app.on_event("startup")
def startup_event():
    load_embedding_model()
    initialize_firebase()


# --- 2. Pydantic 요청 모델 (Flutter에서 받을 데이터) ---

class NewPostRequest(BaseModel):
    """Flutter 앱에서 POST 요청 시 전달되는 모든 데이터."""
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


# --- 3. AI/ML 및 유틸리티 로직 함수 ---

def extract_topic_and_emotion(original_content: str) -> Dict[str, str]:
    """Gemini API를 호출하여 주제와 감정을 추출합니다."""
    if not GEMINI_API_KEY:
        return {"topic": "API 키 오류", "emotion_tag": "오류"}

    # (Gemini API 호출 로직은 이전 파일과 동일합니다.)
    system_prompt = (
        "당신은 사용자의 일기/포스트를 분석하는 AI 분석가입니다. "
        "다음 한국어 일기의 핵심 주제(topic) 한 줄과, 주된 감정(emotion_tag) 한 단어를 "
        "요청된 JSON 형식으로 추출하여 반환해야 합니다. "
        "주제는 문장 형태, 감정은 명사 형태의 단어 하나여야 합니다."
    )
    user_query = f"다음 일기 내용을 분석해 주제와 감정을 추출해 주세요: {original_content}"
    
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "config": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {"topic": {"type": "STRING"}, "emotion_tag": {"type": "STRING"}},
                "propertyOrdering": ["topic", "emotion_tag"]
            }
        }
    }

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{POST_ANALYSIS_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    try:
        response = requests.post(api_url, json=payload, timeout=20)
        response.raise_for_status()
        
        result = response.json()
        json_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
        parsed_json = json.loads(json_text)
        
        return {
            "topic": parsed_json.get("topic", "주제 추출 실패"),
            "emotion_tag": parsed_json.get("emotion_tag", "감정 추출 실패")
        }
    except Exception as e:
        print(f"ERROR: Gemini API 호출 실패 - {e}")
        return {"topic": "API 오류", "emotion_tag": "오류"}

def get_embedding(text_to_embed: str) -> Optional[List[float]]:
    """Hugging Face SentenceTransformer를 사용하여 텍스트 임베딩 벡터를 생성합니다."""
    model = load_embedding_model()
    if model is None:
        return None
        
    try:
        embeddings = model.encode(text_to_embed, convert_to_numpy=True)
        return embeddings.tolist()
    except Exception as e:
        print(f"ERROR: 임베딩 계산 중 오류 발생: {e}")
        return None

def calculate_cosine_similarity(vec_a: Optional[List[float]], vec_b: Optional[List[float]]) -> float:
    """두 임베딩 벡터 간의 코사인 유사도를 계산합니다."""
    if not vec_a or not vec_b:
        return 0.0
    
    try:
        if len(vec_a) != len(vec_b):
             return 0.0
             
        similarity = 1 - cosine(np.array(vec_a), np.array(vec_b))
        return float(similarity)
    except Exception:
        return 0.0

def _perform_recommendation_logic(
    user_id: str,
    new_topic_vector: Optional[List[float]],
    new_content_vector: Optional[List[float]],
    new_emotion_vector: Optional[List[float]],
    current_post_id: Optional[str] = None 
) -> List[Dict[str, Any]]:
    """주어진 벡터들을 기준으로 Firestore에서 추천을 수행하는 내부 함수."""
    if db is None:
        raise HTTPException(status_code=500, detail="서버 설정 오류: Firebase 데이터베이스가 초기화되지 않았습니다.")

    if not new_topic_vector and not new_content_vector and not new_emotion_vector:
         raise HTTPException(status_code=400, detail="유효한 임베딩 벡터가 최소 하나 이상 필요합니다.")

    # 2. 모든 공개 포스트 가져오기 (visibility=PUBLIC 필터 사용)
    try:
        past_posts_query = db.collection(POSTS_COLLECTION).where(filter=FieldFilter('visibility', '==', 'PUBLIC')).stream()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Firestore 과거 포스트 쿼리 중 오류 발생: {e}")

    recommendations = []

    for doc in past_posts_query:
        past_post_id = doc.id
        if past_post_id == current_post_id: 
            continue
            
        past_data = doc.to_dict()
        
        # 과거 포스트의 임베딩 벡터
        past_topic_vector = past_data.get('topic_vector')
        past_content_vector = past_data.get('original_content_vector')
        past_emotion_vector = past_data.get('emotion_vector')

        if not past_topic_vector and not past_content_vector and not past_emotion_vector:
            continue

        # 3. 코사인 유사도 계산 및 결합 (3가지 벡터 모두 사용)
        similarity_scores = []

        # 3-1. Topic Vector 유사도
        if new_topic_vector:
            topic_similarity = calculate_cosine_similarity(new_topic_vector, past_topic_vector)
            if topic_similarity > 0: similarity_scores.append(topic_similarity)

        # 3-2. Original Content Vector 유사도
        if new_content_vector:
            content_similarity = calculate_cosine_similarity(new_content_vector, past_content_vector)
            if content_similarity > 0: similarity_scores.append(content_similarity)
        
        # 3-3. Emotion Vector 유사도
        if new_emotion_vector:
            emotion_similarity = calculate_cosine_similarity(new_emotion_vector, past_emotion_vector)
            if emotion_similarity > 0: similarity_scores.append(emotion_similarity)
        
        if not similarity_scores:
            continue
            
        combined_similarity = sum(similarity_scores) / len(similarity_scores)

        # 4. 임계값(0.6) 이상인 포스트만 후보에 추가
        if combined_similarity >= RECOMMENDATION_THRESHOLD:
            recommendations.append({
                "post_id": past_post_id,
                "similarity": round(combined_similarity, 4), 
                "topic": past_data.get('topic', 'N/A'),
                "emotion_tag": past_data.get('emotion_tag', 'N/A'),
                "original_content": past_data.get('original_content', '내용을 불러올 수 없습니다.'),
                "archive_tags": past_data.get('archive_tags', []),
            })

    # 5. 유사도 기준 내림차순 정렬 및 Top N 선택
    recommendations.sort(key=lambda x: x['similarity'], reverse=True)
    
    final_recommendations = recommendations[:MAX_RECOMMENDATIONS]
    
    return final_recommendations


# --------------------------------------------------------------------------------
# [0단계] 상태 확인 엔드포인트 (GET /health)
# --------------------------------------------------------------------------------
@app.get("/health", summary="서버 상태 확인")
def health_check():
    """서버의 상태와 주요 컴포넌트의 로드 상태를 반환합니다."""
    db_status = "OK" if db is not None else "ERROR"
    model_status = "OK" if embedding_model_instance is not None else "ERROR"
    
    if db_status == "OK" and model_status == "OK":
        status_code = status.HTTP_200_OK
    else:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "Running" if status_code == status.HTTP_200_OK else "Degraded",
        "db_connection": db_status,
        "embedding_model": model_status,
        "model_name": HUGGINGFACE_MODEL_NAME,
    }, status_code


# --------------------------------------------------------------------------------
# [1단계] 엔드포인트: AI 처리 및 저장 (POST /process-and-save-post)
# --------------------------------------------------------------------------------

@app.post("/process-and-save-post", status_code=status.HTTP_201_CREATED)
async def process_and_save_post(request: NewPostRequest):
    """
    Flutter 앱으로부터 포스트 데이터를 받아, AI로 분석하고 임베딩을 생성한 후 
    Firestore에 저장하는 통합 엔드포인트입니다.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="서버 설정 오류: Firebase 데이터베이스가 초기화되지 않았습니다.")
    
    # 4-1. 메타데이터 추출 (Gemini)
    if not request.topic or not request.emotion_tag:
        print("INFO: topic/emotion_tag 누락, Gemini로 추출 시작...")
        extraction_result = extract_topic_and_emotion(request.original_content)
        
        topic = extraction_result['topic']
        emotion_tag = extraction_result['emotion_tag']
    else:
        topic = request.topic
        emotion_tag = request.emotion_tag
    
    # 4-2. 임베딩 입력 텍스트 준비
    tags_str = ", ".join(request.archive_tags)
    
    # 텍스트 1: Topic Vector 생성을 위한 메타데이터 결합
    topic_embed_text = f"주제: {topic}. 감정: {emotion_tag}. 태그: {tags_str}."
    if request.generated_content:
        topic_embed_text += f" 가공된 내용: {request.generated_content}"

    # 텍스트 2: Original Content Vector 생성을 위한 순수 본문
    content_embed_text = request.original_content
    
    # 텍스트 3: Emotion Vector 생성을 위한 감성/주제 결합
    emotion_embed_text = f"감정: {emotion_tag}. 주제: {topic}. 본문: {request.original_content[:100]}..."

    # 4-3. 임베딩 계산 (Hugging Face)
    print("INFO: 임베딩 계산 시작...")
    topic_vector = get_embedding(topic_embed_text)
    content_vector = get_embedding(content_embed_text)
    emotion_vector = get_embedding(emotion_embed_text)
    
    if not topic_vector or not content_vector or not emotion_vector:
        raise HTTPException(status_code=500, detail="임베딩 벡터 생성 실패. 모델 로드 상태를 확인하세요.")

    # 4-4. Firestore 문서 구조 완성
    post_data = {
        'user_id': request.user_id,
        'original_content': request.original_content,
        'generated_content': request.generated_content or '',
        'topic': topic, # 추출된 값
        'emotion_tag': emotion_tag, # 추출된 값
        'archive_tags': request.archive_tags,
        'media_urls': request.media_urls,
        'visibility': request.visibility,
        'constellation_tag': request.constellation_tag,
        'content_format': request.content_format,
        'comment_count': request.comment_count,
        'like_count': request.like_count,
        'is_deleted': request.is_deleted,
        'is_pinned': request.is_pinned,
        
        # AI/ML 결과 저장
        'topic_vector': topic_vector,
        'original_content_vector': content_vector,
        'emotion_vector': emotion_vector,
        
        # 타임스탬프
        'created_at': firestore.SERVER_TIMESTAMP,
        'updated_at': firestore.SERVER_TIMESTAMP,
    }

    # 4-5. Firestore에 저장
    try:
        doc_ref = db.collection(POSTS_COLLECTION).document()
        doc_ref.set(post_data)
        
        print(f"SUCCESS: 새 포스트 저장 완료. ID: {doc_ref.id}")
        
        return {"post_id": doc_ref.id, "message": "Post successfully processed and saved."}

    except Exception as e:
        print(f"FATAL: Firestore 저장 실패. 오류: {e}")
        raise HTTPException(status_code=500, detail="데이터베이스 저장 중 오류가 발생했습니다.")


# --------------------------------------------------------------------------------
# [2단계] 엔드포인트: POST ID 기반 추천 (GET /recommendations/{post_id})
# --------------------------------------------------------------------------------

@app.get(
    "/recommendations/{post_id}", 
    summary="아이템 기반 추천 (POST ID 조회)",
    response_model=List[Dict[str, Any]]
)
async def get_recommendations_by_id(post_id: str):
    """
    Flutter 앱에서 저장된 post_id를 전송하면, 해당 포스트의 벡터를 조회하여 추천합니다.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="서버 설정 오류: Firebase 데이터베이스가 초기화되지 않았습니다.")

    try:
        new_post_doc = db.collection(POSTS_COLLECTION).document(post_id).get()
        if not new_post_doc.exists:
            raise HTTPException(status_code=404, detail="새 포스트 ID를 찾을 수 없습니다.")
        
        new_post_data = new_post_doc.to_dict()
        user_id = new_post_data.get('user_id')
        
        new_topic_vector = new_post_data.get('topic_vector')
        new_content_vector = new_post_data.get('original_content_vector')
        new_emotion_vector = new_post_data.get('emotion_vector')
        
        if not user_id or (not new_topic_vector and not new_content_vector and not new_emotion_vector):
            raise HTTPException(status_code=400, detail="포스트에 user_id 또는 유효한 임베딩 벡터가 누락되었습니다.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Firestore 데이터 로드 중 오류 발생: {e}")

    recommendations = _perform_recommendation_logic(
        user_id,
        new_topic_vector,
        new_content_vector,
        new_emotion_vector,
        current_post_id=post_id # 현재 포스트 제외
    )
    
    print(f"추천 완료 (ID 기반): {post_id}에 대해 {len(recommendations)}개 추천.")
    return recommendations
