import os
import json
import numpy as np
import time
from typing import List, Dict, Any, Optional

# [필요 라이브러리]
# pip install fastapi uvicorn firebase-admin numpy scipy
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from scipy.spatial.distance import cosine
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import FieldFilter 

# --- 설정 및 초기화 ---

POSTS_COLLECTION = "posts"
RECOMMENDATION_THRESHOLD = 0.6  # 코사인 유사도 임계값
MIN_RECOMMENDATIONS = 3        # 최소 추천 개수
MAX_RECOMMENDATIONS = 8        # 최대 추천 개수

# 1. Firebase Admin SDK 초기화 (JSON 환경 변수 사용)
try:
    # Render 환경 변수에서 서비스 계정 JSON 문자열을 가져옵니다.
    SERVICE_ACCOUNT_JSON_STR = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    
    if SERVICE_ACCOUNT_JSON_STR and not firebase_admin._apps:
        # JSON 문자열을 딕셔너리로 로드합니다.
        service_account_info = json.loads(SERVICE_ACCOUNT_JSON_STR)
        
        # 딕셔너리 정보로 인증서를 초기화합니다. (파일 경로 불필요)
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("FastAPI: Firebase Admin SDK 초기화 완료 (환경 변수 사용).")
    elif firebase_admin._apps:
        db = firestore.client()
    else:
        print("FATAL: FIREBASE_SERVICE_ACCOUNT_JSON 환경 변수가 설정되지 않았습니다.")
        db = None 
        
except Exception as e:
    print(f"FATAL: Firebase 초기화 실패. 오류: {e}")
    db = None

app = FastAPI(
    title="STELLINK Recommendation API (Production)",
    description="사용자 포스트의 임베딩 벡터를 기반으로 유사한 기록을 추천합니다."
)

# --- 유틸리티 함수 ---

def calculate_cosine_similarity(vec_a: Optional[List[float]], vec_b: Optional[List[float]]) -> float:
    """
    두 임베딩 벡터 간의 코사인 유사도를 계산합니다.
    """
    if not vec_a or not vec_b:
        return 0.0
    
    try:
        # 벡터 차원 검사 (필수)
        if len(vec_a) != len(vec_b):
             print(f"WARN: 벡터 차원이 일치하지 않아 유사도 계산 불가. ({len(vec_a)} vs {len(vec_b)})")
             return 0.0
             
        # 1 - cosine distance (scipy)
        similarity = 1 - cosine(np.array(vec_a), np.array(vec_b))
        return float(similarity)
    except Exception as e:
        print(f"ERROR: 코사인 유사도 계산 중 오류 발생: {e}")
        return 0.0

# --- 핵심 추천 로직 (재사용을 위해 분리) ---
def _perform_recommendation_logic(
    user_id: str,
    new_topic_vector: Optional[List[float]],
    new_content_vector: Optional[List[float]],
    new_emotion_vector: Optional[List[float]],
    current_post_id: Optional[str] = None # post_id로 조회 시 제외할 ID
) -> List[Dict[str, Any]]:
    """
    주어진 벡터들을 기준으로 Firestore에서 추천을 수행하는 내부 함수.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="서버 설정 오류: Firebase 데이터베이스가 초기화되지 않았습니다.")

    if not new_topic_vector and not new_content_vector and not new_emotion_vector:
         raise HTTPException(status_code=400, detail="유효한 임베딩 벡터가 최소 하나 이상 필요합니다.")

    # 2. 모든 공개 포스트 가져오기 (visibility=PUBLIC 필터 사용)
    try:
        # NOTE: 모든 사용자의 공개 포스트를 쿼리하도록 수정 (visibility=PUBLIC)
        past_posts_query = db.collection(POSTS_COLLECTION).where(filter=FieldFilter('visibility', '==', 'PUBLIC')).stream()
    except Exception as e:
        # NOTE: 이 쿼리를 실행하려면 'visibility' 필드에 대한 인덱스가 Firestore에 필요합니다.
        raise HTTPException(status_code=500, detail=f"Firestore 과거 포스트 쿼리 중 오류 발생. 'visibility' 필드에 인덱스가 필요할 수 있습니다: {e}")

    recommendations = []
    past_posts_count = 0

    for doc in past_posts_query:
        past_posts_count += 1
        past_post_id = doc.id
        
        # 새 포스트 ID는 반드시 제외
        if past_post_id == current_post_id: 
            continue
            
        past_data = doc.to_dict()
        
        # 과거 포스트의 임베딩 벡터
        past_topic_vector = past_data.get('topic_vector')
        past_content_vector = past_data.get('original_content_vector')
        past_emotion_vector = past_data.get('emotion_vector')

        # 과거 포스트에 유효한 벡터가 하나도 없으면 건너뛰기
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
            
        # 3-4. 최종 유사도 계산: 유효한 유사도 점수들의 평균
        combined_similarity = sum(similarity_scores) / len(similarity_scores)

        # 4. 임계값(0.6) 이상인 포스트만 후보에 추가
        if combined_similarity >= RECOMMENDATION_THRESHOLD:
            recommendations.append({
                "post_id": past_post_id,
                "similarity": round(combined_similarity, 4), 
                "topic": past_data.get('topic', 'N/A'),
                "emotion_tag": past_data.get('emotion_tag', 'N/A'),
                # --- Flutter 모델에서 필요로 하는 필드를 추가합니다 ---
                "original_content": past_data.get('original_content', '내용을 불러올 수 없습니다.'),
                "archive_tags": past_data.get('archive_tags', []),
                # ----------------------------------------------------
            })

    # 5. 유사도 기준 내림차순 정렬 및 Top N 선택
    recommendations.sort(key=lambda x: x['similarity'], reverse=True)
    
    final_recommendations = recommendations[:MAX_RECOMMENDATIONS]
    
    return final_recommendations


# ----------------------------------------------------------------------
# 엔드포인트: POST ID 기반 추천 (Flutter 앱이 사용할 주 엔드포인트)
# ----------------------------------------------------------------------

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
