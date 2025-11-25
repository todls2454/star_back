import os
import json
import numpy as np
from typing import List, Dict, Any, Optional

# [필요 라이브러리]
# pip install fastapi uvicorn firebase-admin numpy scipy pydantic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.spatial.distance import cosine
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import FieldFilter 

# --- 설정 및 초기화 ---

POSTS_COLLECTION = "posts"
RECOMMENDATION_THRESHOLD = 0.4  # 코사인 유사도 임계값
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
        print("Test FastAPI: Firebase Admin SDK 초기화 완료 (환경 변수 사용).")
    elif firebase_admin._apps:
        db = firestore.client()
    else:
        print("FATAL: FIREBASE_SERVICE_ACCOUNT_JSON 환경 변수가 설정되지 않았습니다.")
        db = None 
        
except Exception as e:
    print(f"FATAL: Firebase 초기화 실패. 오류: {e}")
    db = None

app = FastAPI(
    title="STELLINK Test Recommendation API",
    description="임베딩 벡터를 직접 입력받아 추천 알고리즘을 테스트합니다."
)

# --- 요청 본문 모델 정의 ---

class TestRecommendationRequest(BaseModel):
    """테스트용 추천 요청 시 입력받는 데이터 모델."""
    user_id: str
    new_post_id: str 
    new_topic_vector: List[float]
    new_content_vector: List[float]
    new_emotion_vector: List[float]

# --- 유틸리티 함수 ---

def calculate_cosine_similarity(vec_a: Optional[List[float]], vec_b: Optional[List[float]]) -> float:
    """
    두 임베딩 벡터 간의 코사인 유사도를 계산합니다.
    """
    if not vec_a or not vec_b:
        return 0.0
    
    try:
        if len(vec_a) != len(vec_b):
             print(f"WARN: 벡터 차원이 일치하지 않아 유사도 계산 불가. ({len(vec_a)} vs {len(vec_b)})")
             return 0.0
             
        similarity = 1 - cosine(np.array(vec_a), np.array(vec_b))
        return float(similarity)
    except Exception as e:
        print(f"ERROR: 코사인 유사도 계산 중 오류 발생: {e}")
        return 0.0

# --- API 엔드포인트 ---

@app.post(
    "/test-recommendations", 
    summary="테스트용 추천 (벡터 직접 입력)",
    response_model=List[Dict[str, Any]]
)
async def get_test_recommendations(request: TestRecommendationRequest):
    """
    요청 본문으로 받은 임베딩 벡터를 기준으로, 모든 공개 포스트 중 
    유사도가 높은 기록을 추천합니다. (user_id 필터 제거)
    """
    
    if db is None:
        raise HTTPException(status_code=500, detail="서버 설정 오류: Firebase 데이터베이스가 초기화되지 않았습니다.")

    # 1. 요청 데이터 파싱 (user_id는 더 이상 쿼리 필터로 사용되지 않음)
    new_post_id = request.new_post_id
    
    # 새 포스트의 임베딩 벡터 (입력값)
    new_topic_vector = request.new_topic_vector
    new_content_vector = request.new_content_vector
    new_emotion_vector = request.new_emotion_vector
    
    # 입력 벡터 중 하나라도 없으면 오류 처리
    if not new_topic_vector and not new_content_vector and not new_emotion_vector:
        raise HTTPException(status_code=400, detail="입력 벡터(topic, content, emotion) 중 최소 하나는 제공되어야 합니다.")

    # 2. 모든 공개 포스트 가져오기 (visibility=PUBLIC 필터 사용)
    try:
        past_posts_query = db.collection(POSTS_COLLECTION).where(filter=FieldFilter('visibility', '==', 'PUBLIC')).stream()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Firestore 과거 포스트 쿼리 중 오류 발생. 'visibility' 필드에 인덱스가 필요할 수 있습니다: {e}")

    recommendations = []
    past_posts_count = 0

    for doc in past_posts_query:
        past_posts_count += 1
        past_post_id = doc.id
        
        if past_post_id == new_post_id: 
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

        # 4. 임계값(0.4) 이상인 포스트만 후보에 추가
        if combined_similarity >= RECOMMENDATION_THRESHOLD:
            recommendations.append({
                "post_id": past_post_id,
                "similarity": round(combined_similarity, 4), 
                "topic": past_data.get('topic', 'N/A'),
                "emotion_tag": past_data.get('emotion_tag', 'N/A'),
                "original_content": past_data.get('original_content', 'N/A'),
                "archive_tags": past_data.get('archive_tags', []),
            })

    print(f"DEBUG: 총 쿼리된 과거 포스트 수: {past_posts_count}")
    
    # 5. 유사도 기준 내림차순 정렬
    recommendations.sort(key=lambda x: x['similarity'], reverse=True)
    
    # 6. Top N 결과 선택
    final_recommendations = recommendations[:MAX_RECOMMENDATIONS]
    
    print(f"추천 완료: {new_post_id} (Test)에 대해 {len(final_recommendations)}개 추천. (임계값 {RECOMMENDATION_THRESHOLD})")
    
    return final_recommendations
