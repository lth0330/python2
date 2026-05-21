from fastapi import APIRouter,Request,Response
router = APIRouter( prefix = '/api/model')

import service

# 핵심 변수(나이, 성별, 유입 채널, 선호 스타일)를 입력하여 요청
@router.post("/learn") 
async def Purchasing_learn ( request : Request) : 
    list = await request.json()
    print(list)
    return service.service.Purchasing_learn(list)

# 모델이 예측한 구매 옷 카테고리(0~7)를 실시간으로 반환한다.
@router.post("/predict")
async def Purchasing_predict (clothes : dict) : 
    print(clothes)
    return service.service.Purchasing_predict(clothes)