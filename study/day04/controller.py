# app.py 연결하는 라우터 설정
from fastapi import APIRouter,Request,Response
router = APIRouter( prefix = '/api/model')


# 서비스 호출
import service
# 매핑 만들기
@router.post("/learn") # 매핑 주소 생성
async def 학습요청 ( request : Request) : # body 정보를 list 타입으로 받기
    list = await request.json() # request 객체 body 값을 직접 json 변환 
    print(list)
    return service.service2.학습요청(list)

@router.post("/predict")
async def 예측요청 (car : dict) : # body 정보를 딕셔너리 타입으로 받기
    print(car)
    return service.service2.예측요청(car)