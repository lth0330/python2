
# [1] 여러가지 특성에 따른 분류 모델 
import pandas as pd
df = pd.read_csv('study/day01/Fish.csv')

# 어종 7개 , Species 
fish_target = df['Species']

# 특성 6개 , Weight,Length1,Length2,Length3,Height,Width
fish_input = df[ ['Weight','Length1','Length2','Length3','Height','Width']]

# 훈련 / 테이스 분리
from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split(fish_input, fish_target, random_state=42)

# 스케일링 
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit (train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 로지스틱 회귀 = 이진분류 = 시그모이드 함수(공식)
# 선형 방정식의 출력값을 0과 1 (확률) 사이의 값으로 변환해주는 공식 / 함수
# 예시] 이진 분류 알고리즘 사용 
# 즉] 컴퓨터는 수치상의 150 또는 -80 이라는 값으로 확률이 어렵다.
import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5 , 5 , 0.1 ) # -5 부터 5까지 1씩 증가하는 리스트  
phi = 1 / ( 1+ np.exp(-z)) # 시그모이드 공식
plt.plot(z, phi) # 시그모이드 시각화
# plt.show()


# [2] 이진 분류 ,  # 로지스틱 회귀 모델
# 이진 분류는 0 또는 1 로 분류하는 방법 
indexs =(train_target == 'Bream') | (train_target == 'Smelt')  # 도미와 빙어만 추출
# print (indexs) 
train_bream_smelt = train_scaled[indexs]
target_bream_smelt = train_target[indexs]
# print(train_bream_smelt)

# 이진분류 모델 구현 
from sklearn.linear_model import LogisticRegression  #
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

# 이진분류 모델 예측
print( lr.predict(train_bream_smelt[:3])) # 3개만 예측  ['Bream' 'Smelt' 'Bream']
print(lr.predict_proba(train_bream_smelt[:3]))  # 3개만 예측 확률 [[도미의 확률 , 빙어의 확률 ]] 
# 임계값이 0.5 기준으로 0.5 이상이면 도미로 예측 ,0.5 미만 이면 빙어로 예측 
# [[0.99793611 0.00206389]
# [0.02391315 0.97608685]
# [0.99575505 0.00424495]]

# [3] 다중 분류  # 로지스틱 회귀
# 하이퍼파라미터 
# C = 규제를 완화하여 릿지/라쏘 모델처럼 정확도 설정 가능하다.  # 모델의 성능을 향상 하기 위해서 가중치 값들을 자동 조정  
# max_iter = 다중분류 계산 회수  # (생략시) 기본값 100 으로 최적의 정확도를 찾을 때 까지 계산 반복회수 조정 # 넉넉하게 
lr = LogisticRegression(C = 10 , max_iter=1000)
lr.fit(train_scaled, train_target)  # 모든 어종 학습

# 모델 예측
print(lr.predict(test_scaled[:3])) # 3개만 예측 ['Perch' 'Smelt' 'Pike']
print(lr.predict_proba(test_scaled[:3])) # 3개만 예측 확률  # 분류개수 만큼의 확률

# 모델 평가 , 선형 회귀와 다르게 *결정 계수* 가 아닌 맞힌 *비율(정확도)* 반환 
print(lr.score(test_scaled,test_target))  # 0.85  , C = 10 이상일때 0.925

# 소프트맥스
from scipy.special import softmax # 소프트맥스
decision = lr.decision_function(test_scaled[:3])
print(softmax(decision))                            # 소프트맥스 라는 함수로 결과값을 확인 했을 때 predict 과 동일하게 출력된다.
print(np.round(softmax(decision), decimals=3))      # np.round (값 , decimals = 소수점)
# [[0.    0.002 0.055 0.    0.014 0.001 0.   ]
# [0.    0.004 0.046 0.    0.009 0.608 0.   ]
# [0.    0.    0.014 0.237 0.007 0.002 0.   ]

# 다중분류으 확률을 검증할때는 .classes_ 종속변수들의 순서 확인 
print(lr.classes_)   # ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']