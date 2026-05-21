
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
z = np.arange(-50 , 50 , 0.1 ) # -5 부터 5까지 1씩 증가하는 리스트  
phi = 1 / ( 1+ np.exp(-z)) # 시그모이드 공식
plt.plot(z, phi) # 시그모이드 시각화
plt.show()


# [2] 이진 분류 ,  # 로지스틱 회귀 모델
# 이진 분류는 0 또는 1 로 분류하는 방법 
indexs =(train_target == 'Bream') | (train_target == 'Smelt')  # 도미와 빙어만 추출
print (indexs) 
train_bream_smelt = train_scaled[indexs]
target_bream_smelt = train_target[indexs]
print(train_bream_smelt)

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

