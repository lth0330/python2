

# T1-01.py
# 캐글의 데이터셋 :https://www.kaggle.com/datasets/vipullrathod/fish-market/data

# [1] csv 불러오기
import pandas as pd

df = pd.read_csv('study/day01/Fish.csv')
# print(df)

# [2] 특정한 몰고기 추출  도미"Bream"
bream_df = df[df['Species'] == 'Bream']

# [3] 도미의 길이 , 무게 추출  , df['열이름'].tolist() : 리스트타입으로 반환 
bream_length = bream_df['Length2'].tolist()  # 도미의 길이
bream_weight = bream_df['Weight'].tolist()   # 도미의 무게
# print(bream_length,bream_weight)

#[4] 특정한 물고기 추출2   빙어"Smelt"
smelt_df = df[df['Species']== 'Smelt']
smelt_length = smelt_df['Length2'].tolist()   
smelt_weight = smelt_df['Weight'].tolist()  

# [4] 시각화
import matplotlib.pyplot as plt
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length(cm)')
plt.ylabel('weight(kg)')
# plt.show()


# [5] 도미와 빙어 자료 합치기 (길이 , 무게) 각자 합치기
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# [6] 2차원 리스트  -> [[길이, 무게]],[길이, 무게],[길이, 무게],[길이, 무게]] 구성
# zip(1차원 리스트 , 1차원리스트) : 두 리스트를 요소 하나씩 반복
# 리스트내포 , [ 표현식 for 반복변수  in 반복값 if 조건식]
fish_data = [[l,w] for l , w in zip(length,weight)]
# print(fish_data)      # [[길이, 무게]],[길이, 무게],[길이, 무게],[길이, 무게]]
# print(bream_df.info)  # 도미 35마리
# print(smelt_df.info)  # 빙어 14마리


# [7] target(정답지) 만들기 , 1 : 도미를 의마히고 35개 마는다.  0: 빙어를 의마하고 14개 만든다.
fish_target = [1]*35 + [0]*14
# print(fish_target)      # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# [8] 알고리즘 모델 중 : (1) k-최근접 이웃(K-NN알고리즘) : 임의값을 넣었을 때 기존 값들 중에 가장 가까운 값 찾기 
# (1) 설치 : 사이킷런(다양한 머신모델 제공 )
#  pip install scikit-learn
# (2) K-NN 모델 호출
from sklearn.neighbors import KNeighborsClassifier

# (3) K-NN 모델 객체 생성
kn = KNeighborsClassifier()

# (4) K-NN 학습하기, kn.fit(문제,답) , 무제와 답을 같이 준다. ----> 지도 학습 : 문제와 정답을 알려주면 >
kn.fit(fish_data, fish_target)  #  fish_data : 도미와 빙어 자료 , fish_target : 도미인지 빙어인지 식별 자료 

# (5) 학습된 모델의 점수(정확도) 측정 , kn.score(문제,답) , 0~1 사이값으로 반환 , 1:100점
# 컴퓨터에게 또 다른 문제(자료) 제공 , 답(자료) 제공하여 채점하기 
print(kn.score(fish_data, fish_target))

# (6) 임의의 값 넣어서 예측 측정 , kn.predict
print(kn.predict([(30,600)])) # 임의의 물고기 길이cm와 무게600 -> 도미 ? 빙어 ? 예측한다. 

# (7) 임이의 값 시각화
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(20,200) # 임의의 값 위치
plt.show()

# (8) 근접한 이웃 찾을 기준 정하기 , 하이퍼파라미터( k값 조절) , 생략시에는 기본값 5 
#  KNeighborsClassifier(n_neighbors=참조할 이웃개수 ) # 접근한 49개 중에서 정답 찾기
# 현재 예제는 도미35 , 빙어14 -> 총 49마리 (전체)
kn = KNeighborsClassifier(n_neighbors=49) # 추후에 방대한 데이터로 활용시 최적의 k값 찾기 
kn.fit(fish_data , fish_target) # 학습
print(kn.score(fish_data, fish_target))  # 정확도 측정 , 0.7142857142857143
# 총 49마리 중에서 참조할 이웃을 49마리로 설정하면 어떠한 임의의값 예측 하더라도 무조건 도미수가 많아서 '도미' 


