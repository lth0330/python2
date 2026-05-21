from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


class Service :
    def __init__(self) :
        self.model = None

        self.ss = None
        self.poly = None
    def Purchasing_learn( self , clothesList ) :
        # 1) df 만들기 
        df = pd.DataFrame( clothesList )
        train_data = df[['age','gender','inflow','style']]
        target_data =df['category'].values
        train_input , test_input , train_target , test_target = train_test_split( train_data , target_data , test_size=0.1, random_state=45)

        ss = StandardScaler()
        ss.fit(train_input)
        train_scaled = ss.transform(train_input)
        test_scaled = ss.transform(test_input)

        poly = PolynomialFeatures()
        poly.fit(train_scaled)
        train_poly = poly.transform(train_scaled)
        test_poly = poly.transform(test_scaled) 


        sc = SGDClassifier(loss='log_loss', random_state=42 , max_iter=100, tol=None)
        sc.fit( train_poly , train_target )

        print( sc.score( test_poly , test_target ) ) 

        self.ss = ss
        self.model = sc 
        self.poly = poly
        print(self.model)
        return True


    def Purchasing_predict( self , clothes ) :
        print(clothes)
        print(self.model)
        if self.model is None : 
            return "학습 모델이 없습니다."
        
        del clothes['id'] 
        clothes = [ value for value in clothes.values() ]
        
        scaled= self.ss.transform([clothes])
        scaled = self.poly.transform(scaled)
        print( scaled )
        
        predict = self.model.predict( scaled  )
        print(predict)
        return int(predict[0])
    

service = Service()