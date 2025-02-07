import pickle
import pandas as pd

# Criando CLASS
class Model_Churn(object):
    def __init__(self):
        self.home_path = '/caminho'
        state = 1

    def feature_engineering(self,df1):
        # Criando Dummies para Paises (Geography - apenas 3: Franca, Alemanha e Espanha)
        df1_encoded = pd.get_dummies(df1,columns=['Geography','Gender'])

        df2 = df1_encoded[['CustomerId', 'CreditScore', 'Age', 'Tenure', 'Balance',
       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
       'Geography_France', 'Geography_Germany', 'Geography_Spain',
       'Gender_Female', 'Gender_Male']]
        
        # Retirando SURNAME (Sobrenome) e RowNumber, pois nao agregam
        # df2 = df1_encoded.drop(['Surname','RowNumber'],axis=1)

        return df2
    
    def get_prediction(self,model,original_data,test_data):
        pred = model.predict(test_data)
        original_data['predictedValues'] = pred

        return original_data.to_json(orient = 'records', date_format = 'iso')
