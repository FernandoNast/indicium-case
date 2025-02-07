import pickle

import pandas as pd

from flask                   import Flask, request,Response
from model_churn.model_churn import Model_Churn

# Carregando modelo
model = pickle.load(open('C:/data_science/model/model_xgboost.pkl','rb'))

app = Flask(__name__)

@app.route('/model_churn/predict',methods=['POST'])

def fun_model_churn_predict():
    test_json = request.get_json()
    
    # Ha dado?
    if test_json:
        if isinstance(test_json,dict):
            test_raw = pd.DataFrame(test_json,index=[0]) # para uma linha
        else:
            test_raw = pd.DataFrame(test_json,columns = test_json[0].keys()) # mais linhas
        
        pipeline = Model_Churn()
        
        df = pipeline.feature_engineering(test_raw)
        
        df_response = pipeline.get_prediction(model,test_raw,df)
        
        return df_response
    else:
        return Response('{}',status=200,minetype = 'application/json')

if __name__ == '__main__':
    app.run('0.0.0.0')