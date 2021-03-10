from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)  # initializing a flask application


@app.route('/',methods=['GET'])  # routes to the home page
@cross_origin()   # so that app is compatible even on the cloud server
def homepage():
    return render_template('index.html')


def getinfo():
    try:
        timeformat = str(request.form['timeformat'])
        White_Rating = int(request.form['White_Rating'])
        Black_Rating = float(request.form['Black_Rating'])
        opening = str(request.form['opening'])

        rf = pickle.load(open('RF_model.pickle','rb')) # we load the RF_model
        count_enc = pickle.load(open('count_enc.pickle','rb')) #we load the Count Encoder model

        val=[]
        enc = ['increment_code', 'opening_name']
        df=pd.DataFrame(np.array([[timeformat,opening]]), columns=['increment_code', 'opening_name'])
        encoded=count_enc.transform(df[enc]).to_numpy()
        encoded_array=encoded.flatten()  #converting 2d array to 1d (for merging purpose)
        transformed=np.concatenate([np.array([White_Rating,Black_Rating]),encoded_array])
        pred=rf.predict([transformed])
        print('Prediction is',pred)
        val.append(pred)
        prob=rf.predict_proba([transformed])
        val.append(prob)
        return val  # returns a 1d array containing 1 element
    except Exception as e:
        print('Exception occurred',e)


@app.route('/predict',methods=['POST','GET'])
@cross_origin()
def index():
    if request.method == 'POST':
        values = getinfo()
        prediction=values[0]
        prob=values[1]
        print (prediction)
        black = round(prob[0][0] * 100,2)
        white = round(prob[0][2] * 100,2)
        # 1st class is black ,2nd is draw ,3rd is white
        if prediction == 1:
            return render_template('index_white.html',black=black,white=white)
        elif prediction == 0:
            return render_template('index_draw.html',black=black,white=white)
        elif prediction == -1:
            return render_template('index_black.html',black=black,white=white)
    else:
        return render_template('index.html')


if __name__=='__main__':
    #app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True)
