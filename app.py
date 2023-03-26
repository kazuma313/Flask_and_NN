from flask import Flask, render_template, url_for, request, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
import json

from cProfile import label

# from salinan_dari_nn_air import mse, data, learn, x_train, y_train
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Learning import normalize1,  normalize, denormalize, Layer, nn, mse, mape, learn



app = Flask(__name__)

@app.route('/')
@app.route('/performa', methods = ["GET", "POST"])
def performa():
    data = {'errors':
    {
        'errors': [],
        'index': [],
    },
    'perbandingan':
    {
        'aktual': [],
        'predicted': [],
        'index' : []
    }
    }
    
    with open("sample.json", "w") as outfile:
        json.dump(data, outfile) 
    
    
    req = request.method == "POST"
    
    if req:
        f = request.files['file']
        while True:
            try:
                f.save(f"data/{secure_filename(f.filename)}")
                break
            except IOError:
                f = open(f'data/{secure_filename(f.filename)}')
                f.close()
                print("File belum di close")
           
        
        # if os.path.exists(f"data/{secure_filename(f.filename)}"):
        #     os.remove(f"data/{secure_filename(f.filename)}")

        hidden1 = request.form.get('hidden1')
        hidden2 = request.form.get('hidden2')
        learningRate = request.form.get('learningRate')
        epoch = request.form.get('epoch')
        
        
        df = pd.read_csv(fr"data/{f.filename}")



        x_train = []
        y_train = []

        x_test = []
        y_test = []

        x = []
        y = []

        for i in range(len(df["bulan"])):
            x.append([
                df["x1"][i],
                df["x2"][i],
                df["x3"][i],
                df["x4"][i],
                df["x5"][i],
                df["x6"][i],
                df["x7"][i],
                df["x8"][i],
                df["x9"][i],
                df["x10"][i],
                df["x11"][i],
                df["x12"][i],
                df["x13"][i],
                df["x14"][i],
            ])
            y.append([df["target"][i]])


        x = np.array(x)
        transpose = x.T

        
        x_norm = []
        y_norm = normalize1(y)
        y_norm_1 = normalize1(y)


        for i in range(len(transpose)):
            x_norm.append(normalize(transpose[i]))

        for i in range(len(x_norm)):
            x_norm_temp_2 = []
            x_norm_temp_3 = []
            for j in range(len(x_norm[0])):
                if(j < 21):
                    x_norm_temp_2.append(x_norm[i][j])
                if(j >= 21):
                    x_norm_temp_3.append(x_norm[i][j])
            x_train.append(x_norm_temp_2)
            x_test.append(x_norm_temp_3)

        for i in range(len(y_norm)):
            if(i < 21):
                y_train.append(y_norm[i])
            if(i >= 21):
                y_test.append(y_norm[i])

        x_train = np.array(x_train)
        x_test = np.array(x_test)

        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        
        errors = learn(x_train=x_train, y_train=y_train, hidden1=int(hidden1),
                hidden2= int(hidden2), learningRate=float(learningRate), epoch=int(epoch))
        
        arr_1 = y_norm_1
        predict_1 = nn.predict(x_test.T)
        for i in range(len(predict_1)):
            arr_1.append(predict_1[i].tolist())

        denormalize_1 = denormalize(arr_1, y)

        predicted_1 = []
        actual_1 = []
        predicted_2 = []
        actual_2 = []

        for i in range(len(denormalize_1)):
            if(i >= 21 and i <= 28):
                actual_1.append(denormalize_1[i])
                actual_2.append(arr_1[i])
            if(i >= len(y_norm)):
                predicted_1.append(denormalize_1[i])
                predicted_2.append(arr_1[i])
        
        
        
        data = {'errors':
        {
            'errors': errors,
            'index': np.arange(0, np.shape(errors)[0]).tolist(),
        },
        'perbandingan':
        {
            'aktual': np.array(actual_2).ravel().tolist(),
            'predicted': np.array(predicted_2).ravel().tolist(),
            'index' : np.arange(0, np.shape(np.array(predicted_2).ravel())[0]).tolist()
        }
        }
        
        with open("sample.json", "w") as outfile:
            json.dump(data, outfile)
        
        nilai_mse = mse(actual_2, predicted_2)
        nilai_mape = mape(actual_2, predicted_2)
        banding = np.array([np.array(actual_2).flatten(), np.array(predicted_2).flatten()]).T
        banding = pd.DataFrame(data=banding, columns=['Aktual', 'Prediksi'])
        
        print(banding)
        return render_template('Index.html', Mse=nilai_mse, Mape=nilai_mape, pesan="Success", request = req, tables=[banding.to_html(index=False, classes="table table-bordered", table_id="dataTable", justify="center")],label=label)

    else:
    
        return render_template('Index.html', pesan="Tidak ada data yang disimpan", request = req)
    
    
@app.route('/API/data/')
def api_data():
    with open("sample.json") as outfile:
        content = outfile.read()
        
    # print(content)
    return (content)



if __name__ == '__main__':
    app.secret_key='ItIsSecret'
    app._static_folder = os.path.abspath("static/")    
    app.debug = True
    app.env="development"
    app.run()