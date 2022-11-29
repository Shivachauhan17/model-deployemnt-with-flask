import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)#sets the name variable to module name
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')#generate output from a template file based on the Jinja2 engine that is found in the application's templates folder.

@app.route('/predict',methods=['POST'])
def predict():
    features=request.form['rate']
    new_features=np.array(int(features))
    
    prediction=model.predict([[new_features]])
    output=round(prediction[0],2)
    return render_template('index.html',prediction_text='salary should be $ {}'.format(output))



if __name__ == '__main__':
    app.run(port=5000,debug=True)
