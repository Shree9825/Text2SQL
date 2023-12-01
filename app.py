from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from transformers import AutoTokenizer

model_name="mrm8488/t5-base-finetuned-wikiSQL"

app = Flask(__name__)

model=pickle.load(open('Text_To_SQL_Prediction_New','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    feature = request.form['textquery']
    tokenizer =AutoTokenizer.from_pretrained(model_name)
    input_text = "translate English to SQL: %s </s>" % feature
    
    features = tokenizer([input_text], return_tensors='pt')

    output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'])
    query=tokenizer.decode(output[0])[6:]
    q1=query[:-4]
    q1=q1.replace("table","Superstore")
    return render_template('index.html',pred=q1)

if __name__ == '__main__':
    app.run(debug=True)