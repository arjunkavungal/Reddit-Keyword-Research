from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
import praw
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from model import *
import pickle
nltk.download('punkt')
nltk.download('stopwords')
plt.switch_backend('Agg') 
name = ""
@app.route("/")
def home():
    return render_template('index.html')
@app.route('/process', methods=['POST'])
def process():


    name = request.form['name'] # get the value in the text box

    if name:
        df = get_hot_titles(name)
        df = get_title_length(df)
        df = get_word_count(df)
        df = unweighted_word_count(df)
        for i in df.columns[4:]:
            df[i] = df[i] * df['ups']
        df = df.append(df.sum().rename('Total'))      

        return jsonify({'name' : " ".join(df.iloc[-1][4:-1].astype(int).nlargest(25).keys().values.tolist()), 'val':' '.join(str(x) for x in df.iloc[-1][4:-1].astype(int).nlargest(25).values.tolist())})
