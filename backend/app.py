import http
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords
import json
from typing import OrderedDict
import requests
from re import T
from flask_cors import CORS, cross_origin

nltk.download('stopwords')

set(stopwords.words('english'))

app = Flask(__name__)


@app.route('/sentiment', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')
    
    #convert to lowercase
    text1 = request.form['text1'].lower()
    
    text_final = ''.join(c for c in text1 if not c.isdigit())
    
    #remove punctuations
    # text3 = ''.join(c for c in text2 if c not in punctuation)
        
    #remove stopwords    
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + dd['compound'])/2, 2)
    output = OrderedDict({
        "final":compound,
        "text1": text_final,
        "text2": dd['pos'],
        "text5": dd['neg'],
        "text4": compound,
        "text3": dd['neu']
    })
    return output


if __name__ == "__main__":
    app.run(debug=True)
