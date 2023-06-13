# -*- coding: utf-8 -*-
"""
Created on Sun June 27 2022
Last modified on 19 July 2022

@author: Ms. Yap Si Qi
@course: Computer Science (Intelligent System)
@school: Asia Pacific University
@tpnumber: TP051058
@program: summary_api
@description: create api endpoint for review summarization model

"""
# import libraries
from flask import Flask, jsonify
from contractions import contractions
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import regex
from transformers import BertTokenizerFast, TFEncoderDecoderModel
import os
import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gdd

IMG_FOLDER = os.path.join('static', 'IMG')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

@app.route("/",methods=['GET'])
def index():
    return "Drop some reviews to summarize. /general, /positive, /negative to generate summary"

# URL for general summary
@app.route('/general/<string:file>')
def general_predict(file):
    gdd.download_file_from_google_drive(file_id=file ,dest_path='./reviews.csv',overwrite=True)
    path = "./reviews.csv"
    try:
        review_df = pd.read_csv(path,header=None)
        if len(review_df.columns)==1:
            review_df.columns = ['text']
            review_df['text'] = review_df['text'].apply(lambda x: clean_text(x))
            corpus = " ".join(str(t) for t in review_df['text'])  
            summary = inference(corpus,30)
            return jsonify({"summary":summary})
        else: 
            return "The CSV should only have one column"
    except:
        return "Invalid file format"
    
# URL for positive summary 
@app.route('/positive/<string:file>')
def positive_predict(file):
    gdd.download_file_from_google_drive(file_id=file ,dest_path='./reviews.csv',overwrite=True)
    path = "./reviews.csv"
    try:
        review_df = pd.read_csv(path,header=None)
        if len(review_df.columns)==1:
            review_df.columns = ['text']
            review_df['text'] = review_df['text'].apply(lambda x: clean_text(x))
            review_df['sentiment'] = review_df['text'].apply(lambda x: analyze_sentiment(x))   
            corpus = " ".join(str(t) for t in review_df['text'])  
            pos = review_df.loc[review_df['sentiment'] == 1]
            corpus = " ".join(str(t) for t in pos['text']) 
            if len(corpus)>5:
                summary = inference(corpus,10)
                return jsonify({"summary":summary})
            else:
                return 'No enough positive review to be summarized :('
           
        else: 
            return "The CSV should only have one column"
    except:
        return "Invalid file format"
            
# URL for negative summary
@app.route('/negative/<string:file>')
def negative_predict(file):
    gdd.download_file_from_google_drive(file_id=file ,dest_path='./reviews.csv',overwrite=True)
    path = "./reviews.csv"
    try:
        review_df = pd.read_csv(path,header=None)
        if len(review_df.columns)==1:
            review_df.columns = ['text']
            review_df['text'] = review_df['text'].apply(lambda x: clean_text(x))
            review_df['sentiment'] = review_df['text'].apply(lambda x: analyze_sentiment(x))   
            corpus = " ".join(str(t) for t in review_df['text'])  
            neg = review_df.loc[review_df['sentiment'] == -1]
            corpus = " ".join(str(t) for t in neg['text']) 
            if len(corpus)>5:
                summary = inference(corpus,10)
                return jsonify({"summary":summary})
            else:
                return 'No enough negative review to be summarized :('        
        else: 
            return "The CSV should only have one column"
    except:
        return "Invalid file format"
        
# clean the text by lowercasing, replace contraction, remove special characters, punctuations and digits
def clean_text(text, lowercase=True):
    if lowercase:
        text = text.lower()       
    text = ' '.join([contractions[word] if word in contractions else word for word in text.split()])    
    text = regex.sub(r'!', ' ! ', text)
    text = regex.sub(r'\?', ' ? ', text)
    text = regex.sub(r'[^a-zA-Z0-9!\?]', ' ', text)
    text = regex.sub(r'\s[0-9]+\s', ' ', text)
    text = regex.sub(r'\s+', ' ', text)  
    return text.strip()

# analyze the sentiment of text 
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    text = str(text)
    analysis = sia.polarity_scores(text.lower())
    if (analysis['compound'] > 0):
      return 1
    elif (analysis['compound'] == 0):
      return 0
    else:
      return -1

# load the trained summarization model and generate summary 
def inference(text,min_length=5):
    trained_model = TFEncoderDecoderModel.from_pretrained('./bert2bert-Checkpoint-epoch3-loss3.5220000743865967')
    test_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    test_sentence = text
    test_inputs = test_tokenizer(test_sentence, return_tensors='np', padding='max_length', truncation=True, max_length=256)
    test_outputs = trained_model.generate(**test_inputs,min_length = min_length, no_repeat_ngram_size = 1, length_penalty = 2.0)
    output_strs = test_tokenizer.batch_decode(test_outputs, skip_special_tokens=True)
    return output_strs

if __name__ == '__main__':
    app.run(port=12345, debug=True)