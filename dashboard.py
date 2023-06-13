# -*- coding: utf-8 -*-
"""
Created on Sun June 27 2022
Last modified on 19 July 2022

@author: Ms. Yap Si Qi
@course: Computer Science (Intelligent System)
@school: Asia Pacific University
@tpnumber: TP051058
@program: dashboard
@description: create web application for review summarization model

"""
# import libraries
from flask import Flask, render_template, request
from contractions import contractions
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import regex
from wordcloud import WordCloud
from transformers import BertTokenizerFast, TFEncoderDecoderModel
from matplotlib import pyplot as plt
import pandas as pd
import os

IMG_FOLDER = os.path.join('static', 'IMG')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

@app.route("/",methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
     try: 
        file = request.files['CSVfile']
        path = "./reviews.csv"
        file.save(path)
        review_df = pd.read_csv(path,header=None)
        if len(review_df.columns)==1:
            if request.method == 'POST':
                if request.form.get('Get Negative Summary') == 'Get Negative Summary':
                    button = 'negative'
                elif request.form.get('Get Positive Summary') == 'Get Positive Summary':
                    button = 'positive'
                else:
                    button = 'general'
            review_df.columns = ['text']
            review_df['text'] = review_df['text'].apply(lambda x: clean_text(x))
            wordcloud_path = wordcloud(review_df,'text')
            review_df['sentiment'] = review_df['text'].apply(lambda x: analyze_sentiment(x))
            sentiment_path = sentiment_pie_chart(review_df)       
            if button == 'general':
                corpus = " ".join(str(t) for t in review_df['text']) 
                summary = inference(corpus,30)
            elif button == 'positive':
                pos = review_df.loc[review_df['sentiment'] == 1]
                corpus = " ".join(str(t) for t in pos['text']) 
                if len(corpus)>5:
                    summary = inference(corpus,10)
                else:
                    summary = 'No enough positive review to be summarized :('
            elif button == 'negative':
                pos = review_df.loc[review_df['sentiment'] == -1]
                corpus = " ".join(str(t) for t in pos['text']) 
                if len(corpus)>5:
                    summary = inference(corpus,10)
                else:
                    summary = 'No enough negative review to be summarized :('
            return render_template('index.html',summary=summary,sentiment=sentiment_path,wordcloud=wordcloud_path,topic=button.capitalize())
        else: 
            return render_template('index.html',error="The CSV should only have one column")
     except:
         return render_template('index.html',error="No CSV file is uploaded")

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

# generate pie chart based on sentiment of text
def sentiment_pie_chart(df):
    values = df['sentiment'].value_counts().keys().tolist()
    labels =[]
    for i in values:
        if i == 1:
            labels.append('positive')
        elif i == 0:
            labels.append('neutral')
        elif i == -1:
            labels.append('negative')
    counts = df['sentiment'].value_counts().tolist()
    plt.pie(counts, labels = labels, textprops={'fontsize': 18})
    sentiment_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sentiment.png')
    plt.savefig(sentiment_path)
    return sentiment_path

# generate wordcloud for text
def wordcloud(df, text):
    stop_words = set(stopwords.words('english'))
    corpus = " ".join(str(t) for t in df[text])    
    wordcloud = WordCloud(stopwords=stop_words,
                          max_font_size=50, 
                          max_words=100, 
                          collocations=False,
                          background_color="white").generate(corpus)  
    plt.figure(figsize=(10,10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    textblob_path = os.path.join(app.config['UPLOAD_FOLDER'], 'wordcloud.png')
    plt.savefig(textblob_path)
    return textblob_path

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


