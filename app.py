from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud
import os
from preprocess import clean_text

app = Flask(__name__)

# Load Model dan Tokenizer
model = None
tokenizer = None

def load_resources():
    global model, tokenizer
    if model is None:
        model = load_model('model/model_lstm.h5')
    if tokenizer is None:
        with open('model/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)

MAX_LEN = 100

def predict_sentiment(text):
    load_resources()
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(pad, batch_size=1)[0][0]
    return "positif" if pred >= 0.5 else "negatif"

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment_result = None
    komentar_baru = ''

    # Membaca data CSV
    df = pd.read_csv(os.path.join(os.getcwd(), 'data/data_youtube_ijazah_jokowi.csv')).head(50)
    df['cleaned'] = df['comment'].astype(str).apply(clean_text)

    # Prediksi sentimen untuk semua komentar
    sentiments = []
    for text in df['cleaned']:
        try:
            pred = predict_sentiment(text)
            sentiments.append(pred)
        except Exception as e:
            sentiments.append('error')
            print(f"Error: {e}")
    df['sentimen'] = sentiments

    # Membuat WordCloud
    all_text = ' '.join(df['cleaned'])
    font_path = os.path.join('static', 'fonts')
    wordcloud = WordCloud(
    font_path=font_path,
    width=600,
    height=300,
    background_color='white'
).generate(all_text)
    wordcloud.to_file('static/wordcloud.png')

    # Jika form dikirim
    if request.method == 'POST':
        komentar_baru = request.form['comment']
        sentiment_result = predict_sentiment(komentar_baru)

    return render_template('index.html',
                           data=df[['platform', 'comment', 'timestamp', 'username', 'sentimen']],
                           hasil=sentiment_result,
                           komentar=komentar_baru)

# Membuat folder 'static' jika belum ada
if not os.path.exists('static'):
    os.makedirs('static')

    if not os.path.exists('static/fonts'):
      os.makedirs('static/fonts')


@app.route('/download')
def download():
    df = pd.read_csv('data/data_youtube_ijazah_jokowi.csv')
    df['cleaned'] = df['comment'].astype(str).apply(clean_text)
    df['sentimen'] = df['cleaned'].apply(predict_sentiment)
    df.to_csv('hasil_analisis.csv', index=False)
    return "<h3>File berhasil disimpan sebagai hasil_analisis.csv</h3>"

if __name__ == '__main__':
    app.run(debug=True)
