from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk

app = Flask(__name__)
app.config['DEBUG'] = True

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

def remove_tags(text):
    text = re.sub('@[a-zA-Z0-9_]*', '', text)
    text = re.sub('#[a-zA-Z0-9_]*', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def preprocess(text):
    text = remove_tags(text)
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_sentiment', methods=['POST'])
def get_sentiment():
    with open("tokenizer.json", "r") as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

    comment = request.json.get("comment")
    print(comment)

    comment_preprocess = preprocess(comment)
    print(comment_preprocess)

    comment_sequences = tokenizer.texts_to_sequences([comment_preprocess])
    print(comment_sequences)

    comment_padded = pad_sequences(comment_sequences, maxlen = 300, padding = 'post')
    print(comment_padded)

    model1 = load_model("best_model1.keras")
    model2 = load_model("best_model2.keras")

    y1_pred = model1.predict(comment_padded)
    print(y1_pred)
    y1_pred_class = np.argmax(y1_pred, axis=1)[0]
    print(y1_pred_class)
    conf_sent = np.max(y1_pred)
    sent_class = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
    sentiment = sent_class[y1_pred_class]

    y2_pred = model2.predict(comment_padded)
    print(y2_pred)
    y2_pred_class = np.argmax(y2_pred, axis=1)[0]
    print(y2_pred_class)
    conf_emot = np.max(y2_pred)
    emot_class = {
        0: "anger",
        1: "anticipation",
        2: "optimism",
        3: "disgust",
        4: "joy",
        5: "fear",
        6: "sadness",
        7: "surprise"
    }
    emotion = emot_class[y2_pred_class]

    return jsonify({"cleaned_comment": comment_preprocess,
                    "sentiment": sentiment,
                    "sentiment_conf": float(conf_sent),
                    "emotion": emotion,
                    "emotion_conf": float(conf_emot)
                    })


if __name__ == '__main__':
    app.run()