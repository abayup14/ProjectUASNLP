from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_sentiment', methods=['POST'])
def get_sentiment():
    with open("tokenizer.json", "r") as f:
        tokenizer_json = f.read()

    tokenizer = tokenizer_from_json(tokenizer_json)
    comment = request.json.get("comment")

    comment_sequences = tokenizer.texts_to_sequences(comment)
    comment_padded = pad_sequences(comment_sequences, maxlen = 300, padding = 'post')

    model1 = load_model("best_model1.keras")
    model2 = load_model("best_model2.keras")

    y1_pred = model1.predict(comment_padded)
    y1_pred_class = np.argmax(y1_pred, axis=1)[0]
    conf1 = np.max(y1_pred)
    sent_class = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
    # print(y1_pred_class)
    sentiment = sent_class[y1_pred_class]

    y2_pred = model2.predict(comment_padded)
    y2_pred_class = np.argmax(y2_pred, axis=1)[0]
    conf2 = np.max(y2_pred)
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
    # print(y2_pred)
    emotion = emot_class[y2_pred_class]

    return jsonify({"sentiment": sentiment,
                    "sentiment_conf": float(conf1),
                    "emotion": emotion,
                    "emotion_conf": float(conf2)
                    })


if __name__ == '__main__':
    app.run()