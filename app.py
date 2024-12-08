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
    y1_pred = np.argmax(y1_pred, axis=1)
    print(y1_pred)

    y2_pred = model2.predict(comment_padded)
    y2_pred = np.argmax(y2_pred, axis=1)
    print(y2_pred)

    return jsonify({"sentiment": "neutral",
                    "sentiment_conf": 0.6,
                    "emotion": "happy",
                    "emotion_conf": 0.8
                    })


if __name__ == '__main__':
    app.run()