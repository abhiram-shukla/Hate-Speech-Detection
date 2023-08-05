from flask import Flask, request
from flask import render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def homepage():
    return render_template('index.html')

@app.route("/getAudio", methods = ["GET", "POST"])
def getAudioInput():
    text = request.form.get("text")
    text = [text]
    loaded_model_svm = pickle.load(open('finalized_model_SVM.sav', 'rb'))
    dp = pd.read_csv('processed_data_vol2.csv', encoding='cp1252')
    Tfidf_vect = TfidfVectorizer()
    Tfidf_vect.fit(dp['text_final'])
    text_Tfidf = Tfidf_vect.transform(text)
    output = loaded_model_svm.predict(text_Tfidf)
    if output == 0:
        result = "Hateful"
    else:
        result = "Normal"
    return render_template("index.html", output = result)

# @app.route('/developer', methods = ["GET", "POST"])
# def developer():
#     return render_template('dev.html')

if __name__ == "__main__":
    app.run(debug = True)