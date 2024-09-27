from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Load pre-trained NER model
ner_model = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['message']
    entities = ner_model(text)
    return jsonify(entities)

if __name__ == '__main__':
    app.run(debug=True)
