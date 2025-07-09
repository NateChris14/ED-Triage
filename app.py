from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')

# === Load models and vectorizers ===
model = pickle.load(open('ed_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
with open('tf-idf-chief.pkl', 'rb') as f:
    tfidf_chief = pickle.load(f)
svd = pickle.load(open('truncated-svd.pkl', 'rb'))

# === Flask app setup ===
app = Flask(__name__)

# === Abbreviation dictionary ===
abbr_dict = {
    "f/c": "fever chills", "ruq": "right upper quadrant", "n/v": "nausea vomiting",
    "cp": "chest pain", "nos": "not otherwise specified", "loc": "loss of consciousness",
    "lbp": "lower back pain", "ant": "anterior", "avf": "arteriovenous fistula",
    "peg": "percutaneous endoscopic gastrostomy", "abd": "abdomen", "cpr": "cardiopulmonary resuscitation",
    "llq": "left lower quadrant", "fb": "foreign body", "ptbd": "percutaneous transhepatic biliary drainage",
    "a/": "associated with", "rlq": "right lower quadrant", "hb": "hemoglobin",
    "dz": "dizziness", "sx": "symptoms", "bsl": "blood sugar level",
    "pprom": "preterm premature rupture of membranes", "ptx": "pneumothorax",
    "c/s": "cesarean section", "g/w": "general weakness"
}

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# === Text cleaning and preprocessing ===
def expand_abbreviations(text):
    text = str(text).lower()
    for abbr, full_form in abbr_dict.items():
        text = text.replace(abbr, full_form)
    return text

def clean_text(text):
    text = expand_abbreviations(text)
    text = re.sub(r'[^a-z\s,-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    text = clean_text(text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# === List of important structured features ===
sig_feat = [
    'Group', 'Sex', 'Age', 'Injury', 'Pain', 'NRS_pain', 'SBP',
    'RR', 'BT', 'Saturation', 'KTAS_RN'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # === 1. Extracting structured features ===
        input_dict = {col: [float(data[col])] for col in sig_feat}
        input_df = pd.DataFrame(input_dict)

        # === 2. Extracting and processing text fields ===
        chief_raw = data['Chief_complain']

        chief_clean = preprocess_text(chief_raw)

        # === 3. TF-IDF transformation ===
        chief_vec = tfidf_chief.transform([chief_clean])


        # === 4. Combining TF-IDF vectors and applying SVD ===
        text_combined = chief_vec.toarray()
        text_reduced = svd.transform(text_combined)
        text_df = pd.DataFrame(text_reduced, columns=[f'svd_{i+1}' for i in range(50)])


        # === 5. Scaling structured data ===
        input_scaled = scaler.transform(input_df[sig_feat])
        input_scaled_df = pd.DataFrame(input_scaled, columns=sig_feat)

        # === 6. Combining scaled structured + reduced text ===
        final_input = pd.concat([input_scaled_df.reset_index(drop=True), text_df], axis=1)

        # Convert to a proper NumPy array
        final_input_np = final_input.to_numpy(dtype=np.float32)  

        ## === 7. Predicting ===
        prediction = model.predict(final_input_np)


        return jsonify({'KTAS_expert_prediction': int(prediction[0])})

    except Exception as e:
        print("ERROR during prediction:", e)  
        return jsonify({
        'error': str(e)
    })


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
