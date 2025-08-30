from flask import Flask, request, render_template, jsonify
import joblib
import PyPDF2
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# --- Load the pre-trained model and vectorizer ---
model = joblib.load('../saved_models/spam_classifier_model.joblib')
vectorizer = joblib.load('../saved_models/tfidf_vectorizer.joblib')

# --- Text Cleaning Function ---
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# --- Define the main route for the web application ---
@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    chart_data = None
    if request.method == 'POST':
        files = request.files.getlist('pdf_files[]') # Get list of files
        
        if not files or files[0].filename == '':
            return render_template('index.html', error="No files selected.")
        
        spam_count = 0
        ham_count = 0

        for file in files:
            if file and file.filename.endswith('.pdf'):
                try:
                    pdf_reader = PyPDF2.PdfReader(file.stream)
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text() or ""
                    
                    if not pdf_text.strip():
                        results.append({'filename': file.filename, 'error': 'Could not extract text.'})
                        continue

                    # 1. Clean and vectorize the text
                    cleaned_text = clean_text(pdf_text)
                    text_vector = vectorizer.transform([cleaned_text])
                    
                    # 2. Get prediction and confidence probabilities
                    prediction_proba = model.predict_proba(text_vector)
                    prediction = model.predict(text_vector)[0]
                    
                    # 3. Determine confidence score
                    confidence = np.max(prediction_proba) * 100
                    
                    # 4. Count results for chart
                    if prediction == 'spam':
                        spam_count += 1
                    else:
                        ham_count += 1
                    
                    results.append({
                        'filename': file.filename,
                        'prediction': prediction,
                        'confidence': f"{confidence:.2f}%"
                    })
                    
                except Exception as e:
                    results.append({'filename': file.filename, 'error': f'Processing error: {e}'})
            else:
                results.append({'filename': file.filename, 'error': 'Invalid file type.'})

        # Prepare data for the chart if more than one file was processed
        if spam_count + ham_count > 0:
            chart_data = {'spam': spam_count, 'ham': ham_count}

    return render_template('index.html', results=results, chart_data=chart_data)

# --- Run the Flask app ---
if __name__ == '__main__':
    app.run(debug=True)