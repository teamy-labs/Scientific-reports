import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from processor import extract_pdf_data
from rag import build_vector_db, generate_answer

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Run Extraction + OCR
        extracted_data = extract_pdf_data(filepath)
        
        # Build RAG Index
        build_vector_db(extracted_data)
        
        return jsonify({"message": "File processed, OCR completed, and Vector DB built successfully!"})

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
        
    answer = generate_answer(question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    print("🚀 Server starting... Go to http://127.0.0.1:5000")
    app.run(debug=True, port=5000)