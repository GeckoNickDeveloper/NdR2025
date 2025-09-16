import os

from translations import translations

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BeitImageProcessor, BeitForImageClassification
import torch

from PIL import Image

# Download the LLM model from Hugging Face
def download_llm(id, path):
    tokenizer = AutoTokenizer.from_pretrained(id)
    model = AutoModelForMaskedLM.from_pretrained(id, trust_remote_code=True)

    tokenizer.save_pretrained(path)
    model.save_pretrained(path)

# Download the Sketch model from Hugging Face
def download_sketch(id, path):
    processor = BeitImageProcessor.from_pretrained(id)
    model = BeitForImageClassification.from_pretrained(id, trust_remote_code=True)

    processor.save_pretrained(path)
    model.save_pretrained(path)

# Process request from LLM API
def process_llm():
    try:
        # Read raw text data
        raw_data = request.get_data(as_text=True).strip()
        print(f"LLM: {raw_data=}")
        
        if "[MASK]" not in raw_data:
            return ("text must contain a mask token", 418)
        
        inference_result = inference_llm(raw_data)
        print(f"LLM: {inference_result}")
        return (jsonify(inference_result), 200)
    except Exception as e:
        print(f"LLM: {e}")
        return (jsonify({"error": str(e)}), 500)

def inference_llm(text):
    encoded_input = llm_tokenizer(text, return_tensors='pt')
    output = llm_model(**encoded_input)

    input = encoded_input['input_ids'][0].tolist()
    print(f"LLM: {input=}")

    logits = output.logits
    # individua la posizione del token <mask>
    mask_token_index = (encoded_input["input_ids"] == llm_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    
    # softmax sulle probabilità
    softmax = torch.nn.functional.softmax(logits[0, mask_token_index, :], dim=-1)
    # top 5 token
    top_tokens = torch.topk(softmax, 5, dim=1).indices[0].tolist()

    tokens = []
    for i in input:
        tokens.append({
            'id': i, 
            'text': llm_tokenizer.decode([i])
        })

    pred_tokens = []
    for token in top_tokens:
        pred_tokens.append({
            'id': token,
            'text': llm_tokenizer.decode([token]),
            'confidence': softmax[0, token].item() * 100
        })

    return {
        'tokens': tokens,
        'predictions': pred_tokens
    }

def process_sketch():
    try:
        # Open with PIL
        img = Image.open(request.stream).convert("RGB")
        print(f"Sketch: {img=}")
        inference_result = sketch_inference(img)
        return jsonify(inference_result, 200)
    except Exception as e:
        return (jsonify({"error": str(e)}), 500)

def sketch_inference(image):
    # Infer result
    inputs = sketch_processor(images=image, return_tensors="pt")
    outputs = sketch_model(**inputs)
    logits = outputs.logits

    # Get most confident class
    # best_guess = { 'label': None, 'conf': 0.0 }

    predicted_class_idx = logits.argmax(-1).item()
    predicted_class_label = sketch_model.config.id2label[predicted_class_idx]

    softmax = torch.nn.functional.softmax(logits[0, :], dim=-1)
    confs, idxs = torch.topk(softmax, 5)#.tolist()

    top_5 = []
    for i in range(len(confs)):
        top_5.append({
            'label': translations[sketch_model.config.id2label[idxs[i].item()]],
            'conf': (confs[i].item() * 100.0)
        })

    return {
        'top': top_5
    }

llm_model_id = "dbmdz/bert-base-italian-xxl-cased"
llm_path = "./models/llm"
sketch_model_id = 'kmewhort/beit-sketch-classifier'
#path = "./beit-base-patch16-224-pt22k-ft22k"
sketch_path = './models/sketch'

# Ensure modes are present
if not os.path.exists(llm_path):
    download_llm(llm_model_id, llm_path)
if not os.path.exists(sketch_path):
    download_sketch(sketch_model_id, sketch_path)
# Load the models
llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
llm_model = AutoModelForMaskedLM.from_pretrained(llm_path)
sketch_processor = BeitImageProcessor.from_pretrained(sketch_path)
sketch_model = BeitForImageClassification.from_pretrained(sketch_path)

# Create Flask app
app = Flask(__name__, static_folder = 'static')
CORS(app)  # Enable CORS for all routes

# Routes
# @app.route("/")
# def home_page():
#     return send_from_directory(app.static_folder, 'home.html')

@app.route("/llm")
def llm_page():
    return send_from_directory(app.static_folder, 'llm.html')

@app.route("/sketch")
def sketch_page():
    return send_from_directory(app.static_folder, 'sketch.html')

@app.route("/<path:filename>")
def static_content(filename):
    return send_from_directory(app.static_folder, filename)

@app.route("/api/llm", methods=["POST"])
def llm_api():
    return process_llm()
    
@app.route("/api/sketch", methods=["POST"])
def sketch_api():
   return process_sketch()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4200, debug=True)
