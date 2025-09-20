import os
import logging
from colorlog import ColoredFormatter

from translations import translations

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BeitImageProcessor, BeitForImageClassification
import torch

from PIL import Image

# Create and initialize a named logger
def init_logger(name: str):
    LOGFORMAT = "%(log_color)s%(name)-8s| %(funcName)s | %(message)s%(reset)s"
    LOG_COLORS = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }

    logger = logging.getLogger(name)
    formatter = ColoredFormatter(LOGFORMAT, log_colors=LOG_COLORS)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

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

class NoMaskException(Exception): pass    
class TooMuchMaskException(Exception): pass

# Process request from LLM API
def process_llm():
    try:
        # Read raw text data
        raw_data = request.get_data(as_text=True).strip()
        llm_logger.debug(f"{raw_data=}")
        
        inference_result = inference_llm(raw_data)
        llm_logger.debug(f"{inference_result=}")
        return (jsonify(inference_result), 200)
    except NoMaskException:
        return ("text must contain a mask token", 418)
    except TooMuchMaskException:
        return ("text must contain one mask token", 419)
    except Exception as e:
        llm_logger.error(e)
        return (jsonify({"error": str(e)}), 500)

def inference_llm(text):
    # tokenize input
    encoded_input = llm_tokenizer(text, return_tensors='pt')
    tokenized_input = encoded_input['input_ids'][0].tolist()
    llm_logger.debug(f"{tokenized_input=}")
    
    tokens = list(map(lambda t: {
        'id': t,
        'text': llm_tokenizer.decode([t])
    }, tokenized_input))
    llm_logger.debug(f"{tokens=}")
   
    # find position of [MASK] token
    mask_token_index = (encoded_input["input_ids"] == llm_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    # Ensure only one mask token is present
    if mask_token_index.shape.numel() == 0:
        llm_logger.error("input does not contain mask token!")
        raise NoMaskException()
    elif mask_token_index.shape.numel() > 1:
        llm_logger.error("input contains too much mask tokens!")
        raise TooMuchMaskException()
            
    # predict masked tokens
    output = llm_model(**encoded_input)
    logits = output.logits
    
    # softmax on probabilities
    softmax = torch.nn.functional.softmax(logits[0, mask_token_index, :], dim=-1)

    # exclude special tokens from softmax
    filtered = softmax.squeeze(0)
    filtered[exclude_token_ids] = float('-inf')
  
    # top 5 token
    # top_tokens = torch.topk(softmax, 5, dim=1).indices[0].tolist()
    top_vals, top_ids = torch.topk(filtered,5)
    top_vals = top_vals.tolist()
    top_ids = top_ids.tolist()
    top_tok = llm_tokenizer.convert_ids_to_tokens(top_ids)
    
    pred_tokens = []
    for val, id, text in zip(top_vals,top_ids,top_tok):
        pred_tokens.append({
            'id': id,
            'text': text,
            'confidence': val * 100
        })
    llm_logger.debug(f"{pred_tokens=}")

    return {
        'tokens': tokens,
        'predictions': pred_tokens
    }

def process_sketch():
    try:
        # Open with PIL
        img = Image.open(request.stream).convert("RGB")
        sketch_logger.info(f"image size: {img.width}x{img.height}")
        
        # img = img.resize((400,300))
        img.save("last_sketch.jpg")
        
        inference_result = sketch_inference(img)
        sketch_logger.info(f"{inference_result=}")
        return jsonify(inference_result, 200)
    except Exception as e:
        sketch_logger.error(e)
        return (jsonify({"error": str(e)}), 500)

def sketch_inference(image):
    # Infer result
    inputs = sketch_processor(images=image, return_tensors="pt")
    outputs = sketch_model(**inputs)
    logits = outputs.logits

    softmax = torch.nn.functional.softmax(logits[0, :], dim=-1)
    confs, idxs = torch.topk(softmax, 4)#.tolist()

    top_5 = []
    for i in range(len(confs)):
        top_5.append({
            'label': translations[sketch_model.config.id2label[idxs[i].item()]],
            'conf': (confs[i].item() * 100.0)
        })

    return {
        'top': top_5
    }

llm_logger = init_logger("LLM")
sketch_logger = init_logger("Sketch")
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

exclude_tokens = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~']
exclude_token_ids = llm_tokenizer.convert_tokens_to_ids(exclude_tokens)

# Create Flask app
app = Flask(__name__, static_folder = 'static')
CORS(app)  # Enable CORS for all routes

# Routes
# @app.route("/")
# def home_page():
#     return send_from_directory(app.static_folder, 'home.html')

@app.route("/llm")
def llm_page():
    return send_from_directory(app.static_folder, 'llm/index.html')

@app.route("/sketch")
def sketch_page():
    return send_from_directory(app.static_folder, 'sketch/index.html')

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
