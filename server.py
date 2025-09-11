from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BeitImageProcessor, BeitForImageClassification

from PIL import Image

import torch
import os
import io



#================================= App creation
app = Flask(__name__, static_folder = 'static')
CORS(app)  # Enable CORS for all routes




#================================= Init
#========= LLM
llm_model_id = "dbmdz/bert-base-italian-xxl-cased"
llm_path = "./models/llm"

def download_llm(id, path):
  # Download
  tokenizer = AutoTokenizer.from_pretrained(id)
  model = AutoModelForMaskedLM.from_pretrained(id, trust_remote_code=True)

  # Save local copy
  tokenizer.save_pretrained(path)
  model.save_pretrained(path)

if not os.path.exists(llm_path):
  download_llm(llm_model_id, llm_path)

llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
llm_model = AutoModelForMaskedLM.from_pretrained(llm_path)



#========= Sketch
sketch_model_id = 'kmewhort/beit-sketch-classifier'
#path = "./beit-base-patch16-224-pt22k-ft22k"
sketch_path = './models/sketch'

def download_sketch(id, path):
  # Download
  processor = BeitImageProcessor.from_pretrained(id)
  model = BeitForImageClassification.from_pretrained(id, trust_remote_code=True)

  # Save local copy
  processor.save_pretrained(path)
  model.save_pretrained(path)

if not os.path.exists(sketch_path):
  download_sketch(sketch_model_id, sketch_path)


sketch_processor = BeitImageProcessor.from_pretrained(sketch_path)
sketch_model = BeitForImageClassification.from_pretrained(sketch_path)












#================================= Behaviors
def llm_inference(text):
    encoded_input = llm_tokenizer(text, return_tensors='pt')
    output = llm_model(**encoded_input)


    #stampo il testo nei token corrispondenti (1,4,2 sono token speciali 4-> mask)
    print(encoded_input['input_ids'])

    logits = output.logits

    # individua la posizione del token <mask>
    mask_token_index = (encoded_input["input_ids"] == llm_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    # softmax sulle probabilità
    softmax = torch.nn.functional.softmax(logits[0, mask_token_index, :], dim=-1)

    # top 5 token
    top_tokens = torch.topk(softmax, 5, dim=1).indices[0].tolist()


    ### Output
    tokens = []
    for i in encoded_input:
        tokens.append(llm_tokenizer.decode([i]))

    outtokens = []
    conf = []

    for token in top_tokens:
        conf.append(softmax[0, token].item() * 100)
        outtokens.append(llm_tokenizer.decode([token]))

    return {
        'tokens': tokens,
        'out-tokens': outtokens,
        'conf': conf,
    }



def sketch_inference(image):
    return {'todo': 'implement'}
    pass




#================================= Static Content
# @app.route("/")
# def home():
#     return send_from_directory(app.static_folder, 'home.html')

@app.route("/llm")
def llm_home():
    return send_from_directory(app.static_folder, 'llm.html')

@app.route("/sketch")
def sketch_home():
    return send_from_directory(app.static_folder, 'sketch.html')

@app.route("/<path:filename>")
def static_content(filename):
    return send_from_directory(app.static_folder, filename)



#================================= APIs
### LLM
@app.route("/api/llm", methods=["POST"])
def llm_api():
    print('ok?')
    try:
        # Read raw text data
        raw_data = request.get_data(as_text=True).strip()
        
        inference_result = llm_inference(raw_data)
        
        print(inference_result)

        return (jsonify(inference_result), 200)
    
    except Exception as e:
        print('E')
        return (f'{str(e)}', 400)


### Sketch
@app.route("/api/sketch", methods=["POST"])
def sketch_api():
    try:
        # Open with PIL
        img = Image.open(request.stream).convert("RGB")

        inference_result = sketch_inference(img)

        # Now you can preprocess for your model
        return jsonify(inference_result, 200)
    
    except Exception as e:
        return (jsonify({"error": str(e)}), 500)






if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
