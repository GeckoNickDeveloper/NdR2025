from flask import Flask, request, jsonify
from flask_cors import CORS

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

#============================== MODEL LOADING
model_id = "dbmdz/bert-base-italian-xxl-cased"
path = "./bert-base-italian-xxl-cased"

def download(id, path):
  # Download
  tokenizer = AutoTokenizer.from_pretrained(id)
  model = AutoModelForMaskedLM.from_pretrained(id, trust_remote_code=True)

  # Save local copy
  tokenizer.save_pretrained(path)
  model.save_pretrained(path)

if not os.path.exists(path):
  download(model_id, path)


tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForMaskedLM.from_pretrained(path)
#==============================


def infer(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)


    #stampo il testo nei token corrispondenti (1,4,2 sono token speciali 4-> mask)
    print(encoded_input['input_ids'])

    logits = output.logits

    # individua la posizione del token <mask>
    mask_token_index = (encoded_input["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    # softmax sulle probabilità
    softmax = torch.nn.functional.softmax(logits[0, mask_token_index, :], dim=-1)

    # top 5 token
    top_tokens = torch.topk(softmax, 5, dim=1).indices[0].tolist()


    ### Output
    tokens = []
    for i in encoded_input:
        tokens.append(tokenizer.decode([i]))

    outtokens = []
    conf = []

    for token in top_tokens:
        conf.append(softmax[0, token].item() * 100)
        outtokens.append(tokenizer.decode([token]))

    return {
        'tokens': tokens,
        'out-tokens': outtokens,
        'conf': conf,
    }



@app.route("/predict", methods=["GET"])
def predict():
    print('RES')
    try:
        # Read raw text data
        raw_data = request.get_data(as_text=True).strip()

        # Run inference
        res = infer(text)

        print(res)

        return (res, 200)

    except Exception as e:
        return (jsonify({"error": str(e)}), 400)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
