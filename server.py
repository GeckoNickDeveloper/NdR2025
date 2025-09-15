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


# Sketch labels
translations = {
    "aircraft carrier": "portaerei",
    "airplane": "aereo",
    "alarm clock": "sveglia",
    "ambulance": "ambulanza",
    "angel": "angelo",
    "animal migration": "migrazione animale",
    "ant": "formica",
    "anvil": "incudine",
    "apple": "mela",
    "arm": "braccio",
    "asparagus": "asparago",
    "axe": "ascia",
    "backpack": "zaino",
    "banana": "banana",
    "bandage": "cerotto",
    "barn": "fienile",
    "baseball": "baseball",
    "baseball bat": "mazza da baseball",
    "basket": "cesto",
    "basketball": "pallacanestro",
    "bat": "pipistrello",
    "bathtub": "vasca da bagno",
    "beach": "spiaggia",
    "bear": "orso",
    "beard": "barba",
    "bed": "letto",
    "bee": "ape",
    "belt": "cintura",
    "bench": "panchina",
    "bicycle": "bicicletta",
    "binoculars": "binocolo",
    "bird": "uccello",
    "birthday cake": "torta di compleanno",
    "blackberry": "mora",
    "blueberry": "mirtillo",
    "book": "libro",
    "boomerang": "boomerang",
    "bottlecap": "tappo di bottiglia",
    "bowtie": "cravattino",
    "bracelet": "braccialetto",
    "brain": "cervello",
    "bread": "pane",
    "bridge": "ponte",
    "broccoli": "broccoli",
    "broom": "scopa",
    "bucket": "secchio",
    "bulldozer": "bulldozer",
    "bus": "autobus",
    "bush": "cespuglio",
    "butterfly": "farfalla",
    "cactus": "cactus",
    "cake": "torta",
    "calculator": "calcolatrice",
    "calendar": "calendario",
    "camel": "cammello",
    "camera": "fotocamera",
    "camouflage": "mimetismo",
    "campfire": "falò",
    "candle": "candela",
    "cannon": "cannone",
    "canoe": "canoa",
    "car": "automobile",
    "carrot": "carota",
    "castle": "castello",
    "cat": "gatto",
    "ceiling fan": "ventilatore da soffitto",
    "cello": "violoncello",
    "cell phone": "telefono cellulare",
    "chair": "sedia",
    "chandelier": "lampadario",
    "church": "chiesa",
    "circle": "cerchio",
    "clarinet": "clarinetto",
    "clock": "orologio",
    "cloud": "nuvola",
    "coffee cup": "tazza da caffè",
    "compass": "bussola",
    "computer": "computer",
    "cookie": "biscotto",
    "cooler": "frigorifero portatile",
    "couch": "divano",
    "cow": "mucca",
    "crab": "granchio",
    "crayon": "matita colorata",
    "crocodile": "coccodrillo",
    "crown": "corona",
    "cruise ship": "nave da crociera",
    "cup": "tazza",
    "diamond": "diamante",
    "dishwasher": "lavastoviglie",
    "diving board": "trampolino",
    "dog": "cane",
    "dolphin": "delfino",
    "donut": "ciambella",
    "door": "porta",
    "dragon": "drago",
    "dresser": "cassettiera",
    "drill": "trapano",
    "drums": "tamburi",
    "duck": "anatra",
    "dumbbell": "manubrio",
    "ear": "orecchio",
    "elbow": "gomito",
    "elephant": "elefante",
    "envelope": "busta",
    "eraser": "gomma",
    "eye": "occhio",
    "eyeglasses": "occhiali",
    "face": "viso",
    "fan": "ventilatore",
    "feather": "piuma",
    "fence": "recinto",
    "finger": "dito",
    "fire hydrant": "idrante",
    "fireplace": "camino",
    "firetruck": "autopompa",
    "fish": "pesce",
    "flamingo": "fenicottero",
    "flashlight": "torcia",
    "flip flops": "infradito",
    "floor lamp": "lampada da terra",
    "flower": "fiore",
    "flying saucer": "disco volante",
    "foot": "piede",
    "fork": "forchetta",
    "frog": "rana",
    "frying pan": "padella",
    "garden": "giardino",
    "garden hose": "tubo da giardino",
    "giraffe": "giraffa",
    "goatee": "pizzetto",
    "golf club": "mazza da golf",
    "grapes": "uva",
    "grass": "erba",
    "guitar": "chitarra",
    "hamburger": "hamburger",
    "hammer": "martello",
    "hand": "mano",
    "harp": "arpa",
    "hat": "cappello",
    "headphones": "cuffie",
    "hedgehog": "riccio",
    "helicopter": "elicottero",
    "helmet": "casco",
    "hexagon": "esagono",
    "hockey puck": "disco da hockey",
    "hockey stick": "mazza da hockey",
    "horse": "cavallo",
    "hospital": "ospedale",
    "hot air balloon": "mongolfiera",
    "hot dog": "hot dog",
    "hot tub": "vasca idromassaggio",
    "hourglass": "clessidra",
    "house": "casa",
    "house plant": "pianta da casa",
    "hurricane": "uragano",
    "ice cream": "gelato",
    "jacket": "giacca",
    "jail": "prigione",
    "kangaroo": "canguro",
    "key": "chiave",
    "keyboard": "tastiera",
    "knee": "ginocchio",
    "knife": "coltello",
    "ladder": "scala",
    "lantern": "lanterna",
    "laptop": "portatile",
    "leaf": "foglia",
    "leg": "gamba",
    "light bulb": "lampadina",
    "lighter": "accendino",
    "lighthouse": "faro",
    "lightning": "fulmine",
    "line": "linea",
    "lion": "leone",
    "lipstick": "rossetto",
    "lobster": "aragosta",
    "lollipop": "lecca-lecca",
    "mailbox": "cassetta postale",
    "map": "mappa",
    "marker": "evidenziatore",
    "matches": "fiammiferi",
    "megaphone": "megafono",
    "mermaid": "sirena",
    "microphone": "microfono",
    "microwave": "microonde",
    "monkey": "scimmia",
    "moon": "luna",
    "mosquito": "zanzara",
    "motorbike": "moto",
    "mountain": "montagna",
    "mouse": "topo",
    "moustache": "baffi",
    "mouth": "bocca",
    "mug": "tazza",
    "mushroom": "fungo",
    "nail": "unghia",
    "necklace": "collana",
    "nose": "naso",
    "ocean": "oceano",
    "octagon": "ottagono",
    "octopus": "polpo",
    "onion": "cipolla",
    "oven": "forno",
    "owl": "gufo",
    "paintbrush": "pennello",
    "paint can": "barattolo di pittura",
    "palm tree": "albero di palma",
    "panda": "panda",
    "pants": "pantaloni",
    "paper clip": "graffetta",
    "parachute": "paracadute",
    "parrot": "pappagallo",
    "passport": "passaporto",
    "peanut": "arachide",
    "pear": "pera",
    "peas": "piselli",
    "pencil": "matita",
    "penguin": "pinguino",
    "piano": "pianoforte",
    "pickup truck": "camioncino",
    "picture frame": "cornice",
    "pig": "maiale",
    "pillow": "cuscino",
    "pineapple": "ananas",
    "pizza": "pizza",
    "pliers": "pinze",
    "police car": "auto della polizia",
    "pond": "stagno",
    "pool": "piscina",
    "popsicle": "ghiacciolo",
    "postcard": "cartolina",
    "potato": "patata",
    "power outlet": "presa di corrente",
    "purse": "borsa",
    "rabbit": "coniglio",
    "raccoon": "procione",
    "radio": "radio",
    "rain": "pioggia",
    "rainbow": "arcobaleno",
    "rake": "rastrello",
    "remote control": "telecomando",
    "rhinoceros": "rinoceronte",
    "rifle": "fucile",
    "river": "fiume",
    "roller coaster": "montagne russe",
    "rollerskates": "pattini a rotelle",
    "sailboat": "vela",
    "sandwich": "panino",
    "saw": "sega",
    "saxophone": "sassofono",
    "school bus": "autobus scolastico",
    "scissors": "forbici",
    "scorpion": "scorpione",
    "screwdriver": "cacciavite",
    "sea turtle": "tartaruga marina",
    "see saw": "altalena",
    "shark": "squalo",
    "sheep": "pecora",
    "shoe": "scarpa",
    "shorts": "pantaloncini",
    "shovel": "pala",
    "sink": "lavello",
    "skateboard": "skateboard",
    "skull": "teschio",
    "skyscraper": "grattacielo",
    "sleeping bag": "sacco a pelo",
    "smiley face": "faccina felice",
    "snail": "lumaca",
    "snake": "serpente",
    "snorkel": "snorkel",
    "snowflake": "fiocco di neve",
    "snowman": "uomo di neve",
    "soccer ball": "pallone da calcio",
    "sock": "calzino",
    "speedboat": "motoscafo",
    "spider": "ragno",
    "spoon": "cucchiaio",
    "spreadsheet": "foglio di calcolo",
    "square": "quadrato",
    "squiggle": "linea ondulata",
    "squirrel": "scoiattolo",
    "stairs": "scale",
    "star": "stella",
    "steak": "bistecca",
    "stereo": "stereo",
    "stethoscope": "stetoscopio",
    "stitches": "punture",
    "stop sign": "segnale di stop",
    "stove": "fornello",
    "strawberry": "fragola",
    "streetlight": "lampione",
    "string bean": "fagiolo verde",
    "submarine": "sottomarino",
    "suitcase": "valigia",
    "sun": "sole",
    "swan": "cigno",
    "sweater": "maglione",
    "swing set": "altalena doppia",
    "sword": "spada",
    "syringe": "siringa",
    "table": "tavolo",
    "teapot": "teiera",
    "teddy-bear": "orsacchiotto",
    "telephone": "telefono",
    "television": "televisione",
    "tennis racquet": "racchetta da tennis",
    "tent": "tenda",
    "The Eiffel Tower": "La Torre Eiffel",
    "The Great Wall of China": "La Grande Muraglia Cinese",
    "The Mona Lisa": "La Gioconda",
    "tiger": "tigre",
    "toaster": "tostapane",
    "toe": "dito del piede",
    "toilet": "gabinetto",
    "tooth": "dente",
    "toothbrush": "spazzolino da denti",
    "toothpaste": "dentifricio",
    "tornado": "tornado",
    "tractor": "trattore",
    "traffic light": "semaforo",
    "train": "treno",
    "tree": "albero",
    "triangle": "triangolo",
    "trombone": "trombone",
    "truck": "camion",
    "trumpet": "tromba",
    "t-shirt": "maglietta",
    "umbrella": "ombrello",
    "underwear": "biancheria intima",
    "van": "furgone",
    "vase": "vaso",
    "violin": "violino",
    "washing machine": "lavatrice",
    "watermelon": "anguria",
    "waterslide": "scivolo d'acqua",
    "whale": "balena",
    "wheel": "ruota",
    "windmill": "mulino a vento",
    "wine bottle": "bottiglia di vino",
    "wine glass": "bicchiere di vino",
    "wristwatch": "orologio da polso",
    "yoga": "yoga",
    "zebra": "zebra",
    "zigzag": "zigzag"
}










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

    # print(confs, idxs)

    top_5 = []
    for i in range(len(confs)):
        top_5.append({
            'label': translations[sketch_model.config.id2label[idxs[i].item()]],
            'conf': (confs[i].item() * 100.0)
        })
        
        # print(f"{sketch_model.config.id2label[idxs[i].item()]} | confidence: {(confs[i].item() * 100.0):.4f}")

    return {
        'top': top_5
    }




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
