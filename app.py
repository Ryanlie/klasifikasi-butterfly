from flask import Flask, render_template, request, make_response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os


app = Flask(__name__)

# 1. Load Model
model = load_model("kupukupu_model.h5")

# 2. Daftar Kelas (PASTE HASIL DARI COLAB DI SINI!)
# Ganti list kosong di bawah ini dengan list yang Anda copy dari Langkah 1
class_names = [
    "Adonis Butterfly",
    "African Giant Swallowtail Butterfly",
    "American Snoot Butterfly",
    "An 88 Butterfly",
    "Appollo Butterfly",
    "Atala Butterfly",
    "Banded Orange Heliconian Butterfly",
    "Banded Peacock Butterfly",
    "Beckers White Butterfly",
    "Black Hairstreak Butterfly",
    "Blue Morpho Butterfly",
    "Blue Spotted Crow Butterfly",
    "Brown Siproeta Butterfly",
    "Cabbage White Butterfly",
    "Cairns Birdwing Butterfly",
    "Checquered Skipper Butterfly",
    "Chestnut Butterfly",
    "Cleopatra Butterfly",
    "Clodius Parnassian Butterfly",
    "Clouded Sulphur Butterfly",
    "Common Banded Awl Butterfly",
    "Common Wood-Nymph Butterfly",
    "Copper Tail Butterfly",
    "Crecent Butterfly",
    "Crimson Patch Butterfly",
    "Danaid Eggfly Butterfly",
    "Eastern Coma Butterfly",
    "Eastern Dapple White Butterfly",
    "Eastern Pine Elfin Butterfly",
    "Elbowed Pierrot Butterfly",
    "Gold Banded Butterfly",
    "Great Eggfly Butterfly",
    "Great Jay Butterfly",
    "Green Celled Cattleheart Butterfly",
    "Grey Hairstreak Butterfly",
    "Indra Swallow Butterfly",
    "Iphiclus Sister Butterfly",
    "Julia Butterfly",
    "Large Marble Butterfly",
    "Malachite Butterfly",
    "Mangrove Skipper Butterfly",
    "Mestra Butterfly",
    "Metalmark Butterfly",
    "Milberts Tortoiseshell Butterfly",
    "Monarch Butterfly",
    "Mourning Cloak Butterfly",
    "Orange Oakleaf Butterfly",
    "Orange Tip Butterfly",
    "Orchard Swallow Butterfly",
    "Painted Lady Butterfly",
    "Paper Kite Butterfly",
    "Peacock Butterfly",
    "Pine White Butterfly",
    "Pipevine Swallow Butterfly",
    "Popinjay Butterfly",
    "Purple Hairstreak Butterfly",
    "Purplish Copper Butterfly",
    "Question Mark Butterfly",
    "Red Admiral Butterfly",
    "Red Cracker Butterfly",
    "Red Postman Butterfly",
    "Red Spotted Purple Butterfly",
    "Scarce Swallow Butterfly",
    "Silver Spot Skipper Butterfly",
    "Sleepy Orange Butterfly",
    "Sootywing Butterfly",
    "Southern Dogface Butterfly",
    "Straited Queen Butterfly",
    "Tropical Leafwing Butterfly",
    "Two Barred Flasher Butterfly",
    "Ulyses Butterfly",
    "Viceroy Butterfly",
    "Wood Satyr Butterfly",
    "Yellow Swallow Tail Butterfly",
    "Zebra Long Wing Butterfly",
]

UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET"])
def get():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def post():
    prediction = None
    confidence = 0
    img_path = None

    if "file" not in request.files:
        return {"prediction": "Tidak ada file!"}

    file = request.files["file"]
    if file.filename == "":
        return {"prediction": "Tidak ada file dipilih!"}

    try:
        filename = file.filename
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(img_path)

        # 1. Load & Resize Gambar (224x224 sesuai training)
        img = image.load_img(img_path, target_size=(224, 224))

        # 2. Ubah ke Array
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # 3. Preprocessing
        x = preprocess_input(x)

        # 4. Prediksi
        preds = model.predict(x)
        result_index = np.argmax(preds)

        prediction = class_names[result_index]
        confidence = round(np.max(preds) * 100, 2)

        return {
            "prediction": prediction,
            "confidence": str(confidence),
            "img_path": img_path,
        } , 200

    except:
        return None


if __name__ == "__main__":
    app.run(debug=True)
