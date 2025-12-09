from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os                     

app = Flask(__name__)

# 1. Load Model
model = load_model('kupukupu_model.h5')

# 2. Daftar Kelas (PASTE HASIL DARI COLAB DI SINI!)
# Ganti list kosong di bawah ini dengan list yang Anda copy dari Langkah 1
class_names = ['ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'AN 88', 'APPOLLO', 'ATALA', 'BANDED ORANGE HELICONIAN', 'BANDED PEACOCK', 'BECKERS WHITE', 'BLACK HAIRSTREAK', 'BLUE MORPHO', 'BLUE SPOTTED CROW', 'BROWN SIPROETA', 'CABBAGE WHITE', 'CAIRNS BIRDWING', 'CHECQUERED SKIPPER', 'CHESTNUT', 'CLEOPATRA', 'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR', 'COMMON BANDED AWL', 'COMMON WOOD-NYMPH', 'COPPER TAIL', 'CRECENT', 'CRIMSON PATCH', 'DANAID EGGFLY', 'EASTERN COMA', 'EASTERN DAPPLE WHITE', 'EASTERN PINE ELFIN', 'ELBOWED PIERROT', 'GOLD BANDED', 'GREAT EGGFLY', 'GREAT JAY', 'GREEN CELLED CATTLEHEART', 'GREY HAIRSTREAK', 'INDRA SWALLOW', 'IPHICLUS SISTER', 'JULIA', 'LARGE MARBLE', 'MALACHITE', 'MANGROVE SKIPPER', 'MESTRA', 'METALMARK', 'MILBERTS TORTOISESHELL', 'MONARCH', 'MOURNING CLOAK', 'ORANGE OAKLEAF', 'ORANGE TIP', 'ORCHARD SWALLOW', 'PAINTED LADY', 'PAPER KITE', 'PEACOCK', 'PINE WHITE', 'PIPEVINE SWALLOW', 'POPINJAY', 'PURPLE HAIRSTREAK', 'PURPLISH COPPER', 'QUESTION MARK', 'RED ADMIRAL', 'RED CRACKER', 'RED POSTMAN', 'RED SPOTTED PURPLE', 'SCARCE SWALLOW', 'SILVER SPOT SKIPPER', 'SLEEPY ORANGE', 'SOOTYWING', 'SOUTHERN DOGFACE', 'STRAITED QUEEN', 'TROPICAL LEAFWING', 'TWO BARRED FLASHER', 'ULYSES', 'VICEROY', 'WOOD SATYR', 'YELLOW SWALLOW TAIL', 'ZEBRA LONG WING'] # <--- CONTOH SAJA, GANTI DENGAN LIST LENGKAP ANDA

# Folder untuk menyimpan gambar sementara yang diupload user
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = "Belum ada prediksi"
    confidence = 0
    img_path = None

    if request.method == 'POST':
        # Cek apakah ada file yang diupload
        if 'file' not in request.files:
            return render_template('index.html', prediction="Tidak ada file")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="Tidak ada file dipilih")

        if file:
            # Simpan file gambar
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img_path = filepath

            # --- PROSES PREDIKSI ---
            # 1. Load & Resize Gambar (224x224 sesuai training)
            img = image.load_img(filepath, target_size=(224, 224))
            
            # 2. Ubah ke Array
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            # 3. Preprocessing (PENTING: Harus sama dengan MobileNetV2 training)
            x = preprocess_input(x)

            # 4. Prediksi
            preds = model.predict(x)
            result_index = np.argmax(preds)
            
            prediction = class_names[result_index]
            confidence = round(np.max(preds) * 100, 2)

    return render_template('index.html', prediction=prediction, confidence=confidence, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
