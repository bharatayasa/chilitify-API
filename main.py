import os
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

model_path = './model/model.h5'

model = load_model(model_path)

def load_and_preprocess_image(image: UploadFile, target_size=(224, 224)):
    image_bytes = BytesIO(image.file.read())
    img = load_img(image_bytes, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0 
    return img_array

def predict_image(image: UploadFile, model):
    img_array = load_and_preprocess_image(image)
    prediction = model.predict(img_array)
    class_names = ['Healthy', 'Leaf Curl', 'Leaf Spot', 'Whitefly', 'Yellowish']
    
    description = [
        'Tanaman cabai sehat menunjukkan pertumbuhan yang normal dengan daun hijau segar tanpa tanda-tanda kerusakan atau infeksi penyakit. Batang dan buah juga tampak sehat tanpa gejala abnormal', 
        'Penyakit ini menyebabkan daun tanaman cabai mengeriting dan menggulung. Biasanya disebabkan oleh virus, kutu daun, atau hama lainnya yang menghisap cairan dari daun.', 
        'Penyakit ini ditandai dengan munculnya bercak-bercak coklat atau hitam pada daun. Penyebab utamanya adalah infeksi jamur atau bakteri', 
        'Hama lalat putih menghisap cairan tanaman dan menyebabkan daun menjadi kuning, layu, dan rontok. Mereka juga dapat menyebarkan berbagai penyakit tanaman', 
        'Tanaman cabai yang mengalami kekuningan biasanya menunjukkan gejala daun yang menguning akibat defisiensi nutrisi, overwatering, atau serangan hama/penyakit'
    ]

    prevention = [
        'Menjaga kesehatan tanaman cabai memerlukan pendekatan holistik. Penting untuk menjaga kebersihan lingkungan tanam dan memastikan tanaman mendapat cukup nutrisi serta air. Rotasi tanaman juga sangat disarankan untuk mencegah penumpukan patogen di tanah. Pemantauan rutin perlu dilakukan untuk mendeteksi dini adanya tanda-tanda penyakit sehingga tindakan cepat bisa diambil.', 
        'Pengendalian keriting daun bisa dimulai dengan mengontrol populasi kutu daun menggunakan insektisida atau predator alami seperti ladybugs. Penting juga untuk menjaga kebersihan alat-alat pertanian guna mencegah penyebaran virus. Menanam varietas cabai yang tahan terhadap penyakit keriting daun sangat dianjurkan. Selain itu, tanaman yang sudah terinfeksi harus segera dihilangkan dan dibakar untuk mencegah penyebaran lebih lanjut.', 
        'Pencegahan bercak daun bisa dilakukan dengan menjaga kelembaban lingkungan tanam agar tidak terlalu tinggi. Penggunaan fungisida atau bakterisida yang tepat sesuai dengan penyebab infeksi sangat penting. Daun yang terinfeksi harus segera dibuang dan dibakar untuk mencegah penyebaran. Menghindari penyiraman dari atas juga dapat mengurangi penyebaran spora jamur atau bakteri yang menjadi penyebab utama penyakit ini.', 
        'Untuk mengatasi lalat putih, gunakan perangkap lengket berwarna kuning yang efektif dalam mengurangi populasi lalat putih. Penyemprotan tanaman dengan insektisida berbasis minyak neem atau sabun insektisida juga dianjurkan. Menjaga kebersihan lingkungan sekitar tanaman membantu mengurangi tempat bertelur lalat putih. Selain itu, memperkenalkan predator alami seperti lacewing atau ladybugs dapat menjadi solusi biologis yang efektif.', 
        'Memastikan tanaman mendapatkan nutrisi yang cukup, terutama nitrogen, magnesium, dan zat besi sangat penting untuk mencegah daun kekuningan. Penyiraman harus diatur agar tidak berlebihan, dengan menjaga kelembaban tanah yang optimal. Mengontrol populasi hama seperti kutu daun dan tungau juga krusial untuk mencegah kekuningan. Melakukan pemupukan secara teratur dengan pupuk yang seimbang akan membantu tanaman tetap sehat dan hijau.'
    ]
    predicted_class = class_names[np.argmax(prediction)]
    predicted_description = description[np.argmax(prediction)]
    predicted_prevention = prevention[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return predicted_class, predicted_description, predicted_prevention, confidence

app = FastAPI()

@app.get("/")
def index():
    return {"message": "Hello world from ML endpoint!"}

@app.post("/predict_image/")
async def predict_image_endpoint(file: UploadFile = File(...)):
    try:
        predicted_class, predicted_description, predicted_prevention, confidence = predict_image(file, model)
        return {
            "data": {
                "class": predicted_class,
                "description": predicted_description,
                "prevention": predicted_prevention,
                "confidence": confidence,
            }
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Listening to http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
