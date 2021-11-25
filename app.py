from flask import Flask, request, jsonify
import skimage.io
import mrcnn.model as modellib
from mrcnn.config import Config

app = Flask(__name__)
class_names = ['BG', 'fried egg']

class foodConfig(Config):
    NAME = "food"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1

class InferenceConfig(foodConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

@app.route('/api/predict/', methods=['GET', 'POST'])
def predict():
    if request.files.get('image'):
        image = skimage.io.imread(request.files['image'])
        results = model.detect([image], verbose=1)
        r = results[0]
        res = []
        for i in r['class_ids']:
            res.append(class_names[i])
            print(class_names[i])
        return jsonify(res)
    return "No Image"

@app.route('/', methods=['GET'])
def home():
    skimage_ver = skimage.__version__
    print(skimage_ver)
    return "Bnn-project ml API " + skimage_ver

if __name__ == '__main__':
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config, model_dir="/logs")
    model.load_weights('mask_rcnn_food_0030.h5', by_name=True)
    model.keras_model._make_predict_function()
    app.run()