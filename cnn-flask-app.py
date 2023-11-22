import io
from PIL import Image

import torch
import torchvision
from flask import Flask, jsonify, request as fl_req
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

DEVICE = torch.device('cpu')  # Do inference on CPU Only
# Change path to where you saved to the model
model = torch.load("./model.pt", map_location=DEVICE)
model.eval()  # Put model in eval mode

# Preprocessing pipeline
IMAGE_SHAPE = (1, 28, 28)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(IMAGE_SHAPE[-2:]),
    torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=0.5, std=0.5),
])

IDX_TO_CLASS = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot',
}


def preprocess_image(req_image_file):
    # apply transform
    return transform(Image.open(io.BytesIO(req_image_file.read())))


def get_prediction(req_image_file):
    # also add batch dimension to processed image
    img_tensor = preprocess_image(req_image_file).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(img_tensor).argmax(dim=-1)[0].item()
    return pred


limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["2 per minute", "1 per second"],
)


@app.route('/predict', methods=['POST'])
@limiter.limit("5 per minute")
def predict():
    if fl_req.method == 'POST':
        image_file = fl_req.files['image']
        pred = get_prediction(image_file)

        return jsonify({
            'class_id': pred,
            'class_name': IDX_TO_CLASS[pred]
        })


if __name__ == '__main__':
    app.run()
