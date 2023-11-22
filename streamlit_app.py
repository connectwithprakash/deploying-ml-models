import streamlit as st
import torch
import torchvision
from PIL import Image
import io

DEVICE = torch.device('cpu')  # Do inference on CPU Only
# Change path to where you saved to the model


@st.cache_resource
def get_model():
    print("Loading model...")
    model = torch.load("./model.pt", map_location=DEVICE)
    model.eval()  # Put model in eval mode
    return model


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
    model = get_model()
    img_tensor = preprocess_image(req_image_file).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(img_tensor).argmax(dim=-1)[0].item()
    return pred


def app():
    st.title("CNN Fashion MNIST Classifier")
    model_name = st.sidebar.selectbox('Model', ['CNN', 'Transformer'])

    input_image = st.file_uploader(
        label="Upload an image of a fashion item", type=['png', 'jpg', 'jpeg'])

    if input_image is not None:
        st.image(input_image, width=200)
        pred = get_prediction(input_image)
        st.write(f"Prediction: {IDX_TO_CLASS[pred]}")


if __name__ == '__main__':
    app()
