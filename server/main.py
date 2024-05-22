from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware

import torch
import timm
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)


# Predict using the model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


class TomatoLeavesClassifer(nn.Module):
    def __init__(self, num_classes=10):
        super(TomatoLeavesClassifer, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Define the directory where the uploaded files will be saved
        upload_directory = "uploaded_images"
        os.makedirs(upload_directory, exist_ok=True)

        # Create a path for the file
        file_path = os.path.join(upload_directory, file.filename)

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor()]
        )

        state_dict = torch.load("model.pth", map_location=torch.device("cpu"))
        model = TomatoLeavesClassifer(10)
        model.load_state_dict(state_dict)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        original_image, image_tensor = preprocess_image(file_path, transform)
        probabilities = predict(model, image_tensor, device)
        print(probabilities)

        class_names = [
            "Tomato Bacterial spot",
            "Tomato Early blight",
            "Tomato Healthy",
            "Tomato Late blight",
            "Tomato Leaf Mold",
            "Tomato Septoria leaf spot",
            "Tomato Spider mites Two spotted spider mite",
            "Tomato Target Spot",
            "Tomato Tomato YellowLeaf Curl Virus",
            "Tomato Tomato mosaic virus",
        ]

        max_p = 0
        for i, class_name in enumerate(class_names):
            if probabilities[i] > max_p:
                max_p = probabilities[i]
                index = i

        return JSONResponse(status_code=200, content={"filename": class_names[index]})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
