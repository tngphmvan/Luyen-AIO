import torch
import cv2
import numpy as np
from functools import cache
from torchvision.models import EfficientNet_B3_Weights
from trainer import WrapperModel
from data_prepare import dataset
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # Fixed device check


@cache
def _get_model(ckpt_path: str) -> WrapperModel:
    model = WrapperModel.load_from_checkpoint(checkpoint_path=ckpt_path)
    model.eval()  # Set model to eval mode
    model.to(device)
    return model


data_classes = dataset.classes
transforms = EfficientNet_B3_Weights.IMAGENET1K_V1.transforms()


def inference(img_path: str, ckpt_path: str):
    # Load model with checkpoint path
    model = _get_model(ckpt_path)

    # Read and preprocess image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image for transforms
    pil_img = Image.fromarray(img_rgb)

    # Apply transforms and add batch dimension
    img_tensor = transforms(pil_img).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        # probs = torch.softmax(output[0], dim=0).cpu().numpy()
        probs = output[0].cpu().numpy()
        print(probs)
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]

    # Draw on image
    pred_class = data_classes[pred_idx]
    text = f"Class: {pred_class} ({confidence:.2%})"

    # Add text to image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (10, 30), font, 1, (0, 255, 0), 2)

    # Show image
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return {
        "class": pred_class,
        "confidence": float(confidence),
        "probabilities": {cls: float(prob) for cls, prob in zip(data_classes, probs)}
    }


if __name__ == "__main__":
    # Example usage
    result = inference(
        img_path=r"C:\Users\tungp\OneDrive\Pictures\Screenshots\Screenshot 2025-10-17 153058.png",
        ckpt_path=r"D:\best.ckpt"
    )
    print(f"Predicted class: {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")
