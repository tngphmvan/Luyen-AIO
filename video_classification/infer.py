import torch
import cv2
import numpy as np
from torchvision import transforms
from video_cnn_trainer import VideoCNNClassifier


def load_model(checkpoint_path, model_name='r3d_18', num_classes=2):
    """Load trained model"""
    model = VideoCNNClassifier(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.eval()
    return model


def predict_video(model, video_path, transform=None, frames_per_clip=16):
    """Predict class for a video file"""
    if transform is None:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.43216, 0.394666, 0.37645],
                                 [0.22803, 0.22145, 0.216989])
        ])

    # Read video frames
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, frame_count-1, frames_per_clip, dtype=int)
    frames = []

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((112, 112, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)

    cap.release()

    # Prepare input
    video = torch.stack(frames).permute(1, 0, 2, 3)  # (C, T, H, W)
    video = video.unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(video)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    return pred_class, confidence


if __name__ == "__main__":
    # Example usage
    checkpoint_path = "checkpoints/video_cnn/best-model.ckpt"
    video_path = "path/to/your/video.mp4"

    # Load model
    model = load_model(checkpoint_path, model_name='r3d_18', num_classes=2)

    # Predict
    pred_class, confidence = predict_video(model, video_path)
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.2f}")
