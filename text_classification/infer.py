import torch
from transformers import AutoTokenizer
from gru_trainer import GRUModel


def load_model(checkpoint_path):
    """Load trained model"""
    model = GRUModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def predict_text(model, text, tokenizer, max_length=128):
    """Predict class for a single text"""
    # Tokenize
    inputs = tokenizer(text,
                       max_length=max_length,
                       padding='max_length',
                       truncation=True,
                       return_tensors='pt')

    # Predict
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    return pred_class, confidence


if __name__ == "__main__":
    # Example usage
    checkpoint_path = "checkpoints/gru/best-gru.ckpt"
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    # Load model
    model = load_model(checkpoint_path)

    # Example prediction
    text = "Đây là văn bản mẫu để test"
    pred_class, confidence = predict_text(model, text, tokenizer)
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.2f}")
