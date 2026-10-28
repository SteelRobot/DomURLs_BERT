import argparse
import pickle
import torch
from argparse import Namespace
from pathlib import Path

from main_charnn import get_class_from_name
from models.rnn import CharBiGRU
from module.pl_module import BaseModel

def encode_url(url, max_length):
    vocabulary = list(
        """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'"\\/|_@#$%^&*~`+-=<>()[]{}"""
    )
    ids = [vocabulary.index(c) + 2 if c in vocabulary else 1 for c in url]
    ids = ids[:max_length]
    ids += [0] * (max_length - len(ids))
    return torch.tensor([ids], dtype=torch.long)

parser = argparse.ArgumentParser(description="Classify a URL using a trained CharBiGRU.")
parser.add_argument("--metadata", required=True, help="Path to model_metadata.p")
parser.add_argument("--url", required=True, help="URL to classify")
args = parser.parse_args()

metadata_path = Path(args.metadata)
checkpoint_path = metadata_path.parent / "best_checkpoint.ckpt"

if not checkpoint_path.is_file():
    parser.error(f"Checkpoint not found: {checkpoint_path}")

with open(args.metadata, "rb") as f:
    metadata = pickle.load(f)

num_classes = metadata["num_classes"]

classifier = get_class_from_name(metadata["model_name"])(
    **metadata["model_params"]
)

model = BaseModel.load_from_checkpoint(
    checkpoint_path,
    classifier=classifier,
    num_classes=num_classes,
    criterion=torch.nn.CrossEntropyLoss(),
    config=Namespace(**metadata),
    names=metadata["label_encoder"].classes_,
)
model.eval()
device = next(model.parameters()).device

with torch.inference_mode():
    inputs = encode_url(args.url, metadata["max_length"]).to(device)
    probabilities = torch.softmax(model(inputs), dim=1)[0].cpu()

labels = metadata["label_encoder"].classes_
predicted = probabilities.argmax().item()

print(f"Prediction: {labels[predicted]}")
print("Probabilities:", dict(zip(labels, probabilities.tolist())))
