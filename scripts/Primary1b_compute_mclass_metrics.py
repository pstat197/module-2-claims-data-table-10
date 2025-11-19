# scripts/Primary1b_compute_val_metrics.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Fix serialization issue
import numpy as np
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


class ClaimsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def main():
    print("Loading cleaned training data...")
    df = pd.read_csv("data/claims_clean_processed.csv")

    X_text = df["text_clean"].astype(str).tolist()
    y_str = df["mclass"].astype(str).tolist()

    # Recreate label encoding exactly how it was fit
    classes = sorted(list(set(y_str)))
    class_to_id = {cls: i for i, cls in enumerate(classes)}
    y = np.array([class_to_id[c] for c in y_str])

    # Use same split as training script
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Loading model...")
    ckpt = torch.load("results/mclass_torch_model.pt", map_location=DEVICE, weights_only=False)

    id_to_class = {int(k): str(v) for k, v in ckpt["id_to_class"].items()}
    input_dim = ckpt["input_dim"]
    num_classes = ckpt["num_classes"]

    embedder = SentenceTransformer(ckpt["embedding_model_name"])

    print("Encoding validation set...")
    val_embeddings = embedder.encode(
        X_val, batch_size=32, convert_to_numpy=True, show_progress_bar=True
    )

    val_dataset = ClaimsDataset(val_embeddings, y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load classifier
    model = Classifier(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_true.extend(yb.cpu().numpy().tolist())

    # Convert numbers → labels
    pred_labels = [id_to_class[p] for p in all_preds]
    true_labels = [id_to_class[t] for t in all_true]

    print("\n=== VALIDATION METRICS ===")
    print("\nAccuracy:", accuracy_score(true_labels, pred_labels))
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels))


if __name__ == "__main__":
    main()
