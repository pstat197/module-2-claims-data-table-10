# scripts/train_mclass_torch.py

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

# ----------------------------
# Config
# ----------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 10
HIDDEN_DIM1 = 256
HIDDEN_DIM2 = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Dataset class
# ----------------------------

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


# ----------------------------
# Model
# ----------------------------

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM1, HIDDEN_DIM2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Main training procedure
# ----------------------------

def main():
    # Load processed training data
    df = pd.read_csv("data/claims_clean_processed.csv")

    # Feature + labels
    X_text = df["text_clean"].astype(str).tolist()
    y_str = df["mclass"].astype(str).tolist()

    # Label encoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    num_classes = len(label_encoder.classes_)

    class_to_id = {cls: int(i) for i, cls in enumerate(label_encoder.classes_)}
    id_to_class = {int(v): k for k, v in class_to_id.items()}

    print("\nClasses:", class_to_id)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # Embedding model
    print("\nLoading SentenceTransformer model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Encode text → embeddings
    print("Encoding training set...")
    train_embeddings = embedder.encode(
        X_train, batch_size=32, convert_to_numpy=True, show_progress_bar=True
    )

    print("Encoding validation set...")
    val_embeddings = embedder.encode(
        X_val, batch_size=32, convert_to_numpy=True, show_progress_bar=True
    )

    input_dim = train_embeddings.shape[1]

    # Build datasets/loaders
    train_dataset = ClaimsDataset(train_embeddings, y_train)
    val_dataset = ClaimsDataset(val_embeddings, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    model = Classifier(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\nStarting training...")
    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_train_loss = total_loss / len(train_dataset)

        # ---- Validation ----
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")

    print("\nFINAL VAL ACCURACY:", val_acc)

    # ---- Retrain full model ----
    print("\nEncoding full dataset...")
    all_embeddings = embedder.encode(
        X_text, batch_size=32, convert_to_numpy=True, show_progress_bar=True
    )

    all_dataset = ClaimsDataset(all_embeddings, y)
    all_loader = DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Classifier(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Training final model on ALL data...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for xb, yb in all_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        print(f"Full Train Epoch {epoch}/{EPOCHS} - Loss: {total_loss/len(all_dataset):.4f}")

    # ---- Save model ----
    save_path = "results/mclass_torch_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_id": class_to_id,
        "id_to_class": id_to_class,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "input_dim": input_dim,
        "num_classes": num_classes
    }, save_path)

    print(f"\nSaved model to {save_path}")

    with open("results/mclass_label_mapping.json", "w") as f:
        json.dump({"class_to_id": class_to_id, "id_to_class": id_to_class}, f, indent=2)

    print("Saved label mapping JSON.\n")


if __name__ == "__main__":
    main()
