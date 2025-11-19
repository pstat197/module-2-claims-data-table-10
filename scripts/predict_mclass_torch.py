# scripts/predict_mclass_torch.py

import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer


BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClaimsTestDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        return x


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
    # Load test data
    df_test = pd.read_csv("data/claims_test_processed.csv")
    X_test = df_test["text_clean"].astype(str).tolist()
    ids_test = df_test[".id"].tolist()

    # Load checkpoint
    ckpt = torch.load(
    "results/mclass_torch_model.pt",
    map_location=DEVICE,
    weights_only=False  # allow loading the full checkpoint dict
    )
    class_to_id = ckpt["class_to_id"]
    id_to_class = ckpt["id_to_class"]
    embedding_model_name = ckpt["embedding_model_name"]
    input_dim = ckpt["input_dim"]
    num_classes = ckpt["num_classes"]

    # Load embedding model
    embedder = SentenceTransformer(embedding_model_name)

    print("Encoding test text...")
    test_embeddings = embedder.encode(
        X_test, batch_size=32, convert_to_numpy=True, show_progress_bar=True
    )

    # Load model
    model = Classifier(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Predict
    test_dataset = ClaimsTestDataset(test_embeddings)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_pred_ids = []

    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_pred_ids.extend(preds.cpu().numpy().tolist())

    all_pred_labels = [id_to_class[int(i)] for i in all_pred_ids]

    # Build DF for export
    pred_df = pd.DataFrame({
        ".id": ids_test,
        "bclass.pred": ["N/A: No relevant content."] * len(ids_test),
        "mclass.pred": all_pred_labels
    })

    out_path = "python_preds_torch.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path}\n")


if __name__ == "__main__":
    main()
