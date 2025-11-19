import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Dataset, DataLoader
import torch.serialization

# allow numpy scalar unpickling
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbedDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32)

def main():
    ckpt = torch.load(
        "results/mclass_torch_model.pt",
        map_location=DEVICE,
        weights_only=False
    )

    id_to_class = ckpt["id_to_class"]  # keys are INT
    embedding_model_name = ckpt["embedding_model_name"]
    input_dim = ckpt["input_dim"]
    num_classes = ckpt["num_classes"]

    # build classifier
    class Classifier(torch.nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    model = Classifier(input_dim, num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Loaded mapping:", id_to_class)

    df_test = pd.read_csv("data/claims_test_processed.csv")
    X_test = df_test["text_clean"].astype(str).tolist()
    ids = df_test[".id"].tolist()

    embedder = SentenceTransformer(embedding_model_name)

    print("Encoding test...")
    test_embeddings = embedder.encode(
        X_test, batch_size=32, convert_to_numpy=True, show_progress_bar=True
    )

    test_dataset = EmbedDataset(test_embeddings)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    preds = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)

    # FIX: lookup using ints
    pred_labels = [id_to_class[int(x)] for x in preds]

    pred_df = pd.DataFrame({
        ".id": ids,
        "mclass.pred": pred_labels
    })

    pred_df.to_csv("python_preds_torch.csv", index=False)
    pred_df.to_pickle("python_preds_torch.pkl")

    print("Saved python_preds_torch.csv + python_preds_torch.pkl")

if __name__ == "__main__":
    main()
