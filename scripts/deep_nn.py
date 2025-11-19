"""
Deep Neural Network for Webpage Classification
Author: Lucas Childs
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#######################################
# Feature Engineering
#######################################

def create_features(train_texts, val_texts, test_texts):
    """Create TF-IDF features"""

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_val = vectorizer.transform(val_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, vectorizer, scaler

#######################################
# Deep NN Model
#######################################

class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze()

#######################################
# Training
#######################################

def train_model(model, train_loader, val_loader, epochs=30, lr=0.001):
    """Train the model"""

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                preds = (outputs > 0.5).float().cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(batch_y.numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'results/deep_nn.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('results/deep_nn.pth'))
    return model

def evaluate_model(model, test_loader, y_test):
    """Evaluate on test set"""

    model.eval()
    test_preds = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = (outputs > 0.5).float().cpu().numpy()
            test_preds.extend(preds)

    test_acc = accuracy_score(y_test, test_preds)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    return test_preds

#######################################
# Main execution
#######################################

if __name__ == "__main__":
    import os
    from lucas_primary_A import load_and_split_data

    os.makedirs('results', exist_ok=True)

    print("Loading and preprocessing data...")
    train_df, val_df, test_df = load_and_split_data()

    # Create features
    print("Creating features...")
    X_train, X_val, X_test, vectorizer, scaler = create_features(
        train_df['text_clean'].values,
        val_df['text_clean'].values,
        test_df['text_clean'].values
    )

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # Save preprocessors
    with open('results/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('results/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Create and train model
    print(f"\nTraining on {device}...")
    model = DeepNN(input_dim=X_train.shape[1]).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train_model(model, train_loader, val_loader, epochs=30)

    # Evaluate
    print("\nEvaluating on test set...") # 0.8014
    test_preds = evaluate_model(model, test_loader, y_test)

    print("\n Training complete!")
