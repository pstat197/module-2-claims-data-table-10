"""
CNN for Webpage Classification
Author: Lucas Childs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import pickle

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#######################################
# STEP 1: Text Preprocessing for CNN
########################################

class TextPreprocessor:
    def __init__(self, max_vocab_size=10000, max_seq_length=500):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.vocab_size = 2

    def build_vocabulary(self, texts):
        """Build vocabulary from training texts"""
        word_counts = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Add top words to vocabulary
        for word, count in sorted_words[:self.max_vocab_size - 2]:
            self.word2idx[word] = self.vocab_size
            self.vocab_size += 1

        return self.vocab_size

    def texts_to_sequences(self, texts):
        """Convert texts to sequences of indices"""
        sequences = []
        for text in texts:
            seq = [self.word2idx.get(word, 1) for word in text.split()]
            # Truncate or pad
            if len(seq) > self.max_seq_length:
                seq = seq[:self.max_seq_length]
            else:
                seq = seq + [0] * (self.max_seq_length - len(seq))
            sequences.append(seq)
        return np.array(sequences)

#######################################
# STEP 2: CNN Model
#######################################

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_filters=100, filter_sizes=[3, 4, 5], dropout=0.5):
        super(TextCNN, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        # Dropout and output layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)

    def forward(self, x):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_length)

        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, seq_length - kernel_size + 1)
            pooled = F.max_pool1d(conv_out, conv_out.shape[2])  # (batch_size, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)

        # Concatenate outputs from all convolutional layers
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))

        # Dropout and fully connected layer
        dropped = self.dropout(concatenated)
        output = torch.sigmoid(self.fc(dropped))

        return output.squeeze()

#######################################
# STEP 3: Training
#######################################

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    """Train the model"""

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    patience = 15
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
            torch.save(model.state_dict(), 'results/cnn.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('results/cnn.pth'))
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

    # Create preprocessor
    print("Building vocabulary...")
    preprocessor = TextPreprocessor(max_vocab_size=10000, max_seq_length=500)
    vocab_size = preprocessor.build_vocabulary(train_df['text_clean'].values)
    print(f"Vocabulary size: {vocab_size}")

    # Convert texts to sequences
    print("Converting texts to sequences...")
    X_train = preprocessor.texts_to_sequences(train_df['text_clean'].values)
    X_val = preprocessor.texts_to_sequences(val_df['text_clean'].values)
    X_test = preprocessor.texts_to_sequences(test_df['text_clean'].values)

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # Save preprocessor
    with open('results/cnn_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    # Create data loaders
    train_dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Create and train model
    print(f"\nTraining on {device}...")
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=128,
        num_filters=100,
        filter_sizes=[3, 4, 5],
        dropout=0.5
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train_model(model, train_loader, val_loader, epochs=20)

    # Evaluate
    print("\nEvaluating on test set...") # 0.834
    test_preds = evaluate_model(model, test_loader, y_test)

    print("\n Training complete!")
