"""
Evaluate CNN model and compute metrics (accuracy, sensitivity, specificity)
without retraining
"""
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from lucas_primary_a_preprocess import load_and_split_data
from cnn import TextCNN, TextPreprocessor

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading data...")
train_df, val_df, test_df = load_and_split_data()

# Load saved model and preprocessor
print("Loading saved model and preprocessor...")
model = torch.load('results/cnn_model.pth')
model.to(device)
model.eval()

with open('results/cnn_preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Preprocess test data
print("Preprocessing test data...")
X_test = preprocessor.texts_to_sequences(test_df['text_clean'].values)
y_test = test_df['label'].values

# Create data loader
test_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=32)

# Make predictions
print("Making predictions...")
predictions = []
true_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        preds = (outputs > 0.5).float().cpu().numpy()

        # Handle both single predictions and batches
        if preds.ndim == 0:
            predictions.append(preds.item())
        else:
            predictions.extend(preds)

        true_labels.extend(batch_y.numpy())

predictions = np.array(predictions)
true_labels = np.array(true_labels)

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)

# Confusion matrix
cm = confusion_matrix(true_labels, predictions)
tn, fp, fn, tp = cm.ravel()

# Calculate sensitivity (recall/true positive rate)
sensitivity = tp / (tp + fn)

# Calculate specificity (true negative rate)
specificity = tn / (tn + fp)

# Create a table with pandas
import pandas as pd

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Sensitivity', 'Specificity'],
    'Value': [accuracy, sensitivity, specificity]
})

print("METRICS TABLE")
print(metrics_df.to_string(index=False))

