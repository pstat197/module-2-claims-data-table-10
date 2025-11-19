"""
Make predictions using saved CNN model for binary classification
"""
import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
from lucas_primary_a_preprocess import load_data, clean_html
from cnn import TextCNN, TextPreprocessor  # Import model classes

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved model (entire model)
print("Loading saved model...")
model = torch.load('results/cnn_model.pth')
model.to(device)
model.eval()

# Load the preprocessor (vocabulary and settings)
print("Loading preprocessor...")
with open('results/cnn_preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load new data
print("Loading new data...")
test_data_df = load_data('data/claims-test.RData')
test_data_df['text_clean'] = test_data_df['text_tmp'].apply(clean_html)
test_data_df = test_data_df[test_data_df['text_clean'].str.len() > 0].copy()

# Preprocess and predict
print("Making predictions...")
X_test = preprocessor.texts_to_sequences(test_data_df['text_clean'].values)
test_dataset = TensorDataset(torch.LongTensor(X_test))
test_loader = DataLoader(test_dataset, batch_size=32)

predictions = []
with torch.no_grad():
    for batch in test_loader:
        batch_x = batch[0].to(device)
        outputs = model(batch_x)
        preds = (outputs > 0.5).float().cpu().numpy()
        # Handle both single predictions and batches
        if preds.ndim == 0:
            predictions.append(preds.item())
        else:
            predictions.extend(preds)

print(f"Made {len(predictions)} predictions")

# Map binary predictions back to categorical labels
# Mapping: 0 to "N/A: No relevant content.", 1 to "Relevant claim content"
label_map = {0: "N/A: No relevant content.", 1: "Relevant claim content"}
categorical_predictions = [label_map[int(p)] for p in predictions]

# Add to dataframe
test_data_df['bclass.pred'] = categorical_predictions

# Create final predictions dataframe with .id and bclass.pred
predictions_df = test_data_df[['.id', 'bclass.pred']].copy()

# Save predictions as RData
print(f"\nSaving predictions to results/preds-group[10].RData...")
import pyreadr
pyreadr.write_rdata('results/preds-group[10].RData', predictions_df, df_name='predictions')

print(f"Predicted class distribution:\n{predictions_df['bclass.pred'].value_counts()}")