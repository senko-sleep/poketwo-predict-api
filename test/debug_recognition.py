import numpy as np
import json

# Load and check the labels
data = np.load('c:\\Users\\Owner\\Anya-Bot-1\\pokemon-predict-api\\models\\event_embedding_index.npz')
labels = data['labels']

print(f"Labels dtype: {labels.dtype}")
print(f"Labels shape: {labels.shape}")
print(f"First 5 labels: {labels[:5]}")
print(f"Type of first label: {type(labels[0])}")
print(f"Type of labels: {type(labels)}")

# Try to convert to strings
try:
    labels_list = [str(label) if label is not None else None for label in labels]
    print(f"Successfully converted to list")
    print(f"First 5 converted: {labels_list[:5]}")
except Exception as e:
    print(f"Error converting: {e}")
