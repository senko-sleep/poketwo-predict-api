import numpy as np

# Check the structure of the new embedding index
data = np.load('c:\\Users\\Owner\\Anya-Bot-1\\pokemon-predict-api\\models\\event_embedding_index.npz')

print("Files in npz:")
for file in data.files:
    print(f"  {file}")
    if file != 'labels':
        arr = data[file]
        print(f"    Shape: {arr.shape}")
        print(f"    Dtype: {arr.dtype}")
        if len(arr.shape) == 1:
            print(f"    Sample: {arr[:5]}")
        else:
            print(f"    Sample shape: {arr.shape}")
