import json
import os

labels_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'labels_v2.json')
with open(labels_path, 'r') as f:
    data = json.load(f)

print(f'Total entries in labels file: {len(data)}')
print(f'Key range: {min(data.keys(), key=int)} to {max(data.keys(), key=int)}')

# Check for contiguous keys
keys_int = sorted([int(k) for k in data.keys()])
expected_keys = list(range(len(data)))
missing_keys = [k for k in expected_keys if k not in keys_int]
extra_keys = [k for k in keys_int if k >= len(data)]

print(f'Missing keys (0 to {len(data)-1}): {missing_keys[:10]}...' if len(missing_keys) > 10 else f'Missing keys: {missing_keys}')
print(f'Extra keys (>= {len(data)}): {extra_keys[:10]}...' if len(extra_keys) > 10 else f'Extra keys: {extra_keys}')

print(f'\nModel outputs 1290 classes, but labels file has keys up to {max(keys_int)}')
print(f'Labels needed for model: 0-1289')
print(f'Labels available in file: 0-{max(keys_int)}')
