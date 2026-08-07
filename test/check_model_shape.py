import onnxruntime as ort
import os

session = ort.InferenceSession(os.path.join(os.path.dirname(__file__), '..', 'models', 'pokemon_cnn_v2.onnx'))
output_shape = session.get_outputs()[0].shape
print(f'Model output shape: {output_shape}')
print(f'Number of classes: {output_shape[-1] if isinstance(output_shape[-1], int) else "dynamic"}')
