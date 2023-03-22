
import io
import json
from model import OnnxModel, Preprocessor
import base64
import numpy as np

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    model = OnnxModel('output_model.onnx')


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model
    pre = Preprocessor()

    im_b64 = json.loads(model_inputs).get('prompt', None)
    # convert it into bytes
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    img = pre.preprocess(io.BytesIO(img_bytes))
    result = model.predict(img)
    class_id = np.argmax(result[0])
    # Return the results as a dictionary

    return {'class_id': class_id}
