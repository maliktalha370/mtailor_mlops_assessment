
import numpy as np
from PIL import Image
from torchvision import transforms
import onnxruntime as rt

class OnnxModel:
    def __init__(self, model_path):
        self.session = rt.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_data):
        input_data = np.expand_dims(input_data, axis=0)
        input_data = input_data.astype('float32')
        output = self.session.run([self.output_name], {self.input_name: input_data})[0]
        predicted_id = np.argmax(output[0])
        return predicted_id


class Preprocessor:
    def __init__(self, input_shape = (224,224)):
        self.input_shape = input_shape
        self.transform = transforms.Compose([
            transforms.Resize(self.input_shape),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def preprocess(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image.numpy()
if __name__ =='__main__':
    pre = Preprocessor()
    onx = OnnxModel('output_model.onnx')
    img = pre.preprocess('n01667114_mud_turtle.jpeg')
    print('OUTPUT ', onx.predict(img))