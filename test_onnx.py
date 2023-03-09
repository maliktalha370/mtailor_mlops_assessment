import unittest
import onnxruntime as rt
import numpy as np
import cv2

class TestModel(unittest.TestCase):
    def setUp(self):
        # initialize the ONNX runtime session and load the model
        self.session = rt.InferenceSession('output_model.onnx')

        # define the expected class IDs and names for the test images
        self.expected_class_ids = [0, 35]
        self.expected_class_names = ['tench', 'mud turtle']
        self.test_images = ['n01440764_tench.jpeg', 'n01667114_mud_turtle.JPEG']
    def test_model_prediction(self):
        # load the test images
        pred_class_id = []
        pred_class_name = []
        for i, image_path in enumerate(self.test_images):
        # load the image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # preprocess the image
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0).astype(np.float32)

            # run inference
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            result = self.session.run([output_name], {input_name: img})

            # get the predicted class ID and class name
            predicted_id = np.argmax(result[0])
            pred_class_id.append(predicted_id)
            if predicted_id == 0:
                pred_class_name.append('tench')
            elif predicted_id == 35:
                pred_class_name.append('mud turtle')

            # print the predicted class ID and class name



        # check the predicted class IDs and names against the expected values
        self.assertListEqual(pred_class_id, self.expected_class_ids)
        self.assertListEqual(pred_class_name, self.expected_class_names)
