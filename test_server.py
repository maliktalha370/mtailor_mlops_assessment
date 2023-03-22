# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``
import json

import banana_dev as banana

import base64
import time

API_KEY = 'abeb86b1-b32a-416d-aced-0f991c3ee386'
MODEL_KEY = '85919015-d372-4af1-96bd-036f1afbf3fe'

def test_image(image_file):

    with open(image_file, "rb") as f:
        im_bytes = f.read()
    im_b64 = base64.b64encode(im_bytes).decode("utf8")

    model_inputs = json.dumps({'prompt': im_b64})

    output = banana.run(API_KEY, MODEL_KEY, model_inputs)
    return output['modelOutputs'][0]['class_id']
def base_test():
    test_images = ['n01440764_tench.jpeg', 'n01667114_mud_turtle.JPEG']
    pred_class_name = []
    expected_class_names = ['tench', 'mud turtle']
    for i, image_path in enumerate(test_images):
        with open(image_path, "rb") as f:
            im_bytes = f.read()
        # load the image
        im_b64 = base64.b64encode(im_bytes).decode("utf8")

        model_inputs = json.dumps({'prompt': im_b64})

        output = banana.run(API_KEY, MODEL_KEY, model_inputs)
        output  = output['modelOutputs'][0]['class_id']
        pred_class_name.append(output)
    if pred_class_name == expected_class_names:
        print("TEST PASSED !!! Predicted classes matches with GroundTruth.")
if __name__ == '__main__':
    import argparse
    # define command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--preset', action='store_true',
                        help='path to save the ONNX model')
    args = parser.parse_args()

    image_file = 'n01440764_tench.jpeg'
    start_time = time.time()

    if args.preset:
        result = base_test()
    else:
        result = test_image(image_file)
        print(f'Model Detection for {image_file} is {result} ')

    end_time = time.time()
    print('Time taken:', end_time - start_time, 'seconds')
