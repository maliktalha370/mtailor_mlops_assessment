# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``
import json

import banana_dev as banana

import base64
import time


def test_image(image_file):

    with open(image_file, "rb") as f:
        im_bytes = f.read()
    im_b64 = base64.b64encode(im_bytes).decode("utf8")

    model_inputs = json.dumps({'prompt': im_b64})
    API_KEY = 'abeb86b1-b32a-416d-aced-0f991c3ee386'
    MODEL_KEY = 'e41574b2-b5d1-42a1-a85b-1de16f8729c3'
    start_time = time.time()
    output = banana.run(API_KEY, MODEL_KEY, model_inputs)
    end_time = time.time()
    print('Time taken:', end_time - start_time, 'seconds')
    return output['class_id']

def base_test():
    test_images = ['n01440764_tench.jpeg', 'n01667114_mud_turtle.JPEG']
    expected_class_ids = [0, 35]
    pred_class_id = []
    pred_class_name = []
    expected_class_names = ['tench', 'mud turtle']
    for i, image_path in enumerate(test_images):
        with open(image_path, "rb") as f:
            im_bytes = f.read()
        # load the image
        im_b64 = base64.b64encode(im_bytes).decode("utf8")

        model_inputs = json.dumps({'prompt': im_b64})
        API_KEY = 'abeb86b1-b32a-416d-aced-0f991c3ee386'
        MODEL_KEY = 'e41574b2-b5d1-42a1-a85b-1de16f8729c3'

        output = banana.run(API_KEY, MODEL_KEY, model_inputs)


        pred_class_id.append(output)
        if output == 0:
            pred_class_name.append('tench')
        elif output == 35:
            pred_class_name.append('mud turtle')

    if pred_class_id == expected_class_ids:
        print("CLASS ID's same")
    if pred_class_name == expected_class_names:
        print("CLASS names's same")
if __name__ == '__main__':
    import argparse
    # define command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--preset', default='False', type=str,
                        help='path to save the ONNX model')
    args = parser.parse_args()

    image_file = 'n01440764_tench.jpeg'

    test_image(image_file)
    if args.preset:
        base_test()
    else:
        test_image(image_file)

