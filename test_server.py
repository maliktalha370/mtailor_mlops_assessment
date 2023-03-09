import argparse
import requests
import time
from PIL import Image


def predict_image(image_path, server_url):
    # Open image
    image = Image.open(image_path)

    # Prepare data
    files = {"image": open(image_path, "rb")}
    data = {}

    # Send POST request
    response = requests.post(server_url, data=data, files=files)

    # Check response
    if response.status_code == 200:
        return response.json()['class_name']
    else:
        return None


def run_custom_tests(server_url):
    # Test image paths
    test_image_paths = ['test_images/cat.jpg', 'test_images/dog.jpg']

    # Expected class names
    expected_class_names = ['Egyptian_cat', 'Pembroke']

    # Run tests
    for i, image_path in enumerate(test_image_paths):
        expected_class_name = expected_class_names[i]
        start_time = time.time()
        class_name = predict_image(image_path, server_url)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Test image {i + 1}: {image_path}")
        print(f"Expected class name: {expected_class_name}")
        print(f"Predicted class name: {class_name}")
        print(f"Elapsed time: {elapsed_time:.3f} seconds")
        print()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='Path to image file')
    parser.add_argument('--server_url', type=str, default='http://localhost:8000/predict',
                        help='URL of the deployed model server')
    parser.add_argument('--run_custom_tests', action='store_true', help='Run preset custom tests')
    args = parser.parse_args()

    if args.run_custom_tests:
        run_custom_tests(args.server_url)
    elif args.image_path:
        # Predict single image
        class_name = predict_image(args.image_path, args.server_url)
        print(f"Class name: {class_name}")
    else:
        print("Please provide either an image path or the --run_custom_tests flag.")
