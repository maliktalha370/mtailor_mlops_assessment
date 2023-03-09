# In this file, we define download_model
# It runs during container build time to get model weights built into the container


from model import OnnxModel, Preprocessor
def download_model():
    model = OnnxModel('output_model.onnx')

if __name__ == "__main__":
    download_model()