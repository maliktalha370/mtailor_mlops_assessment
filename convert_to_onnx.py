import torch
import argparse
from pytorch_model import BasicBlock, Classifier

# define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default = './resnet18-f37072fd.pth', type=str,
                    help='path to the PyTorch model')
parser.add_argument('--output_path', default = './output_model.onnx', type=str,
                    help='path to save the ONNX model')
args = parser.parse_args()

mtailor = Classifier(BasicBlock, [2, 2, 2, 2])

# load the PyTorch model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtailor.load_state_dict(torch.load(args.model_path,  map_location=device))

# set the model to evaluation mode
mtailor.eval()

# define an example input tensor
example_input = torch.randn(1, 3, 224, 224)

# export the model to the ONNX format
torch.onnx.export(mtailor, example_input, args.output_path,
                  input_names=['input'], output_names=['output'],
                  opset_version=11)

print('ONNX model saved to:', args.output_path)