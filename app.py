
import io
from model import OnnxModel, Preprocessor
import base64
from flask import request
from PIL import Image

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    # HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    #
    # # this will substitute the default PNDM scheduler for K-LMS
    # lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    #
    # model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=lms,
    #                                                 use_auth_token=HF_AUTH_TOKEN).to("cuda")
    model = OnnxModel('output_model.onnx')


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model
    pre = Preprocessor()

    # # Parse out your arguments

    # height = model_inputs.get('height', 512)
    # width = model_inputs.get('width', 512)
    # num_inference_steps = model_inputs.get('num_inference_steps', 50)
    # guidance_scale = model_inputs.get('guidance_scale', 7.5)
    # input_seed = model_inputs.get("seed", None)
    #
    # # If "seed" is not sent, we won't specify a seed in the call
    # generator = None
    # if input_seed != None:
    #     generator = torch.Generator("cuda").manual_seed(input_seed)
    #
    # if prompt == None:
    #     return {'message': "No prompt provided"}
    #
    # # Run the model
    # with autocast("cuda"):
    #     image = model(prompt, height=height, width=width, num_inference_steps=num_inference_steps,
    #                   guidance_scale=guidance_scale, generator=generator)["sample"][0]
    #
    # buffered = BytesIO()
    # image.save(buffered, format="JPEG")
    # image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    #
    im_b64 = model_inputs.get('prompt', None)
    # convert it into bytes
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    img = pre.preprocess(io.BytesIO(img_bytes))
    class_id = model.predict(img)
    # Return the results as a dictionary

    return {'class_id': class_id}