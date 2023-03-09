# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64
import json


with open('n01667114_mud_turtle.JPEG', "rb") as f:
    im_bytes = f.read()
im_b64 = base64.b64encode(im_bytes).decode("utf8")

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

model_inputs = json.dumps({"prompt": im_b64})
# model_inputs = {'prompt': open(', 'rb')}
res = requests.post('http://localhost:8000/', json = model_inputs, headers=headers)

return_output = res.json()["class_id"]

print(return_output)
