FROM anibali/pytorch:1.13.0-cuda11.8-ubuntu22.04

# Set up time zone.
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Install system libraries required by OpenCV.
RUN sudo apt-get update \
 && sudo apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
 && sudo rm -rf /var/lib/apt/lists/*

COPY ./ .
WORKDIR .

# Install OpenCV from PyPI.
RUN pip install -r requirements.txt
CMD ["python", "convert_onnx.py"]
CMD ["python", "test_onnx_model.py"]

