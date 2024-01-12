FROM nvcr.io/nvidia/tritonserver:22.11-py3

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt /tmp/
RUN pip3 install --upgrade pip && \
    pip3 install -r /tmp/requirements.txt

ADD https://pixel-go-app.apyhi.com/ismail_weights/retina.onnx /opt/tritonserver/onnx/retina.onnx

COPY . .

RUN chmod +x api_setup.sh
RUN pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

EXPOSE 6270

CMD ["sh", "-c", "./api_setup.sh"]



#CMD ["tritonserver", "--model-repository=models"]


# CUDA 11.8
# CUddnn 8.6.0
# TensorRT v8501
