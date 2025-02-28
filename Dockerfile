FROM pinto0309/ubuntu22.04-cuda11.8-tensorrt8.5.3:latest

ENV DEBIAN_FRONTEND=noninteractive

# RUN sudo apt-get update \
#     && sudo apt-get install -y --no-install-recommends \
#         nano \
#         python3-pip \
#         python3-mock \
#         libpython3-dev \
#         libpython3-all-dev \
#         python-is-python3 \
#         wget \
#         curl \
#         cmake \
#         software-properties-common \
#         sudo \
#         git \
#         build-essential \
#         lsb-release \
#         less \
#         zstd \
#         libopencv-dev \
#         libgtk2.0-dev \
#         pkg-config \
#         libavcodec-dev \
#         libavformat-dev \
#         libswscale-dev \
#         libturbojpeg-dev \
#         freeglut3-dev \
#     && sudo sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \
#     && sudo apt clean \
#     && sudo rm -rf /var/lib/apt/lists/*

# RUN pip install pip -U \
#     && pip install \
#         onnx==1.15.0 \
#         onnxsim==0.4.33 \
#         https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/releases/download/onnx/onnxruntime_gpu-1.16.1-cp310-cp310-linux_x86_64.whl \
#         sit4onnx==1.0.7 \
#         opencv-contrib-python==4.9.0.80 \
#         numpy==1.24.3 \
#         scipy==1.10.1 \
#         requests==2.31.0 \
#         bbalg==1.0.2 \
#         PyOpenGL==3.1.7 \
#         PyOpenGL_accelerate==3.1.7
