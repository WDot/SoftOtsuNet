FROM tensorflow/tensorflow:latest

RUN apt-get update
RUN apt -y install ssh
RUN apt -y install libopenmpi-dev
RUN apt -y install cmake
#RUN pip3 install tensorflow_addons
RUN pip3 install torch
RUN pip3 install torchvision
RUN pip3 install sklearn
RUN pip3 install pandas
RUN pip3 install timm
RUN pip3 install torch_optimizer
RUN pip3 install onnx onnx_tf onnxruntime coremltools
RUN pip3 install onnxsim
RUN pip3 install tensorflow_probability
RUN pip3 install scikit-learn
RUN pip3 install networkx==2.5.1
RUN pip3 install scikit-image