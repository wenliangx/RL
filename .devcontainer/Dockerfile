FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip
RUN pip install \
    numpy \
    pandas \
    matplotlib \
    scipy