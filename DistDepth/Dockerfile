FROM ubuntu:20.04 as distdepth_image
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install python3 -y
RUN apt-get install python3-pip -y

RUN pip3 install --upgrade pip
RUN pip3 install torch==2.2.1
RUN pip3 install torchvision==0.17
RUN pip3 install opencv-python==4.9.0.80
RUN pip3 install matplotlib==3.7.5
RUN pip3 install imageio==2.34.1
RUN pip3 install Pillow==9.5.0
RUN pip3 install augly==1.0.0
RUN pip3 install tensorboardX==2.6.2.2
RUN apt-get install python3-opencv -y
RUN pip3 install kornia==0.7.2
RUN pip3 install h5py==3.11.0
RUN pip3 install scikit-image==0.21.0
RUN pip3 install timm==0.9.16

WORKDIR /app