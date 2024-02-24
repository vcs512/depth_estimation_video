FROM ubuntu:18.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
RUN apt-get install python-opencv -y
RUN mkdir app
RUN cd /app

RUN pip3 install --upgrade pip
RUN pip3 install torch==1.10.2
RUN pip3 install torchvision==0.2.1
RUN pip3 install tensorboardX==1.4
RUN pip3 install tensorboard==2.10.1
RUN pip3 install opencv-python==3.3.1.11
RUN pip3 install matplotlib==3.3.4
RUN pip3 install scikit-image==0.17.2

RUN apt-get update
RUN apt-get install unzip -y

RUN pip3 install pandas==1.1.5
