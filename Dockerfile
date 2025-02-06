FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN apt-get update && apt-get install -y --no-install-recommends wget build-essential libreadline-dev \ 
    libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev liblzma-dev zlib1g-dev uuid-dev libffi-dev libdb-dev \
    libglib2.0-0 libgl1-mesa-glx git ffmpeg unzip

RUN wget --no-check-certificate https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz \
    && tar -xf Python-3.10.13.tgz \
    && cd Python-3.10.13 \
    && ./configure --enable-optimizations\
    && make \
    && make install

RUN apt-get autoremove -y

WORKDIR /kaggle
ENV PYTHONPATH=/kaggle

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r ./requirements.txt