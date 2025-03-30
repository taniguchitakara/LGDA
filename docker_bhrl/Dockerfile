FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive
ARG UID=12345
ARG GID=12345
RUN groupadd -g $GID dev \
    && useradd -m -u $UID -g $GID dev



# apt-get update や apt-get upgrade の前に
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update -y
RUN apt-get install -y libegl1-mesa-dev
RUN apt-get install -y libglib2.0-0
RUN pip install mmcv-full==1.3.3 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
RUN pip install numpy==1.20
RUN pip install yapf==0.40.1
RUN pip install cython
#dockerでは実行できないが，bhrlフォルダで最後に以下のプログラムを動かす必要がある
#RUN pip install -v -e . 

  
