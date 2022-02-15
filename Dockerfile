FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
# RUN pip install tqdm
RUN pip install matplotlib
RUN pip install pytorch_lightning