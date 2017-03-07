FROM nvidia/cuda:7.5-cudnn4-devel

RUN apt-get -y update && \
    apt-get -y install python build-essential python-software-properties software-properties-common ipython python-pip python-scipy python-dev vim gdal-bin python-gdal libgdal-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install Theano Keras h5py

ADD ./bin /
COPY .theanorc /root/.theanorc
COPY keras.json /root/.keras/keras.json
