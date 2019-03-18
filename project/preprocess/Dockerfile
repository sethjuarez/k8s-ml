FROM tensorflow/tensorflow:latest-py3

# wget
RUN apt-get update \
    && apt-get install -y apt-utils \
    && apt-get install -y wget \
    && apt-get install -y apt-transport-https

# for mounting blob storage as a drive
RUN wget https://packages.microsoft.com/config/ubuntu/16.04/packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && apt-get update

RUN apt-get install -y blobfuse fuse

# Clean up APT when done.
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && rm /packages-microsoft-prod.deb

# pip install
COPY requirements.txt /scripts/requirements.txt
RUN pip install -r /scripts/requirements.txt

# paths
ARG BASE_PATH=/mnt/blob
ARG TEMP_PATH=/mnt/blobfusetmp
RUN mkdir -p $TEMP_PATH && mkdir -p $BASE_PATH
RUN chown -R root $TEMP_PATH
ENV BASE_PATH=${BASE_PATH}
ENV TEMP_PATH=${TEMP_PATH}

COPY data.py /scripts/data.py

#ENTRYPOINT [ "bash" ]
ENTRYPOINT [ "python", "/scripts/data.py", "-d", "data/PetImages", "-t", "dataset.txt" ]
