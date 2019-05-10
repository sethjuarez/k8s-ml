FROM tensorflow/tensorflow:2.0.0a0-py3
#FROM onnx/onnx-ecosystem

# pip install
# pip install
COPY requirements.txt /scripts/requirements.txt
RUN pip install -r /scripts/requirements.txt

COPY model scripts/model
COPY convert.py /scripts/convert.py

# python convert.py -m model/latest.h5 -t model/latest.onnx
ENTRYPOINT [ "python", \
           "/scripts/convert.py", \
           "-m", "/scripts/model/latest.h5", \
           "-t", "/scripts/model/latest.onnx" ]

#ENTRYPOINT [ "bash" ]
