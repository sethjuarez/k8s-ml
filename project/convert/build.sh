IMAGE=tlaloc.azurecr.io/kubeflow/convert
docker build -t $IMAGE . && docker run -it --privileged --env-file blob.env $IMAGE