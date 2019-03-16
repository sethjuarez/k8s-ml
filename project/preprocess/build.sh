IMAGE=tlaloc.azurecr.io/kubeflow/preprocess
docker build -t $IMAGE . && docker run -it --privileged --env-file blob.env $IMAGE