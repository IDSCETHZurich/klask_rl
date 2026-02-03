# This is currently just a collection of commands and is not yet an executable script.

docker build -t isaacray -f Ray.Dockerfile .



docker run -it \
   --gpus all \
   --net=host \
   --entrypoint /bin/bash \
   isaacray