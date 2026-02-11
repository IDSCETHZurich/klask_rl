CONTAINER_NAME="isaac-lab:local"

# Get the directory of this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# xhost +
docker run -it --rm \
   --name="${CONTAINER_NAME}" \
   --entrypoint bash \
   --gpus=all \
   --network=host \
   --env="ACCEPT_EULA=Y" \
   --env="PRIVACY_CONSENT=Y" \
   --env="DISPLAY" \
   --env="LIVESTREAM=1" \
   --env="PUBLIC_IP=100.121.89.49" \
   --env-file="${SCRIPT_DIR}/../.devcontainer/devcontainer.env" \
   --volume="$HOME/.Xauthority:/root/.Xauthority" \
   --volume="${CONTAINER_NAME}_cache_kit:/isaac-sim/kit/cache:rw" \
   --volume="${CONTAINER_NAME}_cache_ov:/root/.cache/ov:rw" \
   --volume="${CONTAINER_NAME}_cache_pip:/root/.cache/pip:rw" \
   --volume="${CONTAINER_NAME}_cache_glcache:/root/.cache/nvidia/GLCache:rw" \
   --volume="${CONTAINER_NAME}_cache_computecache:/root/.nv/ComputeCache:rw" \
   --volume="${CONTAINER_NAME}_logs:/root/.nvidia-omniverse/logs:rw" \
   --volume="${CONTAINER_NAME}_data:/root/.local/share/ov/data:rw" \
   --volume="${CONTAINER_NAME}_documents:/root/Documents:rw" \
   --volume="$SCRIPT_DIR/../src/klask_rl:/workspace/klask_rl:rw" \
   --volume="$SCRIPT_DIR/../third_party/dreamerv3-torch:/workspace/dreamerv3-torch:rw" \
   nvcr.io/nvidia/isaac-lab:2.3.1