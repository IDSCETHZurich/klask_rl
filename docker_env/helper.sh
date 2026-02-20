#!/bin/bash

# Klask IsaacLab Docker management script
# Usage: ./helper.sh <command>

set -e

# Get the directory of this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


IMAGE_NAME="isaac-lab-klask-rl"
DOCKERFILE="Dockerfile"
# GPUS="all"
GPUS='"device=1,2,3"'
# UI="LIVESTREAM=1"
UI="HEADLESS=1"

TAG="local"
CONTAINER_NAME="${IMAGE_NAME}_container_${TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  build    Build the Docker image"
    echo "  run      Run the container (detached)"
    echo "  connect  Connect to the running container"
    echo "  stop     Stop the container"
    echo "  status   Show container status"
    echo ""
}

cmd_build() {
    echo -e "${GREEN}Building ${MODE_DISPLAY} Docker image...${NC}"

    docker build -t "${IMAGE_NAME}:${TAG}" -f "$SCRIPT_DIR/$DOCKERFILE" "$SCRIPT_DIR/.."

    echo -e "${GREEN}${MODE_DISPLAY} build complete!${NC}"
}

cmd_run() {
    # Check if container is already running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}Container '${CONTAINER_NAME}' is already running.${NC}"
        echo "Use '$0 connect' to attach to it, or '$0 stop' to stop it first."
        return 1
    fi

    echo -e "${GREEN}Starting container..."

    # xhost +local:root
    docker run -it -d --rm \
        --name="${CONTAINER_NAME}" \
        --gpus=${GPUS} \
        --network=host \
        --env="ACCEPT_EULA=Y" \
        --env="PRIVACY_CONSENT=Y" \
        --env="DISPLAY" \
        --env=${UI} \
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
        "${IMAGE_NAME}:${TAG}"

    echo -e "${GREEN}${MODE_DISPLAY} container started! Use '$0 ${MODE_FLAG}connect' to attach.${NC}"
}

cmd_connect() {
    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${RED}Container '${CONTAINER_NAME}' is not running.${NC}"
        echo "Use '$0 run' to start it first."
        return 1
    fi

    echo -e "${GREEN}Connecting to container..."
    
    WORKDIR="/workspace/klask_rl"

    docker exec -it -w "$WORKDIR" "${CONTAINER_NAME}" bash
}

cmd_stop() {
    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}Container '${CONTAINER_NAME}' is not running."
        return 0
    fi

    echo -e "${GREEN}Stopping container..."

    docker stop "${CONTAINER_NAME}"
    
    echo -e "${GREEN}Container stopped!"
}

cmd_status() {
    echo -e "${GREEN}$Container Status:"
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "  Status: ${GREEN}Running"
        docker ps --filter "name=${CONTAINER_NAME}" --format "  ID: {{.ID}}\n  Created: {{.RunningFor}}\n  Ports: {{.Ports}}"
    else
        echo -e "  Status: ${RED}Not running"
    fi
    
    echo ""
    echo -e "${GREEN}Image:"
    if docker images "${IMAGE_NAME}:${TAG}" --format "{{.Repository}}" | grep -q "${IMAGE_NAME}"; then
        docker images "${IMAGE_NAME}:${TAG}" --format "  Repository: {{.Repository}}:{{.Tag}}\n  ID: {{.ID}}\n  Size: {{.Size}}\n  Created: {{.CreatedSince}}"
    else
        echo -e "  ${RED}Image not found. Run '$0 build' first."
    fi
}

# Main
if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

COMMAND="$1"
shift  # Remove command from arguments

case "$COMMAND" in
    build)
        cmd_build
        ;;
    run)
        cmd_run "$@"
        ;;
    connect)
        cmd_connect
        ;;
    stop)
        cmd_stop
        ;;
    status)
        cmd_status
        ;;
    -h|--help|help)
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
