#!/bin/bash

# Install packages in editable mode
echo "===================================="
echo "Installing klask_rl package..."
echo "===================================="
/workspace/isaaclab/_isaac_sim/python.sh -m pip install --root-user-action=ignore -e /workspace/klask_rl/source/klask_rl
if [ $? -eq 0 ]; then
    echo "✓ klask_rl package installed successfully"
else
    echo "✗ Failed to install klask_rl package"
fi

# Start Ray server in the background
echo "===================================="
echo "Starting Ray server..."
echo "===================================="
cd /workspace/isaaclab
nohup bash -c "echo 'import ray; ray.init(); import time; [time.sleep(10) for _ in iter(int, 1)]' | ./isaaclab.sh -p" > /tmp/ray_server.log 2>&1 &
cd /workspace/klask_rl

# If arguments are provided, execute them; otherwise just exit
if [ $# -gt 0 ]; then
    exec "$@"
fi
