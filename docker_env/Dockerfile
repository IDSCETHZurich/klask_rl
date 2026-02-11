# This is a slightly modified version of the Isaac Lab Dockerfile with Ray and tuning libraries installed that can be found at https://github.com/isaac-sim/IsaacLab/blob/main/scripts/reinforcement_learning/ray/cluster_configs/Dockerfile 

FROM nvcr.io/nvidia/isaac-lab:2.3.1
ARG WS=/workspace/klask_rl

# WGet is needed so that GCS or other cloud providers can mark the container as ready.
# Otherwise the Ray liveliness checks fail.
RUN apt-get update && apt-get install -y \
    wget \
    && \
    rm -rf /var/lib/apt/lists/*

# Set NVIDIA paths
ENV PATH="/usr/local/nvidia/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64"

# Link NVIDIA binaries
RUN ln -sf /usr/local/nvidia/bin/nvidia* /usr/bin

# Install requirements for tuning and logging
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install --upgrade pip
COPY docker_env/res/requirements.txt /tmp/requirements.txt
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install -r /tmp/requirements.txt

# Install Ray and configure it
RUN sed -i "1i $(echo "#!/workspace/isaaclab/_isaac_sim/python.sh")" \
    /isaac-sim/kit/python/bin/ray && ln -s /isaac-sim/kit/python/bin/ray /usr/local/bin/ray

# Copy and set up entrypoint script
COPY docker_env/res/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR $WS

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]