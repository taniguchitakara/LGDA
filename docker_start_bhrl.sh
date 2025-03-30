#!/bin/bash

# Get the current directory
CURRENT_DIR=$(pwd)

# Run the Docker container with the current directory mounted
docker run -it --shm-size=1g --rm --gpus all -v "${CURRENT_DIR}:${CURRENT_DIR}" --name bhrl bhrl:latest /bin/bash