FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git build-essential tmux asciinema \
    ca-certificates curl ripgrep procps && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/local/bin/python
RUN python3 -m pip install --upgrade pip && python3 -m pip install torch==2.9.0
RUN python3 -m pip install git+https://github.com/ScalingIntelligence/KernelBench.git@423217d9fda91e0c2d67e4a43bf62f96f6d104f1
COPY . /opt/task
RUN python3 -m pip freeze > /opt/task/pip-freeze.txt
WORKDIR /workspace
