# This is a sample Dockefile that builds a runtime container and runs the sample Gradio app.
# Note, you must pass in the pretrained models when you run the container.

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y \
        git \
        python3 \
        python-is-python3 \
        python3-pip \
        python3.10-venv \
        libgl1 \
        libgl1-mesa-glx \ 
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ADD requirements.txt .

RUN pip install -r requirements.txt

ADD . .

RUN chmod a+x docker-entrypoint.sh

ENV DEMO_PORT=12345
ENTRYPOINT [ "/workspace/docker-entrypoint.sh" ]