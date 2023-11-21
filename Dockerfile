# This is a sample Dockefile that builds a runtime container and runs the sample Gradio app.
# Note, you must pass in the pretrained models when you run the container.

FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

WORKDIR /workspace

ADD . .

RUN apt-get update && \
    apt-get install -y \
        git \
        python3 \
        python-is-python3 \
        python3-pip \
        libgl1 \
        libgl1-mesa-glx \ 
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Who needs conda?

RUN pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117 && \
    pip install -r requirements.txt

CMD "python scripts/interface.py --port=12345"

# Build with
# docker build . -t pixart

# Run with 
# docker run --gpus all -it -p 12345:12345 -v <path_to_models>:/workspace/output/pretrained_models pixart