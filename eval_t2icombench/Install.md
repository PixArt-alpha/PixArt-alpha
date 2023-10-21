## Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:


To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

1. Create a virtual environment (with Python 3.9):
```
conda create -n t2icombench python=3.9
conda activate t2icombench
```

2. Install diffusers from source:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

3. Install other provided requirements:
```bash
pip install -r requirements.txt
```

4. Install [🤗Accelerate](https://github.com/huggingface/accelerate/) 
```bash
accelerate config
```

5. Install other dependencies required from the necessary repositories, including [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4), [blip](https://github.com/salesforce/BLIP/tree/main), UniDET(https://github.com/xingyizhou/UniDet), [CLIP](https://github.com/openai/CLIP).


We recommend you refer to the original [T2I-Combench](https://github.com/Karine-Huang/T2I-CompBench/tree/main) repo for more details on the installation process.

