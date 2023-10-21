## 🤗 Acknowledgement
We primarily assess the compositionality of generated images using [T2I-Combench](https://arxiv.org/abs/2307.06350).
This evaluation code is built using the [official t2i-combench repo](https://github.com/Karine-Huang/T2I-CompBench/tree/main).

## ⚙️ Usage
### 1. Installation
Please refer to [Install.md](Install.md) for installation instructions.

### 2. Clone the repo
clone the repo and move to the directory
```bash
git clone https://github.com/Karine-Huang/T2I-CompBench.git
mv T2I-CompBench/* eval_t2icombench/
cd eval_t2icombench/
```
## 📝 Data Preparation
Before evaluation, you need to first gerate data covering different dimensions for evaluation.
The specific prompts for image generation are listed as follows:

- color [prompts](https://github.com/Karine-Huang/T2I-CompBench/blob/main/examples/dataset/color_val.txt)
- shaoe [prompts](https://github.com/Karine-Huang/T2I-CompBench/blob/main/examples/dataset/shape_val.txt)
- texture [prompts](https://github.com/Karine-Huang/T2I-CompBench/blob/main/examples/dataset/texture_val.txt)
- spatial [prompts](https://github.com/Karine-Huang/T2I-CompBench/blob/main/examples/dataset/spatial_val.txt)
- non-spatial [prompts](https://github.com/Karine-Huang/T2I-CompBench/blob/main/examples/dataset/action_val.txt)
- complex [prompts](https://github.com/Karine-Huang/T2I-CompBench/blob/main/examples/dataset/complex_val.txt)

The generated images are stored in the "examples" directory.
The directory structure is:

```
examples
├──samples/
│  ├──action/
│  │  ├── xxx.png
│  │  ├── xxx.png
│  │  ├── ......
│  ├──color/
│  │  ├── xxx.png
│  │  ├── xxx.png
│  │  ├── ......
│  ├──complex/
│  │  ├── xxx.png
│  │  ├── xxx.png
│  │  ├── ......
│  ├──shape/
│  │  ├── xxx.png
│  │  ├── xxx.png
│  │  │── ......
│  ├──spatial/
│  │  ├── xxx.png
│  │  ├── xxx.png
│  │  ├── ......
│  ├──texture/
│  │  ├── xxx.png
│  │  ├── xxx.png
│  │  ├── ......
```
where the `action` is specially designed for the `non-spatial` dimension.


## ⚖ Evaluation
### 1. Evaluate the performance in one click.
We provide a script to evaluate the performance of all the dimensions in one click. 
```bash
bash auto_eval.sh
```

If you would like to run the evaluation for a specific dimension, please refer to the following steps.
### 2.1 BLIP-VQA:
```
export project_dir="BLIPvqa_eval/"
cd $project_dir
out_dir="examples/"
python BLIP_vqa.py --out_dir=$out_dir
```
or run
```
cd T2I-CompBench
bash BLIPvqa_eval/test.sh
```
The output files are formatted as a json file named `vqa_result.json` in `examples/annotation_blip/` directory.

### 2.2 UniDet:

download weight and put under repo experts/expert_weights:
```
mkdir -p UniDet_eval/experts/expert_weights
cd UniDet_eval/experts/expert_weights
wget https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/Unified_learned_OCIM_RS200_6x%2B2x.pth
```

run evaluation
```
export project_dir=UniDet_eval
cd $project_dir
python determine_position_for_eval.py
```
The output files are formatted as a json file named `vqa_result.json` in `examples/labels/annotation_obj_detection` directory.

### 2.3 CLIPScore:
```
outpath="examples/"
python CLIPScore_eval/CLIP_similarity.py --outpath=${outpath}
```
or run
```
cd T2I-CompBench
bash CLIPScore_eval/test.sh
```
The output files are formatted as a json file named `vqa_result.json` in `examples/annotation_clip` directory.

### 2.4 3-in-1 score:
```
export project_dir="3_in_1_eval/"
cd $project_dir
outpath="examples/"
data_path="examples/dataset/"
python 3_in_1.py --outpath=${outpath} --data_path=${data_path}
```
The output files are formatted as a json file named `vqa_result.json` in `examples/annotation_3_in_1` directory.