project_dir=3_in_1_eval

folder=$1
ability=$2
cuda=$3

outpath=$folder/$ability
data_path=./examples/dataset

echo $out_dir
cd $project_dir && CUDA_VISIBLE_DEVICES=$cuda python 3_in_1.py --outpath=${outpath} --data_path=${data_path}  > tmp.out