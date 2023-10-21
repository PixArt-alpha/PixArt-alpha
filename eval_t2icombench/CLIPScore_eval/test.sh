project_dir=CLIPScore_eval

folder=$1
ability=$2
cuda=$3

outpath=$folder/$ability

echo $out_dir
cd $project_dir && CUDA_VISIBLE_DEVICES=$cuda python CLIP_similarity.py --outpath=${out_dir} > tmp.out