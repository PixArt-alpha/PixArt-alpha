project_dir=BLIPvqa_eval


folder=$1
ability=$2
cuda=$3
out_dir=../../output/T2ICompBench/junsong_organized_generation/$folder/$ability

echo $out_dir
cd $project_dir && CUDA_VISIBLE_DEVICES=$cuda python BLIP_vqa.py --out_dir=$out_dir > tmp.out