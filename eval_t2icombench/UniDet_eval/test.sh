project_dir=UniDet_eval

folder=$1
ability=$2
cuda=$3

outpath=$folder/$ability

cd $project_dir && CUDA_VISIBLE_DEVICES=$cuda python determine_position_for_eval.py --outpath $out_dir > tmp.out

