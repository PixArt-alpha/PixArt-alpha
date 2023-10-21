# make sure you have at least 6 gpus available

BASE_FOLDER=$1  # denotes the folder name of the generated images e.g., examples/

Organized_data_FLAG=false
CLIP_FLAG=true
BVQA_FLAG=true
UNIDET_FLAG=true
Three_FLAG=true
SHOW_FLAG=true

if $Organized_data_FLAG; then
  python organize_data.py --tgt_file $BASE_FOLDER
fi
echo runing in the folder $BASE_FOLDER


######### evaluate CLIP scores
if $CLIP_FLAG; then
  echo runing clip....
  bash CLIPScore_eval/test.sh $BASE_FOLDER color 1 &
  bash CLIPScore_eval/test.sh $BASE_FOLDER shape 2 &
  bash CLIPScore_eval/test.sh $BASE_FOLDER texture 3 &
  bash CLIPScore_eval/test.sh $BASE_FOLDER spatial 4 &
  bash CLIPScore_eval/test.sh $BASE_FOLDER non-spatial 5 &
  bash CLIPScore_eval/test.sh $BASE_FOLDER complex 6
  echo done clip
fi


##########  evaluate BVQA scores
if $BVQA_FLAG; then
  echo runing b_vqa....
  bash BLIPvqa_eval/test.sh $BASE_FOLDER color 1 &
  bash BLIPvqa_eval/test.sh $BASE_FOLDER shape 2 &
  bash BLIPvqa_eval/test.sh $BASE_FOLDER texture 3 &
  bash BLIPvqa_eval/test.sh $BASE_FOLDER complex 4
  echo done b_vqa
fi


########## evaluate  UNIDET scores
if $UNIDET_FLAG; then
  echo runing unidet....
  bash UniDet_eval/test.sh $BASE_FOLDER spatial 1 &
  bash UniDet_eval/test.sh $BASE_FOLDER complex 2
  echo done unidet
fi


########## evaluate 3-in-1 scores
if $Three_FLAG; then
  echo runing 3-in-1....
  bash 3_in_1_eval/test.sh $BASE_FOLDER complex 1
  echo done 3_in_1
fi


########## organize the results and show
if $SHOW_FLAG; then
  cd $BASE_FOLDER/result &&
  for file in *.txt; do
      # Use regex to extract the desired information
      if [[ $file =~ ([^_]*)_([^_]*)_([0-9]*\.[0-9]*)([^.]*) ]]; then
          prefix="${BASH_REMATCH[1]}"
          attribute="${BASH_REMATCH[2]}"
          value="${BASH_REMATCH[3]}"
          extra="${BASH_REMATCH[4]}"
          echo "${prefix}, ${attribute}: ${value}${extra}"
      fi
  done
fi
