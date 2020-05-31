#!bin/bash
check_point_file=./data/model_epoch4_lr2e-5.pth
python3.7 evaluate_model.py --output_file $2 --predict_file $1 --pretrain_checkpoint $check_point_file --max_answer_length 30 --answerable_threshold 0.5 --batch_size 16
