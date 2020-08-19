## Requirements
```
transformers
tqdm
torch
numpy
ckiptagger
tensorflow==1.15.0
```

## train model
training will save checkpoint and performance in ./save directory 
```
mkdir ./save

python3 train_model.py --train_data $train_json --dev_data $dev_json --max_answer_length 30 --train_epoch 5 --ckip_dir $ckip_model_dir --batch_size 4 --lr 2e-5
```


## predict  
you can set the answerable threshold and max answer length for post processing
```
python3 evaluate_model.py --output_file $prediction_json --predict_file $dev_json --pretrain_checkpoint $check_point_file --max_answer_length 30 --answerable_threshold 0.5 --batch_size 16
```
## evaluate prediction performance 
using ta's evalution script
```
python3 evaluate.py $dev_json $prediction_json $ckip_model_dir
```

## performance 
* EM 
prediction and answer are the same 
* F1
    * True positive (tp) = number of words in prediction that are also in answer.
    * Precision = tp / len(prediction); recall = tp / len(answer)
    * F1 = 2 * precision * recall / (precision + recall)

| Method | EM | F1 | acc | 
| --- | --------- | ------ | ------- | 
| overall      | 0.842 | 0.876 ||
| answerable      | 0.787 | 0.836 ||
| unanswerable   | 0.970 |  0.970 ||
| answerable accuracy |||0.951|


## plotting figures

### answer length plot 
```
cd plot 

python3 ans_len_plot.py $train_json  $save_fig_file
```
### threshold performance
```
python3 threshold_score.py $dir_for_all_threshold_performace  $save_fig_file
```
