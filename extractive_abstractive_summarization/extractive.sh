#!/bin/bash
python3.7 src/preprocess_seq_tag.py datasets/seq_tag/  $1
python3.7 datasets/seq_tag/seq_tag_predict.py $1 $2 
