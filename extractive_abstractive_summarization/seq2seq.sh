#!/bin/bash
python3.7 src/preprocess_seq2seq.py datasets/seq2seq/  $1 1
python3.7 datasets/seq2seq/seq2seq_predict.py $1 $2
