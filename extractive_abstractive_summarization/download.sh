gdrive download 1xIU1EoUf0P5z0tRmcVH5zz7APwFZ4BWe --recursive
mv data/model30.pth datasets/seq_tag/state_dict
mv data/model_decoder7.pth datasets/seq2seq/state_dict/
mv data/model_encoder7.pth datasets/seq2seq/state_dict/
mv data/model_decoder13.pth datasets/seq2seq/state_dict_attn/
mv data/model_encoder13.pth datasets/seq2seq/state_dict_attn/
