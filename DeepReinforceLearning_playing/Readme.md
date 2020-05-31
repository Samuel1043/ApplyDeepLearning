# test dqn
python3 test.py --test_dqn --cpt_dir ./

# test pg 
python3 test.py --test_pg --cpt_dir ./

# train dqn (1000000 steps)
python3 main.py --train_dqn --cpt_dir ./

# train dueling DQN, double DQN
python3 main.py --train_dqn --double_dqn True --cpt_dir $cpt_save_dir
python3 main.py --train_dqn --dueling_dqn True --cpt_dir $cpt_save_dir

# test dueling DQN, double DQN(double DQN cpt require)
python3 test.py --test_dqn --cpt_dir $cpt_dir --double_dqn True
python3 test.py --test_dqn --cpt_dir $cpt_dir --double_dqn True

# train pg (till reach baseline)
python3 main.py --train_pg --cpt_dir ./

# train pg baseline
python3 main.py --train_pg --basline_pg True --cpt_dir $cpt_dir

# test pg baeline
python3 test.py --test_pg --baseline_pg True --cpt_dir $cpt_dir


#plot figure
plot_figure.ipynb 

