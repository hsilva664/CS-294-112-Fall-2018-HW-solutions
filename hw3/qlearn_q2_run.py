import os

os.system('cd code && python3 run_dqn_atari.py Qlearn_Q2_atari --n_processes 3 --visible_gpus 1 --double_q')
os.system('cd code && python3 plot_from_pkl.py Qlearn_Q2_atari')

os.system('cd code && python3 run_dqn_ram.py Qlearn_Q2_ram --n_processes 3 --visible_gpus 1 --double_q')
os.system('cd code && python3 plot_from_pkl.py Qlearn_Q2_ram')

os.system('cd code && python3 run_dqn_lander.py Qlearn_Q2_lander --n_processes 3 --visible_gpus 1 --double_q')
os.system('cd code && python3 plot_from_pkl.py Qlearn_Q2_lander')
