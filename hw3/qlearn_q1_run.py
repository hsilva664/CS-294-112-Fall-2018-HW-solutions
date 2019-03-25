import os

os.system('cd code && python3 run_dqn_atari.py Qlearn_Q1_atari --n_processes 3 --visible_gpus 0')
os.system('cd code && python3 plot_from_pkl.py Qlearn_Q1_atari')

os.system('cd code && python3 run_dqn_ram.py Qlearn_Q1_ram --n_processes 3 --visible_gpus 0')
os.system('cd code && python3 plot_from_pkl.py Qlearn_Q1_ram')

os.system('cd code && python3 run_dqn_lander.py Qlearn_Q1_lander --n_processes 3 --visible_gpus 0')
os.system('cd code && python3 plot_from_pkl.py Qlearn_Q1_lander')
