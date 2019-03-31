import os

os.system('cd code && python3 run_dqn_atari.py Qlearn_atari_double_rbuf_1e4 --n_processes 3 --visible_gpus 1 --double_q --replay_buffer_size 10000')
os.system('cd code && python3 run_dqn_atari.py Qlearn_atari_double_rbuf_1e5 --n_processes 3 --visible_gpus 1 --double_q --replay_buffer_size 100000')
os.system('cd code && python3 run_dqn_atari.py Qlearn_atari_double_rbuf_1e6 --n_processes 1 --visible_gpus 1 --double_q --replay_buffer_size 1000000')
os.system('cd code && python3 run_dqn_atari.py Qlearn_atari_double_rbuf_1e6 --n_processes 1 --visible_gpus 1 --double_q --replay_buffer_size 1000000')
os.system('cd code && python3 run_dqn_atari.py Qlearn_atari_double_rbuf_1e6 --n_processes 1 --visible_gpus 1 --double_q --replay_buffer_size 1000000')
os.system('cd code && python3 plot_multiple_from_pkl.py Qlearn_Q3_atari_multi Qlearn_atari_double_rbuf_1e4 Qlearn_atari_double_rbuf_1e5 Qlearn_atari_double_rbuf_3e5 Qlearn_atari_double_rbuf_1e6')

os.system('cd code && python3 run_dqn_ram.py Qlearn_ram_double_rbuf_1e4 --n_processes 2 --visible_gpus 1 --double_q --replay_buffer_size 10000')
os.system('cd code && python3 run_dqn_ram.py Qlearn_ram_double_rbuf_1e5 --n_processes 3 --visible_gpus 1 --double_q --replay_buffer_size 100000')
os.system('cd code && python3 run_dqn_ram.py Qlearn_ram_double_rbuf_3e5 --n_processes 3 --visible_gpus 1 --double_q --replay_buffer_size 300000')
os.system('cd code && python3 plot_multiple_from_pkl.py  Qlearn_Q3_ram_multi Qlearn_ram_double_rbuf_1e4 Qlearn_ram_double_rbuf_1e5 Qlearn_ram_double_rbuf_3e5 Qlearn_ram_double_rbuf_1e6')

os.system('cd code && python3 run_dqn_lander.py Qlearn_lander_double_rbuf_1e4 --n_processes 2 --visible_gpus 1 --double_q --replay_buffer_size 10000')
os.system('cd code && python3 run_dqn_lander.py Qlearn_lander_double_rbuf_1e5 --n_processes 3 --visible_gpus 1 --double_q --replay_buffer_size 100000')
os.system('cd code && python3 run_dqn_lander.py Qlearn_lander_double_rbuf_1e6 --n_processes 3 --visible_gpus 1 --double_q --replay_buffer_size 1000000')
os.system('cd code && python3 plot_multiple_from_pkl.py  Qlearn_Q3_lander_multi Qlearn_lander_double_rbuf_1e4 Qlearn_lander_double_rbuf_5e4 Qlearn_lander_double_rbuf_1e5 Qlearn_lander_double_rbuf_1e6')
