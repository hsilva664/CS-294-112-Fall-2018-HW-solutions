import os

os.system('cd code && python3 train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 6 -l 1 -s 32 -lr 5e-3 -dna --exp_name sb_no_rtg_dna')
os.system('cd code && python3 train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 6 -l 1 -s 32 -lr 5e-3 -rtg -dna --exp_name sb_rtg_dna')
os.system('cd code && python3 train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 6 -l 1 -s 32 -lr 5e-3 -rtg --exp_name sb_rtg_na')
os.system('cd code && python3 train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 6 -l 1 -s 32 -lr 5e-3 -dna --exp_name lb_no_rtg_dna')
os.system('cd code && python3 train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 6 -l 1 -s 32 -lr 5e-3 -rtg -dna --exp_name lb_rtg_dna')
os.system('cd code && python3 train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 6 -l 1 -s 32 -lr 5e-3 -rtg --exp_name lb_rtg_na')
