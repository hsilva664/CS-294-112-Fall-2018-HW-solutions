import os

lr_choices = [0.005,0.01,0.02]
bs_choices = [10000, 30000, 50000]

for lr in lr_choices:
    for bs in bs_choices:
        os.system("cd code && python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b %d -lr %f -rtg --nn_baseline --exp_name hc_b%d_r%f"%(bs,lr,bs,lr))