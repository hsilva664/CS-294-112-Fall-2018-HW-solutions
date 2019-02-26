import os

lr = 0.02
bs = 50000

os.system("cd code && python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b %d -lr %f --exp_name hc_b%d_r%f"%(bs,lr,bs,lr))
os.system("cd code && python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b %d -lr %f -rtg --exp_name hc_rtg_b%d_r%f"%(bs,lr,bs,lr))
os.system("cd code && python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b %d -lr %f --nn_baseline --exp_name hc_baseline_b%d_r%f"%(bs,lr,bs,lr))
os.system("cd code && python3 train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b %d -lr %f -rtg --nn_baseline --exp_name hc_rtg_baseline_b%d_r%f"%(bs,lr,bs,lr))