import os

lr_choices = [1e-4, 5e-4 ,1e-3, 5e-3,1e-2, 5e-2,1e-1, 5e-1,1]
bs_choices = [50,100,200,500,1000,5000,10000]

for lr in lr_choices:
    for bs in bs_choices:
        os.system('cd code && python3 train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 4 -l 2 -s 64 -b %d -lr %f -rtg --exp_name ip_b_%d_lr_%f'%(bs,lr,bs,lr))

