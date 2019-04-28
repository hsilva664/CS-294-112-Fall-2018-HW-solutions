import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np

"""
Using the plotter:

Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10 
random seeds. The runner code stored it in the directory structure

    data
    L test_EnvName_DateTime
      L  0
        L log.txt
        L params.json
      L  1
        L log.txt
        L params.json
       .
       .
       .
      L  9
        L log.txt
        L params.json

To plot learning curves from the experiment, averaged over all random
seeds, call

    python plot.py data/test_EnvName_DateTime --value AverageReturn

and voila. To see a different statistics, change what you put in for
the keyword --value. You can also enter /multiple/ values, and it will 
make all of them in order.


Suppose you ran two experiments: 'test1' and 'test2'. In 'test2' you tried
a different set of hyperparameters from 'test1', and now you would like 
to compare them -- see their learning curves side-by-side. Just call

    python plot.py data/test1 data/test2

and it will plot them both! They will be given titles in the legend according
to their exp_name parameters. If you want to use custom legend titles, use
the --legend flag and then provide a title for each logdir.

"""

def plot_data(data, val_data):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    if isinstance(val_data, list):
        val_data = pd.concat(val_data, ignore_index=True)        

    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(data=data, time="Iteration", value="AverageReturn", unit="Unit", condition="Condition", color=sns.color_palette("Blues_d"))
    sns.tsplot(data=val_data, time="Iteration", value="ValAverageReturn", unit="Unit", condition="Condition", color=sns.color_palette("Reds"))
    plt.legend(loc='best').draggable()
    plt.show()


def get_datasets(fpath, condition=None):
    unit = 0
    val_unit = 0
    datasets = []
    val_datasets = []
    for root, dir, files in os.walk(fpath):
        if 'log.txt' in files:
            param_path = open(os.path.join(root,'params.json'))
            params = json.load(param_path)
            exp_name = params['exp_name']
            
            log_path = os.path.join(root,'log.txt')
            experiment_data = pd.read_table(log_path)

            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
                )        
            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                condition or exp_name
                )

            val_experiment_data = pd.read_table(log_path)

            val_experiment_data.insert(
                len(val_experiment_data.columns),
                'Unit',
                unit
                )        

            val_name = condition or exp_name
            val_name = val_name + '_val'

            val_experiment_data.insert(
                len(val_experiment_data.columns),
                'Condition',
                val_name
                )                

            datasets.append(experiment_data)
            val_datasets.append(val_experiment_data)
            unit += 1
            val_unit += 1

    return datasets, val_datasets


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    args = parser.parse_args()

    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True

    data = []
    val_data = []
    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            d, vd = get_datasets(logdir, legend_title)
            data += d
            val_data += vd
    else:
        for logdir in args.logdir:
            d, vd = get_datasets(logdir)
            data += d
            val_data += vd


    plot_data(data, val_data)

if __name__ == "__main__":
    main()
