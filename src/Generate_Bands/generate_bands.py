from __future__ import division
from os import listdir
import pandas as pd
import numpy as np
from random import shuffle
from sklearn.tree import DecisionTreeRegressor
import sys
import matplotlib.lines as lines
from matplotlib.lines import Line2D

"""
This code is used to generate the figure 1 of the paper
"""




class solution_holder:
    def __init__(self, id, decisions, objective):
        self.id = id
        self.decision = decisions
        self.objective = objective


def split_data(filename):
    pdcontent = pd.read_csv(filename)
    indepcolumns = [col for col in pdcontent.columns if "$<" not in col]
    depcolumns = [col for col in pdcontent.columns if "$<" in col]
    sortpdcontent = pdcontent.sort(depcolumns[-1])
    content = list()
    for c in xrange(len(sortpdcontent)):
        content.append(solution_holder(
                                       c,
                                       sortpdcontent.iloc[c][indepcolumns].tolist(),
                                       sortpdcontent.iloc[c][depcolumns].tolist()
                                       )
                       )

    shuffle(content)
    indexes = range(len(content))
    train_indexes, test_indexes = indexes[:int(0.3*len(indexes))], indexes[int(.3*len(indexes)):]
    assert(len(train_indexes) + len(test_indexes) == len(indexes)), "Something is wrong"
    train_set = [content[i] for i in train_indexes]
    test_set = [content[i] for i in test_indexes]

    return [train_set, test_set]


def get_mmre(train, test):
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective[-1] for t in train]

    test_independent = [t.decision for t in test]
    test_dependent = [t.objective[-1] for t in test]

    model = DecisionTreeRegressor()
    model.fit(train_independent, train_dependent)
    predicted = model.predict(test_independent)

    mre = []
    for org, pred in zip(test_dependent, predicted):
        if org != 0:
            mre.append(abs(org - pred)/ abs(org))
    return round(np.mean(mre)* 100, 2)


def get_data(data_folder):
    files = [data_folder + file for file in listdir(data_folder)]
    result = []
    for file in files:
        print file , "  ",
        mres = []
        for _ in xrange(20):
            print ". ",
            sys.stdout.flush()
            train, test = split_data(file)
            mres.append(get_mmre(train, test))
        result.append([file, [np.mean(mres), np.std(mres)], len(train) + len(test)])
        print
    return result


def draw_fig_aritifical_models(pickle_file):
    def modify_name(name):
        names = {
            'fifty': '50',
            'no': '0',
            'one': '100',
            'twentyfive': '25',
            'two': '200'
        }
        filename = name.split('/')[-1]
        ss_name = filename.split('_')[-1].split('.')[0]
        first_part = filename.split('_')[0]
        for key in names.keys():
            if key in first_part:
                return names[key] + '-' + ss_name

    import pickle
    data = pickle.load(open(pickle_file, 'r'))
    import matplotlib.pyplot as plt
    data = sorted(data, key=lambda x: x[1])
    for d in data:
        print d
    # projects = ["SS"+str(i+1) for i,d in enumerate(data)]
    projects = [modify_name(d[0]) for d in data]
    y_pos = [i*10 for i in np.arange(len(projects))]
    performance = [d[1][0] for d in data]

    plt.plot([-5, 255], [5, 5], 'k-', lw=2, color='black')
    plt.plot([-5, 255], [10, 10], '--', lw=2, color='black')
    plt.plot([-5, 255], [100, 100], '-.', lw=2, color='black')
    # plt.bar(y_pos, performance, align='center', alpha=0.5, width=8, color='green', label='< 5%', hatch='o', log=True)

    plt.bar(y_pos[:6], performance[:6], align='center', alpha=0.5, width=8, color='green', label='< 5%', hatch='o', log=True)
    plt.bar(y_pos[6:11], performance[6:11], align='center', alpha=0.5, width=8, color='yellow', label='5% < MMRE < 10%', hatch='O', log=True)
    plt.bar(y_pos[11:22], performance[11:22], align='center', alpha=0.5, width=8, color='red', label='10 < MMRE < 100%', hatch='.', log=True)
    plt.bar(y_pos[22:], performance[22:], align='center', alpha=0.5, width=8, color='violet', label='>10-%', hatch='+', log=True)
    plt.xticks(y_pos, projects, rotation='vertical')
    # # plt.yscale('log')
    plt.ylim(-1, 10000)
    plt.xlim(-15, 260)
    plt.ylabel('MMRE')
    plt.xlabel('Software Systems')

    # Now add the legend with some customizations.
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, frameon=False, fontsize=10)

    plt.savefig('./src/Generate_Bands/figures/artifical_models.png', bbox_inches='tight')


def draw_fig_cloud_sim(pickle_file):
    def modify_name(name):
        filename = name.split('/')[-1]
        return filename.split('.')[0]

    import pickle
    data = pickle.load(open(pickle_file, 'r'))
    import matplotlib.pyplot as plt
    data = sorted(data, key=lambda x: x[1])
    for d in data:
        print d
    # projects = ["SS"+str(i+1) for i,d in enumerate(data)]
    projects = [modify_name(d[0]) for d in data]
    y_pos = [i*10 for i in np.arange(len(projects))]
    performance = [d[1][0] for d in data]

    # plt.plot([-5, 255], [5, 5], 'k-', lw=2, color='black')
    # plt.plot([-5, 255], [10, 10], '--', lw=2, color='black')
    # plt.plot([-5, 255], [100, 100], '-.', lw=2, color='black')
    plt.bar(y_pos, performance, align='center', alpha=0.5, width=8, color='green', label='< 5%', hatch='o', log=True)

    # plt.bar(y_pos[:6], performance[:6], align='center', alpha=0.5, width=8, color='green', label='< 5%', hatch='o', log=True)
    # plt.bar(y_pos[6:11], performance[6:11], align='center', alpha=0.5, width=8, color='yellow', label='5% < MMRE < 10%', hatch='O', log=True)
    # plt.bar(y_pos[11:22], performance[11:22], align='center', alpha=0.5, width=8, color='red', label='10 < MMRE < 100%', hatch='.', log=True)
    # plt.bar(y_pos[22:], performance[22:], align='center', alpha=0.5, width=8, color='violet', label='>10-%', hatch='+', log=True)
    plt.xticks(y_pos, projects, rotation='vertical')
    # # plt.yscale('log')
    # plt.ylim(-1, 10000)
    # plt.xlim(-15, 260)
    plt.ylabel('MMRE')
    plt.xlabel('Software Systems')

    # Now add the legend with some customizations.
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, frameon=False, fontsize=10)

    plt.savefig('./src/Generate_Bands/figures/cloud_sim.png', bbox_inches='tight')


def draw_fig_model(pickle_file):
    def modify_name(name):
        filename = name.split('/')[-1]
        return filename.split('.')[0]

    import pickle
    data = pickle.load(open(pickle_file, 'r'))
    import matplotlib.pyplot as plt
    data = sorted(data, key=lambda x: x[1])
    for d in data:
        print d
    # projects = ["SS"+str(i+1) for i,d in enumerate(data)]
    projects = [modify_name(d[0]) for d in data]
    y_pos = [i*10 for i in np.arange(len(projects))]
    performance = [d[1][0] for d in data]

    # plt.plot([-5, 255], [5, 5], 'k-', lw=2, color='black')
    # plt.plot([-5, 255], [10, 10], '--', lw=2, color='black')
    # plt.plot([-5, 255], [100, 100], '-.', lw=2, color='black')
    plt.bar(y_pos, performance, align='center', alpha=0.5, width=8, color='green', label='< 5%', hatch='o', log=True)

    # plt.bar(y_pos[:6], performance[:6], align='center', alpha=0.5, width=8, color='green', label='< 5%', hatch='o', log=True)
    # plt.bar(y_pos[6:11], performance[6:11], align='center', alpha=0.5, width=8, color='yellow', label='5% < MMRE < 10%', hatch='O', log=True)
    # plt.bar(y_pos[11:22], performance[11:22], align='center', alpha=0.5, width=8, color='red', label='10 < MMRE < 100%', hatch='.', log=True)
    # plt.bar(y_pos[22:], performance[22:], align='center', alpha=0.5, width=8, color='violet', label='>10-%', hatch='+', log=True)
    plt.xticks(y_pos, projects, rotation='vertical')
    # # plt.yscale('log')
    # plt.ylim(-1, 10000)
    # plt.xlim(-15, 260)
    plt.ylabel('MMRE')
    plt.xlabel('Software Systems')

    # Now add the legend with some customizations.
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fancybox=True, frameon=False, fontsize=10)

    plt.savefig('./src/Generate_Bands/figures/model.png', bbox_inches='tight')


def gather_data(data_folder):
    data_type = data_folder.split('/')[-2]  # There is a trailing '/'
    data = get_data(data_folder)
    for d in data: print d  # debug
    import pickle
    pickle.dump(data, open("./src/Generate_Bands/pickle_locker/" + data_type + ".p", 'w'))
    print "Done"
    return "./src/Generate_Bands/pickle_locker/" + data_type + ".p"


def draw_figure(pickle_file):
    import pickle
    data = pickle.load(open(pickle_file, 'r'))
    draw_fig(data)

def misc():
    import pickle
    data = pickle.load(open('fig1.p', 'r'))
    data = sorted(data, key=lambda x:x[1][0])
    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    gather_data(data_folder="Data/artifical_models")
    # draw_figure()
    # misc()