data =[['./artificial-models/twentyfive_interaction_apache.csv', 0.083999999999999991],
['./artificial-models/twentyfive_interaction_bdbc.csv', 0.087499999999999994],
['./artificial-models/no_interaction_bdbj.csv', 0.93349999999999977],
['./artificial-models/one_hundred_interaction_bdbj.csv', 1.0245000000000002],
['./artificial-models/fifty_interaction_bdbj.csv', 3.7429999999999999],
['./artificial-models/no_interaction_bdbc.csv', 4.6089999999999991],
['./artificial-models/no_interaction_apache.csv', 6.2700000000000005],
['./artificial-models/two_hundred_interaction_apache.csv', 6.2829999999999995],
['./artificial-models/fifty_interaction_apache.csv', 6.4870000000000001],
['./artificial-models/two_hundred_interaction_bdbj.csv', 7.3845000000000001],
['./artificial-models/one_hundred_interaction_apache.csv', 8.3614999999999995],
['./artificial-models/no_interaction_sqlite.csv', 10.097999999999999],
['./artificial-models/fifty_interaction_llvm.csv', 12.884],
['./artificial-models/no_interaction_llvm.csv', 13.211500000000001],
['./artificial-models/twentyfive_interaction_llvm.csv', 14.0185],
['./artificial-models/twentyfive_interaction_sqlite.csv', 14.347999999999999],
['./artificial-models/one_hundred_interaction_llvm.csv', 14.49],
['./artificial-models/fifty_interaction_sqlite.csv', 16.925000000000001],
['./artificial-models/one_hundred_interaction_bdbc.csv', 24.337499999999999],
['./artificial-models/fifty_interaction_bdbc.csv', 24.835000000000001],
['./artificial-models/one_hundred_interaction_sqlite.csv', 25.990999999999996],
['./artificial-models/two_hundred_interaction_llvm.csv', 42.072000000000003],
['./artificial-models/two_hundred_interaction_sqlite.csv', 1339.671],
['./artificial-models/two_hundred_interaction_bdbc.csv', 1528.663],
['./artificial-models/twentyfive_interaction_bdbj.csv', 4202.9879999999994]]

import numpy as np
import pickle
pickle_data = pickle.load(open('merged.p'))

names = ['SS'+ str(i+1) for i in xrange(len(data))]
print names

new_data = []
for d in data:
    print d
    new_data.append([d[1]] + [np.mean(pickle_data[d[0]]['rank']['train_set_size']),
                              np.mean(pickle_data[d[0]]['progressive']['train_set_size']),
                              np.mean(pickle_data[d[0]]['projective']['train_set_size']),
                              ])
for i,nd in enumerate(new_data): print i, nd

import numpy as np
import matplotlib.pyplot as plt
data = sorted(data, key=lambda x: x[-1])
N = len(data)
dumb_evals = [d[1] for d in new_data]

space = 7
ind = np.arange(space, space*(len(data)+1), space)  # the x locations for the groups
width = 1.5        # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, dumb_evals, width, color='#f0f0f0', log=True, label='Rank-based')

random_evals = [d[2] for d in new_data]
rects2 = ax.bar(ind + 1 * width, random_evals, width, color='#bdbdbd', log=True, label='Progessive Sampling')

atri_evals = [d[3] for d in new_data]
rects3 = ax.bar(ind + 2 * width, atri_evals, width, color='#636363', log=True, label='Projective Sampling')

# add some text for labels, title and axes ticks
# For shading
# plt.axvspan(5, 34, color='g', alpha=0.2, lw=0)
# plt.axvspan(34, 90, color='y', alpha=0.2, lw=0)
# plt.axvspan(90, 154, color='r', alpha=0.2, lw=0)

# ax.add_patch ()

ax.set_ylabel('Number of Evaluations')
# ax.set_title('Scores by group and gender')

ax.set_xticks(ind + 3*width / 2)
ax.set_xticklabels(['SS'+str(x+1) for x in xrange(len(data))], rotation='vertical')

ax.set_xlim(3, 183)
ax.set_ylim(1, 1400)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, frameon=False)

# ax.legend((rects1[0], rects2[0], rects3[0]), ('Rank based Approach', 'Progressive Sampling', 'Projective Sampling'))


# plt.show()

fig.set_size_inches(14, 5)
# plt.show()
plt.savefig('figure6.png', bbox_inches='tight')
