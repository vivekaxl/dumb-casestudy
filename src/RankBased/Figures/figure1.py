import matplotlib.pyplot as plt
import numpy as np

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

import pickle
pickle_data = pickle.load(open('merged.p'))

names = ['SS'+ str(i+1) for i in xrange(len(data))]
print names

new_data = []
for d in data:
    print d
    new_data.append([d[1]] + [np.mean(pickle_data[d[0]]['rank']['min_rank']),
                              np.mean(pickle_data[d[0]]['progressive']['min_rank']),
                              np.mean(pickle_data[d[0]]['projective']['min_rank']),
                              ])
for i,nd in enumerate(new_data): print i, nd
# import pdb
# pdb.set_trace()

gap = 35
left, width = .53, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

f, ((ax1, ax2, ax3)) = plt.subplots(1, 3)

# print ">> ", len([d[2] for d in data if 5 < d [-1] <= 10.5]), len([5*(i+1) for i in xrange(4, 13)])
print [d[1] for d in new_data if d[0] <= 5]
# for dumb learner
ax1.scatter([gap*(i+1) for i in xrange(0, 6)], [d[1] for d in new_data if d[0] <= 5], color='g', marker='v', s=34)
ax1.scatter([gap*(i+1) for i in xrange(6, 11)], [d[1] for d in new_data if 5 < d[0] <= 10], color='y', marker='o', s=34)
ax1.scatter([gap*(i+1) for i in xrange(11, 22)], [d[1] for d in new_data if 10 < d[0] < 100], color='r', marker='x', s=34)
ax1.scatter([gap*(i+1) for i in xrange(22, 25)], [d[1] for d in new_data if 100 < d[0] < 10000], color='violet', marker='h', s=34)

ax1.tick_params(axis=u'both', which=u'both',length=0)
ax1.set_ylim(-2,25)
ax1.set_xlim(10, 900)
ax1.set_title('Rank-based', fontsize=16)
ax1.set_ylabel("Rank Difference (RD)", fontsize=16)

# ax1.set_xlabel("Accuracy")
# ax1.set_yscale('log')

ax2.set_ylim(-2,25)
ax2.set_xlim(10, 900)
ax2.scatter([gap*(i+1) for i in xrange(0, 6)], [d[2] for d in new_data if d[0] <= 5], color='g', marker='v', s=34)
ax2.scatter([gap*(i+1) for i in xrange(6, 11)], [d[2] for d in new_data if 5 < d[0] <= 10], color='y', marker='o', s=34)
ax2.scatter([gap*(i+1) for i in xrange(11, 22)], [d[2] for d in new_data if 10 < d[0] < 100], color='r', marker='x', s=34)
ax2.scatter([gap*(i+1) for i in xrange(22, 25)], [d[2] for d in new_data if 100 < d[0] < 10000], color='violet', marker='h', s=34)

ax2.tick_params(axis=u'both', which=u'both',length=0)
ax2.set_title('Progressive Sampling', fontsize=16)
ax2.set_ylabel("Rank Difference (RD)", fontsize=16)
# ax2.set_xlabel("Accuracy")

ax3.set_ylim(-2,14)
ax3.set_xlim(10, 900)
ax3.scatter([gap*(i+1) for i in xrange(0, 6)], [d[3] for d in new_data if d[0] <= 5], color='g', marker='v', s=34)
ax3.scatter([gap*(i+1) for i in xrange(6, 11)], [d[3] for d in new_data if 5 < d[0] <= 10], color='y', marker='o', s=34)
ax3.scatter([gap*(i+1) for i in xrange(11, 22)], [d[3] for d in new_data if 10 < d[0] < 100], color='r', marker='x', s=34)
ax3.scatter([gap*(i+1) for i in xrange(22, 25)], [d[3] for d in new_data if 100 < d[0] < 10000], color='violet', marker='h', s=34)

ax3.tick_params(axis=u'both', which=u'both',length=0)
ax3.set_title('Projective Sampling', fontsize=16)
ax3.set_ylabel("Rank Difference (RD)", fontsize=16)
# ax3.set_xlabel("Accuracy")

from matplotlib.lines import Line2D

circ3 = Line2D([0], [0], linestyle="none", marker="x", alpha=0.3, markersize=10, color="r")
circ1 = Line2D([0], [0], linestyle="none", marker="v", alpha=0.4, markersize=10, color="g")
circ2 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.3, markersize=10, color="y")
circ4 = Line2D([0], [0], linestyle="none", marker="h", alpha=0.3, markersize=10, color="violet")

plt.sca(ax1)
plt.xticks([gap*(i+1) for i in xrange(0, 25)], names, rotation=90, fontsize=12)

plt.sca(ax2)
plt.xticks([gap*(i+1) for i in xrange(0, 25)], names, rotation=90, fontsize=12)

plt.sca(ax3)
plt.xticks([gap*(i+1) for i in xrange(0, 25)], names, rotation=90, fontsize=12)

plt.figlegend((circ1, circ2, circ3, circ4), ('<5%', '5%<MMRE<10%', '10%<MMRE<100%', '>100%'), frameon=False, loc='lower center',
              bbox_to_anchor=(0.4, -0.04),fancybox=True, ncol=4, fontsize=16)

f.set_size_inches(22, 5.5)
# plt.show()
plt.savefig('figure4_2.png', bbox_inches='tight')