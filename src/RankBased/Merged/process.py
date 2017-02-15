from __future__ import division
import pickle

"""This code is required to merge the atri_result.p and artificial_models.p into a single pickle file"""


mre_data = pickle.load(open('aritifical-models.p'))
atri_data = pickle.load(open('atri_result.p'))

merged = {}
files = mre_data.keys()
for file in files:
    merged[file] = {}
    dumb = mre_data[file][0.4]['dumb']
    progressive = mre_data[file][0.4]['random-progressive']
    try:
        projective = {}
        projective['mres'] = atri_data[file.split('/')[-1]]['atri_min_mre']
        projective['train_set_size'] = atri_data[file.split('/')[-1]]['atri_train_set']
        projective['min_rank'] = atri_data[file.split('/')[-1]]['atri_min_rank']
    except:
        print file, " doesn't exist"

    merged[file]['rank'] = dumb
    merged[file]['progressive'] = progressive
    if file == './artificial-models/twentyfive_interaction_bdbj.csv':
        print "ASdas"
        merged[file]['projective'] = {'mres':1e6, 'train_set_size':1e6, 'min_rank':1e6}
    else:
        merged[file]['projective'] = projective

import pdb
pdb.set_trace()
pickle.dump(merged, open('merged.p', 'w'))
