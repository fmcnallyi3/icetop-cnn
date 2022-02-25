from itertools import product, permutations
import numpy as np

default_options = { 'q' :  ["mean","sum","product","min","max", None], 
                        't' : ["mean","sum","product","min","max", None, False], #this may be wrong, check in datatool
                        'normed' : [True, False],
                        'reco' : [ None, "plane", "laputop", "small"],
                        'cosz' : [False, True]}

set_options = {'q':[], 't':[], 'normed':[], 'reco':[], 'cosz':[]}
temp = list(product(default_options['q'],default_options['normed']))
print(temp) 

final = []
print(set_options.keys(0))
"""for el in temp:
    for val in el:


    default_options
"""


#practice = np.ones(2,3,4,5)

 practice[..., ..., ]