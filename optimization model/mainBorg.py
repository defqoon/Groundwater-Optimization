
import numpy as np
from sys import *
from math import *
from borg import *
from matplotlib import pyplot as plt
from mainFlopy import take_action
import time

start_time = time.time()
nvars = 2
nobjs = 2
nconstr = 0

test = []
test2 = []

hk_range = (np.linspace(-3,1,5))
rch_range = (np.linspace(-3,1,5))
bound_range = (np.linspace(0.8,1.2,5))
for i in range(0,5,1):
	factor_hk = np.power(10,hk_range[i])
	np.savetxt("inputs/hkfactor.txt",[factor_hk,0])
	for j in range(0,5,1):
		factor_rch = np.power(10,rch_range[j])
		np.savetxt("inputs/rchfactor.txt",[factor_rch,0])
		for k in range(0,5,1):
		        factor_bound = bound_range[k]
			np.savetxt("inputs/boundfactor.txt",[factor_bound,0])
			
			borg = Borg(nvars, nobjs, nconstr, take_action)
			borg.setBounds(*[[-300, 0]]*nvars)
			borg.setEpsilons(*[0.1]*nobjs)
			result = borg.solve({"maxEvaluations":5000})
			
			for solution in result:
				obj = solution.getObjectives()
				test.append(obj[:])
				var = solution.getVariables()
				test2.append(var[:])

			name = "outputs/resultsPR1000"+str(i)+str(j)+str(k)+".txt"
			np.savetxt(name,test2)
            name = "outputs/resultsProfit1000"+str(i)+str(j)+str(k)+".txt"
            np.savetxt(name,test)



print("--- %s seconds ---" % (time.time() - start_time))

