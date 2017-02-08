
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import flopy
import flopy.modflow as fmf

def take_action(pumping_rate1,pumping_rate2):
#	print(pumping_rate1)
#	print(pumping_rate2)
	# Assign name and create modflow model object
	modelname = 'model'
	ml = fmf.Modflow(modelname, exe_name='../mf2005')

	# Model domain and grid definition
	ztop = 80.
	zbot = 0.
	nlay = 8
	nrow = 15
	ncol = 21
	delr = 10
	delc = 10
	delv = 10
	botm = np.linspace(ztop, zbot, nlay + 1) # bottom elevation of layers

	# Create the discretization object
	dis = fmf.ModflowDis(ml, nlay, nrow, ncol, delr=delr, delc=delc,
								   top=ztop, botm=botm[1:])

	# Variables for the BAS package
	ibound = np.ones((nlay, nrow, ncol), dtype=np.float128)
	ibound[:, :, 0] = -1 #layer,row,column
	ibound[:, :, -1] = -1
	
	strt = np.ones((nlay, nrow, ncol), dtype=np.float128)
	bound_factor = np.loadtxt("inputs/boundfactor.txt")
	strt[:, :, 0:20] = bound_factor[0]*80.
	strt[:, :, -1] = 60.
	bas = fmf.ModflowBas(ml, ibound=ibound, strt=strt) # basic class package

	# Add LPF package to the MODFLOW model
	hk_factor = np.loadtxt("inputs/hkfactor.txt")

	# load the stanford rock model
	#hk = np.loadtxt("rock-model/rock-model1/permeability.dat", skiprows=1)
	#hk = hk.reshape(30,130,100)
	#hk = hk.transpose((0,2,1)) # modflow: lays, rows, cols
	#hk = np.flipud(hk)
	#hk = hk*8.6e-4 # 1 Darcy ---> 0.86 meter per day
	#hk = hk[0:8,0:15,0:21]
	#print(hk.shape)
	#fig = plt.figure()
	#plt.imshow(hk[0,:,:])
	#plt.colorbar()
	#fig.savefig("starterhk.png")

	
	hk =hk_factor[0]* np.ones((nlay, nrow, ncol), dtype=np.float128)
	#hk[:,8,:] = 5
	#hk[:,1:8,:] = 5
	lpf = fmf.ModflowLpf(ml, hk=hk, vka=1.)

	# Add OC package to the MODFLOW model
	spd = {(0, 0): ['print head', 'print budget', 'save head', 'save budget']}
	oc = fmf.ModflowOc(ml, stress_period_data=spd, compact=True)

	# Create the well package
	# Remember to use zero-based layer, row, column indices!
	#pumping_rate = -500.
	wel_sp = [[5, 2 , 10,pumping_rate1],[5, 12, 10,pumping_rate2]]
	stress_period_data = {0: wel_sp}
	wel = fmf.ModflowWel(ml,stress_period_data=stress_period_data)

	#recharge
        factor_rch = np.loadtxt("inputs/rchfactor.txt")
	recharge = factor_rch[0]*1e-4
        rch = flopy.modflow.ModflowRch(ml, nrchop=1, rech=recharge)


	# Add PCG package to the MODFLOW model
	pcg = fmf.ModflowPcg(ml) #solver
	
	
	# Write the MODFLOW model input files
	ml.write_input()

	# Run the MODFLOW model
	success, buff = ml.run_model(silent=True)

	# Post process the results
	import flopy.utils.binaryfile as bf

	headobj = bf.HeadFile(modelname+'.hds',precision = 'double')

	idx1 = (0, 2, 10)
	idx2 = (0, 12 , 10)
	ts1 = headobj.get_ts(idx1)
	ts2 = headobj.get_ts(idx2)
	
	pumping_rate1 = -pumping_rate1
	pumping_rate2 = -pumping_rate2
	
	pi1 = 180*pumping_rate1-np.square(pumping_rate1)-5*pumping_rate1*(80-ts1[0,1])
	pi2 = 180*pumping_rate2-np.square(pumping_rate2)-5*pumping_rate2*(80-ts2[0,1])
	return (([-pi1,-pi2]))
	

c = False	
if c:	
	a = np.zeros((20,20),dtype='f')
	l = 0
	for i in range(0,-200,-10):
 		k = 0 
 		for j in range(0,-200,-10):
			profit = mftuto1(i,j)
			a[k,l] = profit[0]
			print('****** the pumping rates are ',i,' and ',j, ' % the profit is ',profit[0])
			k += 1
 		l += 1
		
	np.savetxt('resultsProfit.txt',a)


