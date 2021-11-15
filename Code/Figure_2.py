############ Figure 2
# Flies reinitiate a local search at a former fictive food site after circling the arena.


import numpy as np
import matplotlib.pyplot as plt

from custom_fcns import adjust_spines
from custom_fcns import safe_div
from custom_fcns import mean_confidence_interval
from custom_fcns import vonmises_kde


sc = (180/np.pi)/13.75
fil = 360/13.75

panel_B = 1
panel_C = 1
panel_D = 1
panel_E = 1
panel_F = 1
panel_G = 1


# Sand DDCC77
r1 = 221/255
g1 = 204/255
b1 = 119/255 
# Olive 999933
r2 = 153/255
g2 = 153/255
b2 = 51/255
# Purple AA4499
r3 = 170/255
g3 = 68/255
b3 = 153/255 
# Indigo 332288
r4 = 51/255
g4 = 34/255
b4 = 136/255 
# Green 117733
r5 = 17/255
g5 = 119/255
b5 = 51/255
# Cyan 88CCEE
r6 = 136/255
g6 = 204/255
b6 = 238/255 

r = [r1, r2, r3, r4, r5, r6]
g = [g1, g2, g3, g4, g5, g6]
b = [b1, b2, b3, b4, b5, b6]

###### panel A: Experimental setup top view -- no data processing

###### panel B: sample trace from 
if panel_B == 1:
	fly = [1]
	#
	s1_Figure2B = 4.5
	s2_Figure2B = 3

	lw = 1
	no = 24

	for kk in range(len(fly)):
		# loading data
		# baseline
		exec('t_be_temp= np.loadtxt("../Data/real_fly_data/Circular/time_theta/time_theta_be/t_be_"+str(fly[kk])+".txt")')
		exec('theta_be_temp= np.loadtxt("../Data/real_fly_data/Circular/time_theta/time_theta_be/theta_be_"+str(fly[kk])+".txt")')
		#
		for ii in range(6):
			# AP
			exec('t_du'+str(ii+1)+'_temp = np.loadtxt("../Data/real_fly_data/Circular/time_theta/time_theta_du/t_du"+str(ii+1)+"_"+str(fly[kk])+".txt")')
			exec('theta_du'+str(ii+1)+'_temp = np.loadtxt("../Data/real_fly_data/Circular/time_theta/time_theta_du/theta_du"+str(ii+1)+"_"+str(fly[kk])+"_uw.txt")')
			# post-AP
			exec('t_af'+str(ii+1)+'_temp = np.loadtxt("../Data/real_fly_data/Circular/time_theta/time_theta_af/t_af"+str(ii+1)+"_"+str(fly[kk])+".txt")')
			exec('theta_af'+str(ii+1)+'_temp_uw = np.loadtxt("../Data/real_fly_data/Circular/time_theta/time_theta_af/theta_af"+str(ii+1)+"_"+str(fly[kk])+"_uw.txt")')
			#
			exec('t_af'+str(ii+1)+'_temp_dis = np.hstack([t_af'+str(ii+1)+'_temp, 1.5*fil])')
			exec('theta_af'+str(ii+1)+'_temp_uw_dis = np.hstack([theta_af'+str(ii+1)+'_temp_uw, 1.5*fil])')
			#
			exec('aba_temp = next(x for x, val in enumerate(abs(theta_af'+str(ii+1)+'_temp_uw_dis)*sc) if val > fil)') 
			# 
			exec('t_af'+str(ii+1)+'_temp_1rev = t_af'+str(ii+1)+'_temp_dis[aba_temp:-1]')
			exec('theta_af'+str(ii+1)+'_temp_uw_1rev = theta_af'+str(ii+1)+'_temp_uw_dis[aba_temp:-1]')

		fig = plt.figure(201+kk)
		fig.set_size_inches(s1_Figure2B, s2_Figure2B)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		plt.setp(ax.yaxis.get_ticklabels(), visible=False)
		#
		for ii in range(15):
		    ax.plot(np.linspace(0,6,2), (ii-7)*fil*np.ones(2), 'k', alpha = 1, linewidth = 0.25)
	
		for hh in range(6):
			exec('plt.plot(t_af'+str(hh+1)+'_temp - t_af'+str(hh+1)+'_temp[0], theta_af'+str(hh+1)+'_temp_uw*sc, color = "black", linewidth = 0.25*lw, markersize = 0, alpha = 1)')
			#
			exec('plt.plot(t_af'+str(hh+1)+'_temp_1rev - t_af'+str(hh+1)+'_temp[0], theta_af'+str(hh+1)+'_temp_uw_1rev*sc, color = (r[hh],g[hh],b[hh]), linewidth = 0.5*lw, markersize = 0, alpha = 1)')

		#	
		plt.xticks(np.arange(6), ('', '', '', '', '' ,''))
		plt.yticks(fil*np.arange(no-1)-7*fil, ('', '', '', '', '', '', '', '', '', '', '', '', '', '', ''))
		#
		plt.xlim([0, 5])
		plt.ylim([-7*fil-2, 7*fil+2])
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/Figure2_plots/Figure2_B_sample_trace_'+str(fly[kk])+'.pdf')

###### panel C: sum of the transits for the sample fly
if panel_C == 1:
	s1_Figure2C = 1.5
	s2_Figure2C = 3
	#
	no = 24


	# loading data
	theta = np.loadtxt('../Data/real_fly_data/Circular/transits/theta.txt')
	co_af = np.loadtxt('../Data/real_fly_data/Circular/transits/co_af.txt')
	co_af = co_af[0:7,:]

	co_af_norm = np.zeros(np.shape(co_af))
	#
	for kk in range(np.shape(co_af)[0]):
		for uu in range(np.shape(co_af)[1]):
			co_af_norm[kk,uu] = safe_div(co_af[kk,uu],np.sum(co_af[kk,:]))

	co_af_norm = co_af_norm.sum(axis = 0)/6

	fig = plt.figure(301)
	fig.set_size_inches(s1_Figure2C, s2_Figure2C)
	ax = fig.add_subplot(111, polar=False)
	ax.grid(False)
	plt.setp(ax.yaxis.get_ticklabels(), visible=False)
	#
	ax.plot(co_af_norm, theta, color = 'k', alpha = 1, linewidth = 0.5)
	#
	MM = 0.10
	#
	plt.xticks(0.03*np.arange(2), ('', ''))
	plt.yticks(360*np.arange(no-1)-2520, ('', '', '', '', '', '', '', '', '', '', '', '', '', '', ''))
	#
	# plt.xlim([0, 0.2])
	plt.ylim([-7*360, 7*360])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#	
	fig.savefig('../Plots/Figure2_plots/Figure2_C_transits_af_sample_vert_sum.pdf')


###### panel D: heatmap for transits for all the trials
if panel_D == 1:
	s1_Figure2D = 4.5
	s2_Figure2D = 5

	# loading data
	co_af = np.loadtxt('../Data/real_fly_data/Circular/transits/co_af.txt')
	co_af_norm = np.zeros((np.shape(co_af)[0],np.shape(co_af)[1]+1))

	ind1 = int(336/2) + 24
	ind2 = int(336/2) - 24
	for kk in range(np.shape(co_af)[0]):
		for uu in range(np.shape(co_af)[1]):
			co_af_norm[kk,uu] = safe_div(co_af[kk,uu],np.sum(co_af[kk,:]))
			co_af_norm[kk,-1] = -(co_af_norm[kk,ind1] + co_af_norm[kk,ind2])

	co_af_norm_sorted = co_af_norm[co_af_norm[:,-1].argsort()]
	co_af_norm_sorted = co_af_norm_sorted[:,0:-1]
	co_af_transpose = np.transpose(co_af_norm_sorted)
	#

	fig = plt.figure(401)
	fig.set_size_inches(s1_Figure2D, s2_Figure2D)
	ax = fig.add_subplot(111, polar=False)
	ax.grid(False)
	plt.setp(ax.yaxis.get_ticklabels(), visible=False)
	#
	plt.imshow(co_af_transpose, cmap ='Greys', vmin = 0, vmax = 0.03)
	#

	ind = np.arange(0,2*7*24+1,24)
	for ii in range(len(ind)):
		ax.plot(np.linspace(0,170,2), ind[ii]*np.ones(2), 'k', alpha = 1, linewidth = 0.25)	
	#
	plt.yticks(ind, ('', '', '', '', '', '', '', '', '', '', '', '', '', '', ''))
	plt.xticks([])
	# 
	adjust_spines(ax, ['left', 'bottom'])
	#	
	fig.savefig('../Plots/Figure2_plots/Figure2_D_heatmap_all_sorted_transpose.pdf')

###### panel E: sum of the transits for all flies
if panel_E == 1:
	s1_Figure2E = 1.5
	s2_Figure2E = 3*0.9936
	#
	no = 24

	# loading data
	theta = np.loadtxt('../Data/real_fly_data/Circular/transits/theta.txt')
	co_af = np.loadtxt('../Data/real_fly_data/Circular/transits/co_af.txt')

	co_af_norm = np.zeros(np.shape(co_af))

	#
	for kk in range(np.shape(co_af)[0]):
		for uu in range(np.shape(co_af)[1]):
			co_af_norm[kk,uu] = safe_div(co_af[kk,uu],np.sum(co_af[kk,:]))

	co_af_norm_sum = co_af_norm.sum(axis = 0)/np.shape(co_af)[0]

	# CI
	m_co_af_norm = np.zeros(np.shape(co_af)[1])
	h_co_af_norm = np.zeros(np.shape(co_af)[1])
	l_co_af_norm = np.zeros(np.shape(co_af)[1])

	for hh in range(np.shape(co_af)[1]):
		[m_co_af_norm[hh], h_co_af_norm[hh], l_co_af_norm[hh]] = mean_confidence_interval(co_af_norm[:,hh])

	fig = plt.figure(501)
	fig.set_size_inches(s2_Figure2E, s1_Figure2E)
	ax = fig.add_subplot(111, polar=False)
	ax.grid(False)
	plt.setp(ax.yaxis.get_ticklabels(), visible=False)
	#
	#
	ax.plot(theta, m_co_af_norm, color = 'k', alpha = 1, linewidth = 0.5)
	ax.fill_between(theta, h_co_af_norm, l_co_af_norm, color = 'black', alpha = 0.3, linewidth = 0)
	#
	plt.xticks(360*np.arange(no-1)-2520, ('', '', '', '', '', '', '', '', '', '', '', '', '', '', ''))
	plt.yticks(0.008*np.arange(3), ('', '', '', ''))
	#
	plt.xlim([-7*360, 7*360])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#	
	fig.savefig('../Plots/Figure2_plots/Figure2_E_transits_af_all_vert_sum.pdf')

###### panel F: Kernel Density Estimate of run midpoints post-departure
if panel_F == 1:
	s1_Figure2F = 1.75
	s2_Figure2F = 2.5/2
	#
	run_thresh = 0.75*np.pi
	run_thresh = 1*np.pi
	#

	# loading data
	fly = np.loadtxt('../Data/real_fly_data/Circular/rev_after_1rev/fly.txt')
	rev_t_1rev_all = np.loadtxt('../Data/real_fly_data/Circular/rev_after_1rev/rev_t_1rev_all.txt')
	rev_th_1rev_all = np.loadtxt('../Data/real_fly_data/Circular/rev_after_1rev/rev_th_1rev_all.txt')
	#
	run_mid_th_1rev_all = np.zeros((6*len(fly),200))
	#
	for kk in range(6*len(fly)):
		rev_th_temp = rev_th_1rev_all[kk,:]
		rev_th_temp = rev_th_temp[np.nonzero(rev_th_temp)]
		#
		if len(rev_th_temp) > 1:
			for ii in range(len(rev_th_temp)-1):
				if np.abs(rev_th_temp[ii+1] - rev_th_temp[ii]) < run_thresh:
					run_mid_th_1rev_all[kk,ii] = 0.5*(rev_th_temp[ii]+rev_th_temp[ii+1])

	run_mid_th_1rev_all_f = run_mid_th_1rev_all.flatten()
	run_mid_th_1rev_all_f_nz = run_mid_th_1rev_all_f[np.nonzero(run_mid_th_1rev_all_f)]
	#
	kappa = 200
	n_bins = 2000
	#
	[bins_circ, kde_circ] = vonmises_kde(run_mid_th_1rev_all_f_nz, kappa, n_bins)  
	#
	kde_circ_all = np.zeros((len(fly), len(kde_circ)))
	#
	for kk in range(len(fly)):
		rev_temp = run_mid_th_1rev_all[6*kk+0:6*kk+6,:]
		rev_temp = rev_temp.flatten()
		rev_temp = rev_temp[np.nonzero(rev_temp)]
		# wrapping
		rev_temp = (rev_temp + np.pi) % (2 * np.pi) - np.pi
		#
		[bins_circ_temp, kde_circ_all[kk,:]] = vonmises_kde(rev_temp, kappa, n_bins) 

	kde_circ_m = np.zeros(len(kde_circ))
	kde_circ_u = np.zeros(len(kde_circ))
	kde_circ_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_circ_m)):
		[kde_circ_m[ii], kde_circ_l[ii], kde_circ_u[ii]] = mean_confidence_interval(kde_circ_all[:,ii])
		
	kde_max = np.max(kde_circ_u)
	kde_min = np.min(kde_circ_l)
	#
	fig = plt.figure(701)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure2F, s2_Figure2F)
	#
	ax.plot(bins_circ_temp, kde_circ_m, color = 'black', alpha = 1)
	ax.fill_between(bins_circ_temp, kde_circ_u, kde_circ_l, color = 'black', alpha = 0.3, linewidth = 0)
	#
	plt.plot(0*np.ones(2), np.linspace(0, 0.5, 2), '--k', alpha = 0.5)
	# #
	#
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 0.5])
	plt.yticks([0, 0.25, 0.5])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure2_plots/Figure2_F_KDE_kappa_'+str(kappa)+'_thresh'+str(int(180/np.pi*run_thresh))+'.pdf')
	


###### panel G: number of midruns in each quadrant
if panel_G == 1:
	s1_Figure2G = 1.75
	s2_Figure2G = 2.5/2
	#
	NN = 27

	# loading data
	No_mid_W_fly = np.loadtxt('../Data/real_fly_data/Circular/Midruns_quadrant/No_mid_W_fly.txt')
	No_mid_E_fly = np.loadtxt('../Data/real_fly_data/Circular/Midruns_quadrant/No_mid_E_fly.txt')
	No_mid_N_fly = np.loadtxt('../Data/real_fly_data/Circular/Midruns_quadrant/No_mid_N_fly.txt')
	No_mid_S_fly = np.loadtxt('../Data/real_fly_data/Circular/Midruns_quadrant/No_mid_S_fly.txt')
	#
	fig = plt.figure(601)
	ax = fig.add_subplot(111, polar=False)
	ax.grid(False)
	fig.set_size_inches(s1_Figure2G, s2_Figure2G)
	#
	for kk in range(NN):
		xx = [0, 1, 2]
		yy = [No_mid_W_fly[kk], No_mid_E_fly[kk], np.mean([No_mid_N_fly[kk], No_mid_S_fly[kk]])]
		#
		ax.plot(xx, yy, color = 'k', alpha = 1, linewidth = 0.5)		
	#
	plt.xticks([0,1,2])
	plt.yticks([0,10,20])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure2_plots/Figure2_G_trippled_plot_mid_fly.pdf')

