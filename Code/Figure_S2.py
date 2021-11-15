############ Figure S2
# Memory-less models cannot recapitulate Drosophila local search.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from custom_fcns import adjust_spines
from custom_fcns import safe_div
from custom_fcns import find_nearest
from custom_fcns import mean_confidence_interval
from custom_fcns import vonmises_kde


sc = (180/np.pi)/6.87

panel_A = 1
panel_B = 1

panel_C = 1
panel_D = 1

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


##### panel A: randomly sampled traces
if panel_A == 1:
	ind_sampling = [0, 1, 3, 4, 5, 7]
	#
	S2_FigureS2A = 2.5
	s2_FigureS2A = 2.08

	lw = 0.5

	co1 = 0
	for kk in range(len(ind_sampling)):
		# loading data
		t_temp = np.loadtxt('../Data/simulations/randomly_sampled/time_theta/t_sampling_'+str(ind_sampling[kk])+'.txt')
		theta_temp = np.loadtxt('../Data/simulations/randomly_sampled/time_theta/theta_sampling_'+str(ind_sampling[kk])+'.txt')

		fig = plt.figure(101)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(S2_FigureS2A, s2_FigureS2A)
		#
		ax.plot(t_temp/60, theta_temp, color = (r[co1],g[co1],b[co1]), linewidth = lw, alpha = 1)
		co1 = co1+1
		#
		# plt.xlim([8,60])
		plt.ylim([-26,26])
		#
		# plt.xticks([])
		plt.yticks(13*np.arange(5)-26)
		#
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/FigureS2_plots/FigureS2_A_sampling_samples.pdf')

##### panel B: sample trace Levy flight
if panel_B == 1:
	ind_Levy = [1, 2, 3, 5, 6, 11]
	#
	S2_FigureS2B = 2.5
	s2_FigureS2B = 2.08

	lw = 0.5

	co1 = 0
	for kk in range(len(ind_Levy)):
		# loading data
		t_temp = np.loadtxt('../Data/simulations/Levy_flight/time_theta/t_Levy_'+str(ind_Levy[kk])+'.txt')
		theta_temp = np.loadtxt('../Data/simulations/Levy_flight/time_theta/theta_Levy_'+str(ind_Levy[kk])+'.txt')

		fig = plt.figure(201)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(S2_FigureS2B, s2_FigureS2B)
		#
		ax.plot(t_temp/60, theta_temp, color = (r[co1],g[co1],b[co1]), linewidth = lw, alpha = 1)
		co1 = co1+1
		#
		plt.ylim([-26,26])
		#
		plt.yticks(13*np.arange(5)-26)
		#
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/FigureS2_plots/FigureS2_B_Levy_samples.pdf')
			


########### KDE post-AP
if panel_C == 1:
	s1_FigureS2C = 1.5
	s2_FigureS2C = 1.2
	#
	min_mid = 0
	kappa = 200
	n_bins = 1000

	###### post-AP
	mid_th_all_adj = np.loadtxt('../Data/real_fly_data/KDE_run_midpoint/trial_mid_run_af.txt')
	# 
	NN = np.shape(mid_th_all_adj)[0]
	#
	mid_th_all_adj_f = mid_th_all_adj.flatten()
	mid_th_all_adj_f_nz = mid_th_all_adj_f[np.nonzero(mid_th_all_adj_f)]
	#
	[bins_circ, kde_circ] = vonmises_kde(mid_th_all_adj_f_nz, kappa, n_bins)  
	#
	kde_af = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		run_th_temp = mid_th_all_adj[kk,:]
		#
		run_th_temp = run_th_temp[np.nonzero(run_th_temp)]	
		if len(run_th_temp) > min_mid:
			# wrapping
			run_th_temp = (run_th_temp + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_af[kk,:]] = vonmises_kde(run_th_temp, kappa, n_bins) 

	kde_af_m = np.zeros(len(kde_circ))
	kde_af_u = np.zeros(len(kde_circ))
	kde_af_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_af_m)):
		kde_circ_temp = kde_af[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_af_m[ii], kde_af_l[ii], kde_af_u[ii]] = mean_confidence_interval(kde_circ_temp)
				
	kde_max = np.max(kde_af_u)
	kde_min = np.min(kde_af_l)
	#
	fig = plt.figure(301)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_FigureS2C, s2_FigureS2C)
	#
	ax.plot(bins_circ_af_temp, kde_af_m, color = 'blue', alpha = 0.5)
	ax.fill_between(bins_circ_af_temp, kde_af_u, kde_af_l, color = 'blue', alpha = 0.5, linewidth = 0)
	#	
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 1.2])
	plt.yticks([0, 0.6, 1.2])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/FigureS2_plots/FigureS2_C_KDE_run_mid_af.pdf')


	###### post-AP randomly sampled
	mid_th_sampling_all = np.loadtxt('../Data/simulations/randomly_sampled/reversals/mid_th_sampling_all.txt')	
	# 
	NN = np.shape(mid_th_sampling_all)[0]
	#
	mid_th_sampling_all_f = mid_th_sampling_all.flatten()
	mid_th_sampling_all_f_nz = mid_th_sampling_all_f[np.nonzero(mid_th_sampling_all_f)]
	#
	[bins_circ, kde_circ] = vonmises_kde(mid_th_sampling_all_f_nz, kappa, n_bins)  
	#
	kde_af = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		run_th_temp = mid_th_sampling_all[kk,:]
		#
		run_th_temp = run_th_temp[np.nonzero(run_th_temp)]	
		if len(run_th_temp) > min_mid:
			# wrapping
			run_th_temp = (run_th_temp + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_af[kk,:]] = vonmises_kde(run_th_temp, kappa, n_bins) 

	kde_af_m = np.zeros(len(kde_circ))
	kde_af_u = np.zeros(len(kde_circ))
	kde_af_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_af_m)):
		kde_circ_temp = kde_af[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_af_m[ii], kde_af_l[ii], kde_af_u[ii]] = mean_confidence_interval(kde_circ_temp)
				
	kde_max = np.max(kde_af_u)
	kde_min = np.min(kde_af_l)
	#
	fig = plt.figure(302)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_FigureS2C, s2_FigureS2C)
	#
	ax.plot(bins_circ_af_temp, kde_af_m, color = 'blue', alpha = 0.5)
	ax.fill_between(bins_circ_af_temp, kde_af_u, kde_af_l, color = 'blue', alpha = 0.5, linewidth = 0)
	#	
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 1.2])
	plt.yticks([0, 0.6, 1.2])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/FigureS2_plots/FigureS2_C_KDE_run_mid_randomly_sampled.pdf')


	###### post-AP levy
	mid_th_Levy_all = np.loadtxt('../Data/simulations/Levy_flight/reversals/mid_th_Levy_all.txt')	
	# 
	NN = np.shape(mid_th_Levy_all)[0]
	#
	mid_th_Levy_all_f = mid_th_Levy_all.flatten()
	mid_th_Levy_all_f_nz = mid_th_Levy_all_f[np.nonzero(mid_th_Levy_all_f)]
	#
	[bins_circ, kde_circ] = vonmises_kde(mid_th_Levy_all_f_nz, kappa, n_bins)  
	#
	kde_af = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		run_th_temp = mid_th_Levy_all[kk,:]
		#
		run_th_temp = run_th_temp[np.nonzero(run_th_temp)]	
		if len(run_th_temp) > min_mid:
			# wrapping
			run_th_temp = (run_th_temp + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_af[kk,:]] = vonmises_kde(run_th_temp, kappa, n_bins) 

	kde_af_m = np.zeros(len(kde_circ))
	kde_af_u = np.zeros(len(kde_circ))
	kde_af_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_af_m)):
		kde_circ_temp = kde_af[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_af_m[ii], kde_af_l[ii], kde_af_u[ii]] = mean_confidence_interval(kde_circ_temp)
				
	kde_max = np.max(kde_af_u)
	kde_min = np.min(kde_af_l)
	#
	fig = plt.figure(303)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_FigureS2C, s2_FigureS2C)
	#
	ax.plot(bins_circ_af_temp, kde_af_m, color = 'blue', alpha = 0.5)
	ax.fill_between(bins_circ_af_temp, kde_af_u, kde_af_l, color = 'blue', alpha = 0.5, linewidth = 0)
	#	
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 1.2])
	plt.yticks([0, 0.6, 1.2])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/FigureS2_plots/FigureS2_C_KDE_run_mid_Levy.pdf')


