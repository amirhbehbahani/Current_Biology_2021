############ Figure 3
# An agent-based model using iterative odometric integration recapitulates Drosophila local search around a single fictive food site. 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from custom_fcns import adjust_spines
from custom_fcns import safe_div
from custom_fcns import find_nearest
from custom_fcns import mean_confidence_interval
from custom_fcns import vonmises_kde


sc = (180/np.pi)/6.87


panel_B = 1
panel_C = 1
panel_D = 1
#
panel_E = 1
panel_F = 1
panel_G = 1
panel_H = 1


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


###### panel A: State transition diagram -- no data processing 

###### panel B: 2D FR sample trace single trial 52 BL arena
if panel_B == 1:
	ind_FR = [3]
	#
	s1_Figure2B = 1.8*1.1
	s2_Figure2B = 2.08

	lw = 0.5

	co1 = 0
	for kk in range(len(ind_FR)):
		cond = 10
		Name_FR = '../Data/simulations/modeling_extracted_data/FR_extracted_data/FR_extracted_data_single/'
		# loading data
		food_temp = np.loadtxt(Name_FR+'food_FR_'+str(cond)+'_'+str(ind_FR[kk])+'.txt')
		#
		t_be_temp = np.loadtxt(Name_FR+'t_be_FR_'+str(cond)+'_'+str(ind_FR[kk])+'.txt')
		theta_be_temp = np.loadtxt(Name_FR+'theta_be_FR_'+str(cond)+'_'+str(ind_FR[kk])+'.txt')
		#
		ind_be = find_nearest(t_be_temp/60, 4)
		t_be_temp = t_be_temp[ind_be:]
		theta_be_temp = theta_be_temp[ind_be:]
		#
		t_du_temp = np.loadtxt(Name_FR+'t_du_FR_'+str(cond)+'_'+str(ind_FR[kk])+'.txt')
		theta_du_temp = np.loadtxt(Name_FR+'theta_du_FR_'+str(cond)+'_'+str(ind_FR[kk])+'.txt')
		#
		t_af_temp = np.loadtxt(Name_FR+'t_af_FR_'+str(cond)+'_'+str(ind_FR[kk])+'.txt')
		theta_af_temp = np.loadtxt(Name_FR+'theta_af_FR_'+str(cond)+'_'+str(ind_FR[kk])+'.txt')
		#
		t_b_aba_temp = np.loadtxt(Name_FR+'t_b_aba_FR_'+str(cond)+'_'+str(ind_FR[kk])+'.txt')
		theta_b_aba_temp = np.loadtxt(Name_FR+'theta_b_aba_FR_'+str(cond)+'_'+str(ind_FR[kk])+'.txt')

		t_temp_all = np.hstack([t_be_temp, t_du_temp, t_af_temp])
		theta_temp_all = np.hstack([theta_be_temp, theta_du_temp, theta_af_temp])
		#
		t_temp_all = np.hstack([t_be_temp, t_du_temp, t_b_aba_temp])
		theta_temp_all = np.hstack([theta_be_temp, theta_du_temp, theta_b_aba_temp])

		fig = plt.figure(201+kk)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_Figure2B, s2_Figure2B)
		#
		ax.plot(t_temp_all/60, theta_temp_all*sc, color = 'k', linewidth = 0.5*lw, alpha = 1)
		for hh in range(len(food_temp)):
			ax.plot((food_temp[hh]/60)*np.ones(2), np.linspace(25.5,26,2), color = 'r', linewidth = lw, alpha = 1)
		#
		plt.ylim([-26,26])
		#
		plt.yticks(13*np.arange(5)-26)
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/Figure3_plots/Figure3_B_single_FR_sample_'+str(cond)+'_'+str(ind_FR[kk])+'.pdf')

############# panel C KDE AP fly and FR model
if panel_C == 1:
	s1_Figure3C = 1
	s2_Figure3C = 0.9
	#
	min_mid = 0
	kappa = 200
	n_bins = 1000

	###### AP
	mid_th_all_adj = np.loadtxt('../Data/real_fly_data/KDE_run_midpoint/trial_mid_run_du.txt')
	# 
	NN = np.shape(mid_th_all_adj)[0]
	#
	mid_th_all_adj_f = mid_th_all_adj.flatten()
	mid_th_all_adj_f_nz = mid_th_all_adj_f[np.nonzero(mid_th_all_adj_f)]
	#
	[bins_circ, kde_circ] = vonmises_kde(mid_th_all_adj_f_nz, kappa, n_bins)  
	#
	kde_du = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		run_th_temp = mid_th_all_adj[kk,:]
		#
		run_th_temp = run_th_temp[np.nonzero(run_th_temp)]	
		if len(run_th_temp) > min_mid:
			# wrapping
			run_th_temp = (run_th_temp + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_du[kk,:]] = vonmises_kde(run_th_temp, kappa, n_bins) 

	kde_du_m = np.zeros(len(kde_circ))
	kde_du_u = np.zeros(len(kde_circ))
	kde_du_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_du_m)):
		kde_circ_temp = kde_du[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_du_m[ii], kde_du_l[ii], kde_du_u[ii]] = mean_confidence_interval(kde_circ_temp)
				
	kde_max = np.max(kde_du_u)
	kde_min = np.min(kde_du_l)
	#
	fig = plt.figure(301)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure3C, s2_Figure3C)
	#
	ax.plot(bins_circ_af_temp, kde_du_m, color = 'red', alpha = 0.5)
	ax.fill_between(bins_circ_af_temp, kde_du_u, kde_du_l, color = 'red', alpha = 0.5, linewidth = 0)
	#	
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 2.5])
	plt.yticks([0, 1.25, 2.5])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure3_plots/Figure3_C_KDE_run_mid_du.pdf')


	######## FR model
	cond = 10
	Name_data = '../Data/simulations/modeling_extracted_data/FR_extracted_data/'
	# AP
	mid_t_FR_du_all = np.loadtxt(Name_data+'FR_extracted_data_population/mid_t_FR_'+str(cond)+'_du_all.txt')
	mid_th_FR_du_all = np.loadtxt(Name_data+'FR_extracted_data_population/mid_th_FR_'+str(cond)+'_du_all.txt')

	NN = np.shape(mid_th_FR_du_all)[0]
	#
	mid_th_FR_du_all_f = mid_th_FR_du_all.flatten()
	mid_th_FR_du_all_f_nz = mid_th_FR_du_all_f[np.nonzero(mid_th_FR_du_all_f)]
	#
	[bins_circ, kde_circ] = vonmises_kde(mid_th_FR_du_all_f_nz, kappa, n_bins)  
	#
	kde_du_FR = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		run_th_temp = mid_th_FR_du_all[kk,:]
		#
		run_th_temp = run_th_temp[np.nonzero(run_th_temp)]	
		if len(run_th_temp) > min_mid:
			# wrapping
			run_th_temp = (run_th_temp + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_du_FR[kk,:]] = vonmises_kde(run_th_temp, kappa, n_bins) 

	kde_du_FR_m = np.zeros(len(kde_circ))
	kde_du_FR_u = np.zeros(len(kde_circ))
	kde_du_FR_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_du_FR_m)):
		kde_circ_temp = kde_du_FR[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_du_FR_m[ii], kde_du_FR_l[ii], kde_du_FR_u[ii]] = mean_confidence_interval(kde_circ_temp)
				
	kde_max = np.max(kde_du_FR_u)
	kde_min = np.min(kde_du_FR_l)

	#
	fig = plt.figure(302)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure3C, s2_Figure3C)
	#
	ax.plot(bins_circ_af_temp, kde_du_FR_m, color = 'red', alpha = 0.5)
	ax.fill_between(bins_circ_af_temp, kde_du_FR_u, kde_du_FR_l, color = 'red', alpha = 0.5, linewidth = 0)
	#	
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 2.5])
	plt.yticks([0, 1.25, 2.5])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure3_plots/Figure3_C_KDE_run_mid_du_FR.pdf')

############# panel C KDE AP fly and FR model
if panel_D == 1:
	s1_Figure3D = 1
	s2_Figure3D = 0.9
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
	fig = plt.figure(401)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure3D, s2_Figure3D)
	#
	ax.plot(bins_circ_af_temp, kde_af_m, color = 'blue', alpha = 0.5)
	ax.fill_between(bins_circ_af_temp, kde_af_u, kde_af_l, color = 'blue', alpha = 0.5, linewidth = 0)
	#	
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 2.5])
	plt.yticks([0, 1.25, 2.5])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure3_plots/Figure3_D_KDE_run_mid_af.pdf')

	######## FR model
	cond = 10
	Name_data = '../Data/simulations/modeling_extracted_data/FR_extracted_data/'
	# AP
	mid_t_FR_af_all = np.loadtxt(Name_data+'FR_extracted_data_population/mid_t_FR_'+str(cond)+'_af_all.txt')
	mid_th_FR_af_all = np.loadtxt(Name_data+'FR_extracted_data_population/mid_th_FR_'+str(cond)+'_af_all.txt')

	NN = np.shape(mid_th_FR_af_all)[0]
	#
	mid_th_FR_af_all_f = mid_th_FR_af_all.flatten()
	mid_th_FR_af_all_f_nz = mid_th_FR_af_all_f[np.nonzero(mid_th_FR_af_all_f)]
	#
	[bins_circ, kde_circ] = vonmises_kde(mid_th_FR_af_all_f_nz, kappa, n_bins)  
	#
	kde_af_FR = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		run_th_temp = mid_th_FR_af_all[kk,:]
		#
		run_th_temp = run_th_temp[np.nonzero(run_th_temp)]	
		if len(run_th_temp) > min_mid:
			# wrapping
			run_th_temp = (run_th_temp + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_af_FR[kk,:]] = vonmises_kde(run_th_temp, kappa, n_bins) 

	kde_af_FR_m = np.zeros(len(kde_circ))
	kde_af_FR_u = np.zeros(len(kde_circ))
	kde_af_FR_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_af_FR_m)):
		kde_circ_temp = kde_af_FR[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_af_FR_m[ii], kde_af_FR_l[ii], kde_af_FR_u[ii]] = mean_confidence_interval(kde_circ_temp)
				
	kde_max = np.max(kde_af_FR_u)
	kde_min = np.min(kde_af_FR_l)

	#
	fig = plt.figure(402)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure3D, s2_Figure3D)
	#
	ax.plot(bins_circ_af_temp, kde_af_FR_m, color = 'blue', alpha = 0.5)
	ax.fill_between(bins_circ_af_temp, kde_af_FR_u, kde_af_FR_l, color = 'blue', alpha = 0.5, linewidth = 0)
	#	
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 2.5])
	plt.yticks([0, 1.25, 2.5])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure3_plots/Figure3_D_KDE_run_mid_af_FR.pdf')

###### panel E: small arena trial FR trial sample trace
if panel_E == 1:

	ind_FR = [0, 2, 5, 6, 12, 13]
	#
	s1_Figure3E = 4.5
	s2_Figure3E = 3

	lw = 1
	fil = 180
	no = 15
	cond = 1

	sc = (180/np.pi)/(2*6.87)

	co1 = 0
	Name_FR = '../Data/simulations/modeling_extracted_data/FR_extracted_data/FR_extracted_data_single/'
	
	for kk in range(len(ind_FR)):
		#
		t_af_temp_all = np.loadtxt(Name_FR+'t_af_FR_'+str(cond)+'_'+str(kk)+'.txt')
		theta_af_temp_all = np.loadtxt(Name_FR+'theta_af_FR_'+str(cond)+'_'+str(kk)+'.txt')
		#
		t_b_aba_ind = np.loadtxt(Name_FR+'t_b_aba_FR_'+str(cond)+'_'+str(kk)+'.txt')
		#
		theta_af_temp_all = np.unwrap(theta_af_temp_all)
		#
		t_b_aba_temp = t_af_temp_all[t_af_temp_all < t_b_aba_ind[-1]]
		theta_b_aba_temp = theta_af_temp_all[t_af_temp_all < t_b_aba_ind[-1]]
		#
		t_aba_temp = t_af_temp_all[t_af_temp_all > t_b_aba_ind[-1]]
		theta_aba_temp = theta_af_temp_all[t_af_temp_all > t_b_aba_ind[-1]]

		t_b_aba_temp = t_b_aba_temp - t_af_temp_all[0]
		t_aba_temp = t_aba_temp - t_af_temp_all[0]
		fig = plt.figure(501)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_Figure3E, s2_Figure3E)
		#
		ax.plot(t_b_aba_temp/60, theta_b_aba_temp*(180/np.pi), color = 'k', linewidth = 0.25*lw, alpha = 1)
		ax.plot(t_aba_temp/60, theta_aba_temp*(180/np.pi), color = (r[kk],g[kk],b[kk]), linewidth = 0.5*lw, alpha = 1)
		#
		#	
		plt.yticks(2*fil*np.arange(no)-7*fil, ('', '', '', '', '', '', '', '', '', '', '', '', '', '', ''))
		#
		plt.ylim([-7*2*fil-2, 7*2*fil+2])
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/Figure3_plots/Figure3_E_small_arena_FR_sample_'+str(cond)+'.pdf')

###### panel F: panel transits FR small arena
if panel_F == 1:
	cond = 1
	no = 15
	#
	s1_Figure3F = 1.5
	s2_Figure3F = 3*0.9936
	#
	Name_FR = '../Data/simulations/modeling_extracted_data/FR_extracted_data/FR_extracted_data_population/'

	theta = np.loadtxt(Name_FR+'/theta_FR_'+str(cond)+'.txt')
	m_co_aba_norm = np.loadtxt(Name_FR+'/m_co_aba_norm_FR_'+str(cond)+'.txt')
	l_co_aba_norm = np.loadtxt(Name_FR+'/l_co_aba_norm_FR_'+str(cond)+'.txt')
	h_co_aba_norm = np.loadtxt(Name_FR+'/h_co_aba_norm_FR_'+str(cond)+'.txt')

	fig = plt.figure(601)
	fig.set_size_inches(s2_Figure3F, s1_Figure3F)
	ax = fig.add_subplot(111, polar=False)
	#
	#
	ax.plot(theta, m_co_aba_norm, color = 'k', alpha = 1, linewidth = 0.5)
	ax.fill_between(theta, h_co_aba_norm, l_co_aba_norm, color = 'black', alpha = 0.3, linewidth = 0)
	#
	plt.xticks(360*np.arange(no-1)-2520, ('', '', '', '', '', '', '', '', '', '', '', '', '', '', ''))
	plt.yticks(0.008*np.arange(3))
	#
	plt.xlim([-7*360, 7*360])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure3_plots/Figure3_F_FR_transits_'+str(cond)+'.pdf')

###### KDE for run midpoint
if panel_G == 1:
	s1_Figure3G = 1.75
	s2_Figure3G = 2.5/2
	#	
	cond = 1
	Name_FR = '../Data/simulations/modeling_extracted_data/FR_extracted_data/FR_extracted_data_population/'
	kappa = 200
	n_bins = 2000
	#	
	bins_circ_temp = np.loadtxt(Name_FR+'/bins_circ_FR_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_m = np.loadtxt(Name_FR+'/kde_circ_m_FR_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_l = np.loadtxt(Name_FR+'/kde_circ_l_FR'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_u = np.loadtxt(Name_FR+'/kde_circ_u_FR'+str(cond)+'_'+str(kappa)+'.txt')


	fig = plt.figure(701)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure3G, s2_Figure3G)
	#
	ax.plot(bins_circ_temp, kde_circ_m, color = 'black', alpha = 1)
	ax.fill_between(bins_circ_temp, kde_circ_u, kde_circ_l, color = 'black', alpha = 0.3, linewidth = 0)
	#
	plt.plot(0*np.ones(2), np.linspace(0, 0.32, 2), '--k', alpha = 0.5)
	# #
	#
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 0.4])
	plt.yticks([0, 0.2, 0.4])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure3_plots/Figure3_G_FR_run_midpoint_KDE_kappa_'+str(kappa)+'_'+str(cond)+'.pdf')

######### number of midruns in quadrants
if panel_H == 1:
	s1_Figure3H = 1.6
	s2_Figure3H = 2.5/2
	#
	cond = 1
	Name_FR = '../Data/simulations/modeling_extracted_data/FR_extracted_data/FR_extracted_data_population/'
	#
	mid_quad_no = np.loadtxt(Name_FR+'/mid_quad_no_FR_'+str(cond)+'.txt')
	NN = np.shape(mid_quad_no)[0]
	#
	fig = plt.figure(801)
	ax = fig.add_subplot(111, polar=False)
	ax.grid(False)
	fig.set_size_inches(s1_Figure3H, s2_Figure3H)
	#
	for kk in range(NN):
		xx = [0, 1]
		yy = [mid_quad_no[kk,0], np.mean([mid_quad_no[kk,1], mid_quad_no[kk,2], mid_quad_no[kk,3]])]
		#
		ax.plot(xx, yy, color = 'k', alpha = 1, linewidth = 0.1)		
		#
		plt.xticks([0,1])
		plt.yticks([0,25,50])
		#
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
	fig.savefig('../Plots/Figure3_plots/Figure3_H_FR_run_midpoint_dual_plot_'+str(cond)+'.pdf')


