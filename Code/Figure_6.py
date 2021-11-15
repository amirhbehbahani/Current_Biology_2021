############ Figure 6
# Flies reset their path integrator at the center of a cluster of fictive food sites. 

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from custom_fcns import adjust_spines
from custom_fcns import find_nearest
from custom_fcns import vonmises_kde
from custom_fcns import mean_confidence_interval

sc = (180/np.pi)/6.87

panel_CDE = 1
panel_F = 1
#
panel_GHI = 1
panel_J = 1
#
panel_KLM = 1
panel_N = 1


# top Top green 117733
r1 = 17/255
g1 = 119/255
b1 = 51/255
# Middle Olive 999933
r2 = 153/255
g2 = 153/255
b2 = 51/255
# bottom Bottom Purple AA4499
r3 = 170/255
g3 = 68/255
b3 = 153/255

rr_com = [r1, r2, r3]
gg_com = [g1, g2, g3]
bb_com = [b1, b2, b3]


kappa = 200
n_bins = 1000
min_mid = 3

##### panel A: experimental setup top view -- no data processing

##### panel B: Cartoon  -- no data processing

##### panel CDE sample trace FR_p
if panel_CDE == 1:
	s1_Figure6CDE = 2.25
	s2_Figure6CDE = 1.5

	Name_data = '../Data/simulations/modeling_extracted_data/FR_p_extracted_data/FR_p_one_shot/'
	dur_du = 3
	dur_af = 1
	dur = dur_du + dur_af
	#
	lw = 0.5
	al = 1

	ind_FR_p_b = [43]
	ind_FR_p_m = [72]
	ind_FR_p_t = [71]
	#
	# bottom
	for kk in range(len(ind_FR_p_b)):
		t_du_temp = np.loadtxt(Name_data+'t_du_FR_p_1_'+str(ind_FR_p_b[kk])+'.txt')
		theta_du_temp = np.loadtxt(Name_data+'theta_du_FR_p_1_'+str(ind_FR_p_b[kk])+'.txt')
		#
		t_af_temp = np.loadtxt(Name_data+'t_af_FR_p_1_'+str(ind_FR_p_b[kk])+'.txt')
		theta_af_temp = np.loadtxt(Name_data+'theta_af_FR_p_1_'+str(ind_FR_p_b[kk])+'.txt')
		#
		
		ind_du_s = find_nearest(t_du_temp, t_du_temp[-1] - dur_du*60)
		ind_af_e = find_nearest(t_af_temp, t_af_temp[0] + dur_af*60)
		#
		t_du_temp = t_du_temp[ind_du_s:]
		theta_du_temp = theta_du_temp[ind_du_s:]
		#
		t_af_temp = t_af_temp[0:ind_af_e]
		theta_af_temp = theta_af_temp[0:ind_af_e]

		t_temp = np.hstack([t_du_temp, t_af_temp])
		theta_temp = np.hstack([theta_du_temp, theta_af_temp])

		#
		fig = plt.figure(301+kk)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_Figure6CDE, s2_Figure6CDE)
		#
		ax.plot((t_temp - t_temp[0])/60, theta_temp*sc, color = 'k', linewidth = lw, alpha = al)
		#
		plt.xlim([0,dur])
		plt.ylim([-26,26])
		#
		plt.xticks(dur*np.arange(2))
		plt.yticks(13*np.arange(5)-26)
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/Figure6_plots/Figure6_C_FR_p_bottom_'+str(ind_FR_p_b[kk])+'.pdf')

	# middle
	for kk in range(len(ind_FR_p_m)):
		t_du_temp = np.loadtxt(Name_data+'t_du_FR_p_2_'+str(ind_FR_p_m[kk])+'.txt')
		theta_du_temp = np.loadtxt(Name_data+'theta_du_FR_p_2_'+str(ind_FR_p_m[kk])+'.txt')
		#
		t_af_temp = np.loadtxt(Name_data+'t_af_FR_p_2_'+str(ind_FR_p_m[kk])+'.txt')
		theta_af_temp = np.loadtxt(Name_data+'theta_af_FR_p_2_'+str(ind_FR_p_m[kk])+'.txt')
		#
		#
		ind_du_s = find_nearest(t_du_temp, t_du_temp[-1] - dur_du*60)
		ind_af_e = find_nearest(t_af_temp, t_af_temp[0] + dur_af*60)
		#
		t_du_temp = t_du_temp[ind_du_s:]
		theta_du_temp = theta_du_temp[ind_du_s:]
		#
		t_af_temp = t_af_temp[0:ind_af_e]
		theta_af_temp = theta_af_temp[0:ind_af_e]

		t_temp = np.hstack([t_du_temp, t_af_temp])
		theta_temp = np.hstack([theta_du_temp, theta_af_temp])
		#
		fig = plt.figure(401+kk)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_Figure6CDE, s2_Figure6CDE)
		#
		ax.plot((t_temp - t_temp[0])/60, theta_temp*sc, color = 'k', linewidth = lw, alpha = al)
		#
		plt.xlim([0,dur])
		plt.ylim([-26,26])
		#
		plt.xticks(dur*np.arange(2))
		plt.yticks(13*np.arange(5)-26)
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/Figure6_plots/Figure6_D_FR_p_middle_'+str(ind_FR_p_m[kk])+'.pdf')


	# top
	for kk in range(len(ind_FR_p_t)):
		t_du_temp = np.loadtxt(Name_data+'t_du_FR_p_3_'+str(ind_FR_p_t[kk])+'.txt')
		theta_du_temp = np.loadtxt(Name_data+'theta_du_FR_p_3_'+str(ind_FR_p_t[kk])+'.txt')
		#
		t_af_temp = np.loadtxt(Name_data+'t_af_FR_p_3_'+str(ind_FR_p_t[kk])+'.txt')
		theta_af_temp = np.loadtxt(Name_data+'theta_af_FR_p_3_'+str(ind_FR_p_t[kk])+'.txt')
		#
		#
		ind_du_s = find_nearest(t_du_temp, t_du_temp[-1] - dur_du*60)
		ind_af_e = find_nearest(t_af_temp, t_af_temp[0] + dur_af*60)
		#
		t_du_temp = t_du_temp[ind_du_s:]
		theta_du_temp = theta_du_temp[ind_du_s:]
		#
		t_af_temp = t_af_temp[0:ind_af_e]
		theta_af_temp = theta_af_temp[0:ind_af_e]

		t_temp = np.hstack([t_du_temp, t_af_temp])
		theta_temp = np.hstack([theta_du_temp, theta_af_temp])
		#
		fig = plt.figure(501+kk)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_Figure6CDE, s2_Figure6CDE)
		#
		ax.plot((t_temp - t_temp[0])/60, theta_temp*sc, color = 'k', linewidth = lw, alpha = al)
		#
		plt.xlim([0,dur])
		plt.ylim([-26,26])
		#
		plt.xticks(dur*np.arange(2))
		plt.yticks(13*np.arange(5)-26)
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/Figure6_plots/Figure6_E_FR_p_near_'+str(ind_FR_p_t[kk])+'.pdf')

#### KDE run midpoints FR_p
if panel_F == 1:
	s1_Figure6F = 2
	s2_Figure6F = 1
	#
	Name = '../Data/simulations/modeling_extracted_data/'
	run_af_FR_p_t_Bottom = np.loadtxt(Name+'FR_p_extracted_data/FR_p_one_shot_population/run_af_FR_p_t_Bottom.txt')
	run_af_FR_p_t_Middle = np.loadtxt(Name+'FR_p_extracted_data/FR_p_one_shot_population/run_af_FR_p_t_Middle.txt')
	run_af_FR_p_t_Top = np.loadtxt(Name+'FR_p_extracted_data/FR_p_one_shot_population/run_af_FR_p_t_Top.txt')
	#
	run_af_FR_p_th_Bottom = np.loadtxt(Name+'FR_p_extracted_data/FR_p_one_shot_population/run_af_FR_p_th_Bottom.txt')
	run_af_FR_p_th_Middle = np.loadtxt(Name+'FR_p_extracted_data/FR_p_one_shot_population/run_af_FR_p_th_Middle.txt')
	run_af_FR_p_th_Top = np.loadtxt(Name+'FR_p_extracted_data/FR_p_one_shot_population/run_af_FR_p_th_Top.txt')

	#

	# Bottom
	NN = np.shape(run_af_FR_p_th_Bottom)[0]
	run_af_FR_p_th_Bottom_f = run_af_FR_p_th_Bottom.flatten()
	run_af_FR_p_th_Bottom_f_nz = run_af_FR_p_th_Bottom_f[np.nonzero(run_af_FR_p_th_Bottom_f)]
	#
	[bins_circ, kde_circ] = vonmises_kde(run_af_FR_p_th_Bottom_f_nz, kappa, n_bins)  
	#
	kde_FR_p_Bottom = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		run_af_t_temp = run_af_FR_p_t_Bottom[kk,:]
		run_af_th_temp = run_af_FR_p_th_Bottom[kk,:]
		#
		run_af_t_temp_z = run_af_t_temp
		#
		run_af_t_temp_nz = run_af_t_temp[np.nonzero(run_af_t_temp_z)]
		run_af_th_temp_nz = run_af_th_temp[np.nonzero(run_af_t_temp_z)]	
		if len(run_af_th_temp_nz) > min_mid:
			# wrapping
			run_af_th_temp_nz = (run_af_th_temp_nz + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_FR_p_Bottom[kk,:]] = vonmises_kde(run_af_th_temp_nz, kappa, n_bins) 

	kde_FR_p_Bottom_m = np.zeros(len(kde_circ))
	kde_FR_p_Bottom_u = np.zeros(len(kde_circ))
	kde_FR_p_Bottom_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_FR_p_Bottom_m)):
		kde_circ_temp = kde_FR_p_Bottom[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_FR_p_Bottom_m[ii], kde_FR_p_Bottom_l[ii], kde_FR_p_Bottom_u[ii]] = mean_confidence_interval(kde_circ_temp)
				
	kde_max = np.max(kde_FR_p_Bottom_u)
	kde_min = np.min(kde_FR_p_Bottom_l)
	#
	fig = plt.figure(601)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure6F, s2_Figure6F)
	#
	ax.plot(bins_circ_af_temp, kde_FR_p_Bottom_m, color = (r3, g3, b3), alpha = 1)
	ax.fill_between(bins_circ_af_temp, kde_FR_p_Bottom_u, kde_FR_p_Bottom_l, color = (r3, g3, b3), alpha = 0.5, linewidth = 0)
	# 
	adjust_spines(ax, ['left', 'bottom'])


	# Middle
	NN = np.shape(run_af_FR_p_th_Middle)[0]
	run_af_FR_p_th_Middle_f = run_af_FR_p_th_Middle.flatten()
	run_af_FR_p_th_Middle_f_nz = run_af_FR_p_th_Middle_f[np.nonzero(run_af_FR_p_th_Middle_f)]
	#
	[bins_circ, kde_circ] = vonmises_kde(run_af_FR_p_th_Middle_f_nz, kappa, n_bins)  
	#
	kde_FR_p_Middle = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		run_af_t_temp = run_af_FR_p_t_Middle[kk,:]
		run_af_th_temp = run_af_FR_p_th_Middle[kk,:]
		#
		run_af_t_temp_z = run_af_t_temp
		#
		run_af_t_temp_nz = run_af_t_temp[np.nonzero(run_af_t_temp_z)]
		run_af_th_temp_nz = run_af_th_temp[np.nonzero(run_af_t_temp_z)]	
		if len(run_af_th_temp_nz) > min_mid:
			# wrapping
			run_af_th_temp_nz = (run_af_th_temp_nz + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_FR_p_Middle[kk,:]] = vonmises_kde(run_af_th_temp_nz, kappa, n_bins) 

	kde_FR_p_Middle_m = np.zeros(len(kde_circ))
	kde_FR_p_Middle_u = np.zeros(len(kde_circ))
	kde_FR_p_Middle_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_FR_p_Middle_m)):
		kde_circ_temp = kde_FR_p_Middle[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_FR_p_Middle_m[ii], kde_FR_p_Middle_l[ii], kde_FR_p_Middle_u[ii]] = mean_confidence_interval(kde_circ_temp)
				
	kde_max = np.max(kde_FR_p_Middle_u)
	kde_min = np.min(kde_FR_p_Middle_l)
	#
	fig = plt.figure(601)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure6F, s2_Figure6F)
	#
	ax.plot(bins_circ_af_temp, kde_FR_p_Middle_m, color = (r2, g2, b2), alpha = 1)
	ax.fill_between(bins_circ_af_temp, kde_FR_p_Middle_u, kde_FR_p_Middle_l, color = (r2, g2, b2), alpha = 0.5, linewidth = 0)
	#	
	adjust_spines(ax, ['left', 'Middle'])
	#

	# Top
	NN = np.shape(run_af_FR_p_th_Top)[0]
	run_af_FR_p_th_Top_f = run_af_FR_p_th_Top.flatten()
	run_af_FR_p_th_Top_f_nz = run_af_FR_p_th_Top_f[np.nonzero(run_af_FR_p_th_Top_f)]
	#
	[bins_circ, kde_circ] = vonmises_kde(run_af_FR_p_th_Top_f_nz, kappa, n_bins)  
	#
	kde_FR_p_Top = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		run_af_t_temp = run_af_FR_p_t_Top[kk,:]
		run_af_th_temp = run_af_FR_p_th_Top[kk,:]
		#
		run_af_t_temp_z = run_af_t_temp
		#
		run_af_t_temp_nz = run_af_t_temp[np.nonzero(run_af_t_temp_z)]
		run_af_th_temp_nz = run_af_th_temp[np.nonzero(run_af_t_temp_z)]	
		if len(run_af_th_temp_nz) > min_mid:
			# wrapping
			run_af_th_temp_nz = (run_af_th_temp_nz + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_FR_p_Top[kk,:]] = vonmises_kde(run_af_th_temp_nz, kappa, n_bins) 

	kde_FR_p_Top_m = np.zeros(len(kde_circ))
	kde_FR_p_Top_u = np.zeros(len(kde_circ))
	kde_FR_p_Top_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_FR_p_Top_m)):
		kde_circ_temp = kde_FR_p_Top[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_FR_p_Top_m[ii], kde_FR_p_Top_l[ii], kde_FR_p_Top_u[ii]] = mean_confidence_interval(kde_circ_temp)
				
	kde_max = np.max(kde_FR_p_Top_u)
	kde_min = np.min(kde_FR_p_Top_l)
	#
	fig = plt.figure(601)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure6F, s2_Figure6F)
	#
	ax.plot(bins_circ_af_temp, kde_FR_p_Top_m, color = (r1, g1, b1), alpha = 1)
	ax.fill_between(bins_circ_af_temp, kde_FR_p_Top_u, kde_FR_p_Top_l, color = (r1, g1, b1), alpha = 0.5, linewidth = 0)
	#	
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 2.2])
	plt.yticks([0, 1.1, 2.2])
	#
	adjust_spines(ax, ['left', 'Top'])
	#
	fig.savefig('../Plots/Figure6_plots/Figure6_F_KDE_run_mid_FR_p_'+str(kappa)+'.pdf')


##### panel GHI sample trace CR
if panel_GHI == 1:
	s1_Figure6GHI = 2.25
	s2_Figure6GHI = 1.5

	Name_data = '../Data/simulations/modeling_extracted_data/CR_extracted_data/CR_one_shot/'
	dur_du = 3
	dur_af = 1
	dur = dur_du + dur_af
	#
	lw = 0.5
	al = 1

	ind_CR_b = [131]
	ind_CR_m = [93]
	ind_CR_t = [94]
	# bottom
	for kk in range(len(ind_CR_b)):
		t_du_temp = np.loadtxt(Name_data+'t_du_CR_1_'+str(ind_CR_b[kk])+'.txt')
		theta_du_temp = np.loadtxt(Name_data+'theta_du_CR_1_'+str(ind_CR_b[kk])+'.txt')
		#
		t_af_temp = np.loadtxt(Name_data+'t_af_CR_1_'+str(ind_CR_b[kk])+'.txt')
		theta_af_temp = np.loadtxt(Name_data+'theta_af_CR_1_'+str(ind_CR_b[kk])+'.txt')
		#
		#
		ind_du_s = find_nearest(t_du_temp, t_du_temp[-1] - dur_du*60)
		ind_af_e = find_nearest(t_af_temp, t_af_temp[0] + dur_af*60)
		#
		t_du_temp = t_du_temp[ind_du_s:]
		theta_du_temp = theta_du_temp[ind_du_s:]
		#
		t_af_temp = t_af_temp[0:ind_af_e]
		theta_af_temp = theta_af_temp[0:ind_af_e]

		t_temp = np.hstack([t_du_temp, t_af_temp])
		theta_temp = np.hstack([theta_du_temp, theta_af_temp])
		#
		fig = plt.figure(701+kk)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_Figure6GHI, s2_Figure6GHI)
		#
		ax.plot((t_temp - t_temp[0])/60, theta_temp*sc, color = 'k', linewidth = lw, alpha = al)
		#
		plt.xlim([0,dur])
		plt.ylim([-26,26])
		#
		plt.xticks(dur*np.arange(2))
		plt.yticks(13*np.arange(5)-26)
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/Figure6_plots/Figure6_G_CR_bottom_'+str(ind_CR_b[kk])+'.pdf')

	# middle
	for kk in range(len(ind_CR_m)):
		t_du_temp = np.loadtxt(Name_data+'t_du_CR_2_'+str(ind_CR_m[kk])+'.txt')
		theta_du_temp = np.loadtxt(Name_data+'theta_du_CR_2_'+str(ind_CR_m[kk])+'.txt')
		#
		t_af_temp = np.loadtxt(Name_data+'t_af_CR_2_'+str(ind_CR_m[kk])+'.txt')
		theta_af_temp = np.loadtxt(Name_data+'theta_af_CR_2_'+str(ind_CR_m[kk])+'.txt')
		#
		#
		ind_du_s = find_nearest(t_du_temp, t_du_temp[-1] - dur_du*60)
		ind_af_e = find_nearest(t_af_temp, t_af_temp[0] + dur_af*60)
		#
		t_du_temp = t_du_temp[ind_du_s:]
		theta_du_temp = theta_du_temp[ind_du_s:]
		#
		t_af_temp = t_af_temp[0:ind_af_e]
		theta_af_temp = theta_af_temp[0:ind_af_e]

		t_temp = np.hstack([t_du_temp, t_af_temp])
		theta_temp = np.hstack([theta_du_temp, theta_af_temp])
		#
		fig = plt.figure(801+kk)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_Figure6GHI, s2_Figure6GHI)
		#
		ax.plot((t_temp - t_temp[0])/60, theta_temp*sc, color = 'k', linewidth = lw, alpha = al)
		#
		plt.xlim([0,dur])
		plt.ylim([-26,26])
		#
		plt.xticks(dur*np.arange(2))
		plt.yticks(13*np.arange(5)-26)
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/Figure6_plots/Figure6_H_CR_middle_'+str(ind_CR_m[kk])+'.pdf')


	# top
	for kk in range(len(ind_CR_t)):
		t_du_temp = np.loadtxt(Name_data+'t_du_CR_3_'+str(ind_CR_t[kk])+'.txt')
		theta_du_temp = np.loadtxt(Name_data+'theta_du_CR_3_'+str(ind_CR_t[kk])+'.txt')
		#
		t_af_temp = np.loadtxt(Name_data+'t_af_CR_3_'+str(ind_CR_t[kk])+'.txt')
		theta_af_temp = np.loadtxt(Name_data+'theta_af_CR_3_'+str(ind_CR_t[kk])+'.txt')
		#
		#
		ind_du_s = find_nearest(t_du_temp, t_du_temp[-1] - dur_du*60)
		ind_af_e = find_nearest(t_af_temp, t_af_temp[0] + dur_af*60)
		#
		t_du_temp = t_du_temp[ind_du_s:]
		theta_du_temp = theta_du_temp[ind_du_s:]
		#
		t_af_temp = t_af_temp[0:ind_af_e]
		theta_af_temp = theta_af_temp[0:ind_af_e]

		t_temp = np.hstack([t_du_temp, t_af_temp])
		theta_temp = np.hstack([theta_du_temp, theta_af_temp])
		#
		fig = plt.figure(901+kk)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_Figure6GHI, s2_Figure6GHI)
		#
		ax.plot((t_temp - t_temp[0])/60, theta_temp*sc, color = 'k', linewidth = lw, alpha = al)
		#
		plt.xlim([0,dur])
		plt.ylim([-26,26])
		#
		plt.xticks(dur*np.arange(2))
		plt.yticks(13*np.arange(5)-26)
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/Figure6_plots/Figure6_I_CR_near_'+str(ind_CR_t[kk])+'.pdf')

#### KDE run midpoints CR
if panel_J == 1:
	s1_Figure6J = 2
	s2_Figure6J = 1
	#
	Name = '../Data/simulations/modeling_extracted_data/'
	run_af_CR_t_Bottom = np.loadtxt(Name+'CR_extracted_data/CR_one_shot_population/run_af_CR_t_Bottom.txt')
	run_af_CR_t_Middle = np.loadtxt(Name+'CR_extracted_data/CR_one_shot_population/run_af_CR_t_Middle.txt')
	run_af_CR_t_Top = np.loadtxt(Name+'CR_extracted_data/CR_one_shot_population/run_af_CR_t_Top.txt')
	#
	run_af_CR_th_Bottom = np.loadtxt(Name+'CR_extracted_data/CR_one_shot_population/run_af_CR_th_Bottom.txt')
	run_af_CR_th_Middle = np.loadtxt(Name+'CR_extracted_data/CR_one_shot_population/run_af_CR_th_Middle.txt')
	run_af_CR_th_Top = np.loadtxt(Name+'CR_extracted_data/CR_one_shot_population/run_af_CR_th_Top.txt')

	#

	# Bottom
	NN = np.shape(run_af_CR_th_Bottom)[0]
	run_af_CR_th_Bottom_f = run_af_CR_th_Bottom.flatten()
	run_af_CR_th_Bottom_f_nz = run_af_CR_th_Bottom_f[np.nonzero(run_af_CR_th_Bottom_f)]
	#
	[bins_circ, kde_circ] = vonmises_kde(run_af_CR_th_Bottom_f_nz, kappa, n_bins)  
	#
	kde_CR_Bottom = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		run_af_t_temp = run_af_CR_t_Bottom[kk,:]
		run_af_th_temp = run_af_CR_th_Bottom[kk,:]
		#
		run_af_t_temp_z = run_af_t_temp
		#
		run_af_t_temp_nz = run_af_t_temp[np.nonzero(run_af_t_temp_z)]
		run_af_th_temp_nz = run_af_th_temp[np.nonzero(run_af_t_temp_z)]	
		if len(run_af_th_temp_nz) > min_mid:
			# wrapping
			run_af_th_temp_nz = (run_af_th_temp_nz + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_CR_Bottom[kk,:]] = vonmises_kde(run_af_th_temp_nz, kappa, n_bins) 

	kde_CR_Bottom_m = np.zeros(len(kde_circ))
	kde_CR_Bottom_u = np.zeros(len(kde_circ))
	kde_CR_Bottom_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_CR_Bottom_m)):
		kde_circ_temp = kde_CR_Bottom[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_CR_Bottom_m[ii], kde_CR_Bottom_l[ii], kde_CR_Bottom_u[ii]] = mean_confidence_interval(kde_circ_temp)
				
	kde_max = np.max(kde_CR_Bottom_u)
	kde_min = np.min(kde_CR_Bottom_l)
	#
	fig = plt.figure(1001)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure6J, s2_Figure6J)
	#
	ax.plot(bins_circ_af_temp, kde_CR_Bottom_m, color = (r3, g3, b3), alpha = 1)
	ax.fill_between(bins_circ_af_temp, kde_CR_Bottom_u, kde_CR_Bottom_l, color = (r3, g3, b3), alpha = 0.5, linewidth = 0)
	# 
	adjust_spines(ax, ['left', 'bottom'])


	# Middle
	NN = np.shape(run_af_CR_th_Middle)[0]
	run_af_CR_th_Middle_f = run_af_CR_th_Middle.flatten()
	run_af_CR_th_Middle_f_nz = run_af_CR_th_Middle_f[np.nonzero(run_af_CR_th_Middle_f)]
	#
	[bins_circ, kde_circ] = vonmises_kde(run_af_CR_th_Middle_f_nz, kappa, n_bins)  
	#
	kde_CR_Middle = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		run_af_t_temp = run_af_CR_t_Middle[kk,:]
		run_af_th_temp = run_af_CR_th_Middle[kk,:]
		#
		run_af_t_temp_z = run_af_t_temp
		#
		run_af_t_temp_nz = run_af_t_temp[np.nonzero(run_af_t_temp_z)]
		run_af_th_temp_nz = run_af_th_temp[np.nonzero(run_af_t_temp_z)]	
		if len(run_af_th_temp_nz) > min_mid:
			# wrapping
			run_af_th_temp_nz = (run_af_th_temp_nz + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_CR_Middle[kk,:]] = vonmises_kde(run_af_th_temp_nz, kappa, n_bins) 

	kde_CR_Middle_m = np.zeros(len(kde_circ))
	kde_CR_Middle_u = np.zeros(len(kde_circ))
	kde_CR_Middle_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_CR_Middle_m)):
		kde_circ_temp = kde_CR_Middle[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_CR_Middle_m[ii], kde_CR_Middle_l[ii], kde_CR_Middle_u[ii]] = mean_confidence_interval(kde_circ_temp)
				
	kde_max = np.max(kde_CR_Middle_u)
	kde_min = np.min(kde_CR_Middle_l)
	#
	fig = plt.figure(1001)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure6J, s2_Figure6J)
	#
	ax.plot(bins_circ_af_temp, kde_CR_Middle_m, color = (r2, g2, b2), alpha = 1)
	ax.fill_between(bins_circ_af_temp, kde_CR_Middle_u, kde_CR_Middle_l, color = (r2, g2, b2), alpha = 0.5, linewidth = 0)
	#	
	adjust_spines(ax, ['left', 'Middle'])
	#

	# Top
	NN = np.shape(run_af_CR_th_Top)[0]
	run_af_CR_th_Top_f = run_af_CR_th_Top.flatten()
	run_af_CR_th_Top_f_nz = run_af_CR_th_Top_f[np.nonzero(run_af_CR_th_Top_f)]
	#
	[bins_circ, kde_circ] = vonmises_kde(run_af_CR_th_Top_f_nz, kappa, n_bins)  
	#
	kde_CR_Top = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		run_af_t_temp = run_af_CR_t_Top[kk,:]
		run_af_th_temp = run_af_CR_th_Top[kk,:]
		#
		run_af_t_temp_z = run_af_t_temp
		#
		run_af_t_temp_nz = run_af_t_temp[np.nonzero(run_af_t_temp_z)]
		run_af_th_temp_nz = run_af_th_temp[np.nonzero(run_af_t_temp_z)]	
		if len(run_af_th_temp_nz) > min_mid:
			# wrapping
			run_af_th_temp_nz = (run_af_th_temp_nz + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_CR_Top[kk,:]] = vonmises_kde(run_af_th_temp_nz, kappa, n_bins) 

	kde_CR_Top_m = np.zeros(len(kde_circ))
	kde_CR_Top_u = np.zeros(len(kde_circ))
	kde_CR_Top_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_CR_Top_m)):
		kde_circ_temp = kde_CR_Top[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_CR_Top_m[ii], kde_CR_Top_l[ii], kde_CR_Top_u[ii]] = mean_confidence_interval(kde_circ_temp)
				
	kde_max = np.max(kde_CR_Top_u)
	kde_min = np.min(kde_CR_Top_l)
	#
	fig = plt.figure(1001)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure6J, s2_Figure6J)
	#
	ax.plot(bins_circ_af_temp, kde_CR_Top_m, color = (r1, g1, b1), alpha = 1)
	ax.fill_between(bins_circ_af_temp, kde_CR_Top_u, kde_CR_Top_l, color = (r1, g1, b1), alpha = 0.5, linewidth = 0)
	# plt.plot(bins_1F, kde_circ, color = 'cyan')
	#	
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 2.2])
	plt.yticks([0, 1.1, 2.2])
	#
	adjust_spines(ax, ['left', 'Top'])
	#
	fig.savefig('../Plots/Figure6_plots/Figure6_J_KDE_run_mid_CR_'+str(kappa)+'.pdf')

#################### fly data
##### panels K,L,M sample trace -- fly data
if panel_KLM == 1:
	s1_Figure6KLM = 2.25
	s2_Figure6KLM = 1.5

	dur_du = 3
	dur_af = 1
	dur = dur_du + dur_af
	#
	lw = 0.5
	al = 1

	rr = [2]
	fly_b = [26]
	tri_b = [9]
	# bottom
	for kk in range(len(fly_b)):
		t_temp_du = np.loadtxt('../Data/real_fly_data/One_shot/time_theta/t_round'+str(rr[kk])+'_du'+str(tri_b[kk])+'_'+str(fly_b[kk])+'.txt')
		theta_temp_du = np.loadtxt('../Data/real_fly_data/One_shot/time_theta/theta_round'+str(rr[kk])+'_du'+str(tri_b[kk])+'_'+str(fly_b[kk])+'.txt')
		#
		t_temp_af = np.loadtxt('../Data/real_fly_data/One_shot/time_theta/t_round'+str(rr[kk])+'_af'+str(tri_b[kk])+'_'+str(fly_b[kk])+'.txt')
		theta_temp_af = np.loadtxt('../Data/real_fly_data/One_shot/time_theta/theta_round'+str(rr[kk])+'_af'+str(tri_b[kk])+'_'+str(fly_b[kk])+'.txt')
		#
		ind_du_s = find_nearest(t_temp_du, t_temp_du[-1] - dur_du)
		t_temp_du = t_temp_du[ind_du_s:]
		theta_temp_du = theta_temp_du[ind_du_s:]
		#
		ind_af_e = find_nearest(t_temp_af, t_temp_af[0] + dur_af)
		t_temp_af = t_temp_af[0:ind_af_e]
		theta_temp_af = theta_temp_af[0:ind_af_e]

		fig = plt.figure(1101+kk)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_Figure6KLM, s2_Figure6KLM)
		#
		ax.plot(t_temp_du - t_temp_du[0], theta_temp_du*sc, color = 'k', linewidth = lw, alpha = al)
		ax.plot(t_temp_af - t_temp_du[0], theta_temp_af*sc, color = 'k', linewidth = lw, alpha = al)
		#
		plt.xlim([0,dur])
		plt.ylim([-26,26])
		#
		plt.xticks(dur*np.arange(2))
		plt.yticks(13*np.arange(5)-26)
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/Figure6_plots/Figure6_K_fly_bottom_round'+str(rr[kk])+'_'+str(fly_b[kk])+'_'+str(tri_b[kk])+'.pdf')

	rr = [2]
	fly_m = [26]
	tri_m = [5]
	# middle
	for kk in range(len(fly_m)):
		t_temp_du = np.loadtxt('../Data/real_fly_data/One_shot/time_theta/t_round'+str(rr[kk])+'_du'+str(tri_m[kk])+'_'+str(fly_m[kk])+'.txt')
		theta_temp_du = np.loadtxt('../Data/real_fly_data/One_shot/time_theta/theta_round'+str(rr[kk])+'_du'+str(tri_m[kk])+'_'+str(fly_m[kk])+'.txt')
		#
		t_temp_af = np.loadtxt('../Data/real_fly_data/One_shot/time_theta/t_round'+str(rr[kk])+'_af'+str(tri_m[kk])+'_'+str(fly_m[kk])+'.txt')
		theta_temp_af = np.loadtxt('../Data/real_fly_data/One_shot/time_theta/theta_round'+str(rr[kk])+'_af'+str(tri_m[kk])+'_'+str(fly_m[kk])+'.txt')
		#
		ind_du_s = find_nearest(t_temp_du, t_temp_du[-1] - dur_du)
		t_temp_du = t_temp_du[ind_du_s:]
		theta_temp_du = theta_temp_du[ind_du_s:]
		#
		ind_af_e = find_nearest(t_temp_af, t_temp_af[0] + dur_af)
		t_temp_af = t_temp_af[0:ind_af_e]
		theta_temp_af = theta_temp_af[0:ind_af_e]

		fig = plt.figure(1201+kk)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_Figure6KLM, s2_Figure6KLM)
		#
		ax.plot(t_temp_du - t_temp_du[0], theta_temp_du*sc, color = 'k', linewidth = lw, alpha = al)
		ax.plot(t_temp_af - t_temp_du[0], theta_temp_af*sc, color = 'k', linewidth = lw, alpha = al)
		#
		plt.xlim([0,dur])
		plt.ylim([-26,26])
		#
		plt.xticks(dur*np.arange(2))
		plt.yticks(13*np.arange(5)-26)
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/Figure6_plots/Figure6_L_fly_middle_round'+str(rr[kk])+'_'+str(fly_m[kk])+'_'+str(tri_m[kk])+'.pdf')

	rr = [2]
	fly_t = [26]
	tri_t = [8]
	# top
	for kk in range(len(fly_t)):
		t_temp_du = np.loadtxt('../Data/real_fly_data/One_shot/time_theta/t_round'+str(rr[kk])+'_du'+str(tri_t[kk])+'_'+str(fly_t[kk])+'.txt')
		theta_temp_du = np.loadtxt('../Data/real_fly_data/One_shot/time_theta/theta_round'+str(rr[kk])+'_du'+str(tri_t[kk])+'_'+str(fly_t[kk])+'.txt')
		#
		t_temp_af = np.loadtxt('../Data/real_fly_data/One_shot/time_theta/t_round'+str(rr[kk])+'_af'+str(tri_t[kk])+'_'+str(fly_t[kk])+'.txt')
		theta_temp_af = np.loadtxt('../Data/real_fly_data/One_shot/time_theta/theta_round'+str(rr[kk])+'_af'+str(tri_t[kk])+'_'+str(fly_t[kk])+'.txt')
		#
		ind_du_s = find_nearest(t_temp_du, t_temp_du[-1] - dur_du)
		t_temp_du = t_temp_du[ind_du_s:]
		theta_temp_du = theta_temp_du[ind_du_s:]
		#
		ind_af_e = find_nearest(t_temp_af, t_temp_af[0] + dur_af)
		t_temp_af = t_temp_af[0:ind_af_e]
		theta_temp_af = theta_temp_af[0:ind_af_e]

		fig = plt.figure(1301+kk)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_Figure6KLM, s2_Figure6KLM)
		#
		ax.plot(t_temp_du - t_temp_du[0], theta_temp_du*sc, color = 'k', linewidth = lw, alpha = al)
		ax.plot(t_temp_af - t_temp_du[0], theta_temp_af*sc, color = 'k', linewidth = lw, alpha = al)
		#
		plt.xlim([0,dur])
		plt.ylim([-26,26])
		#
		plt.xticks(dur*np.arange(2))
		plt.yticks(13*np.arange(5)-26)
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/Figure6_plots/Figure6_M_fly_top_round'+str(rr[kk])+'_'+str(fly_t[kk])+'_'+str(tri_t[kk])+'.pdf')


######### KDE run mid point real fly from two rounds
if panel_N == 1:
	s1_Figure6N = 2
	s2_Figure6N = 1
	#
	Name = '../Data/real_fly_data/One_shot/One_shot_rev_run/'
	#### round 1
	fly_num_round1 = np.loadtxt(Name+'One_shot_rev_run_round1/fly_num.txt')

	# runs
	run_af_t_Bottom_round1 = np.loadtxt(Name+'One_shot_rev_run_round1/run_af_t_Bottom.txt')
	run_af_th_Bottom_round1 = np.loadtxt(Name+'One_shot_rev_run_round1/run_af_th_Bottom.txt')
	#
	run_af_t_Middle_round1 = np.loadtxt(Name+'One_shot_rev_run_round1/run_af_t_Middle.txt')
	run_af_th_Middle_round1 = np.loadtxt(Name+'One_shot_rev_run_round1/run_af_th_Middle.txt')
	#
	run_af_t_Top_round1 = np.loadtxt(Name+'One_shot_rev_run_round1/run_af_t_Top.txt')
	run_af_th_Top_round1 = np.loadtxt(Name+'One_shot_rev_run_round1/run_af_th_Top.txt')

	#### round 1
	fly_num_round2 = np.loadtxt(Name+'One_shot_rev_run_round2/fly_num.txt')

	# runs
	run_af_t_Bottom_round2 = np.loadtxt(Name+'One_shot_rev_run_round2/run_af_t_Bottom.txt')
	run_af_th_Bottom_round2 = np.loadtxt(Name+'One_shot_rev_run_round2/run_af_th_Bottom.txt')
	#
	run_af_t_Middle_round2 = np.loadtxt(Name+'One_shot_rev_run_round2/run_af_t_Middle.txt')
	run_af_th_Middle_round2 = np.loadtxt(Name+'One_shot_rev_run_round2/run_af_th_Middle.txt')
	#
	run_af_t_Top_round2 = np.loadtxt(Name+'One_shot_rev_run_round2/run_af_t_Top.txt')
	run_af_th_Top_round2 = np.loadtxt(Name+'One_shot_rev_run_round2/run_af_th_Top.txt')

	# stacking two rounds
	fly_num = np.vstack([fly_num_round1, fly_num_round2])
	#
	run_af_th_Bottom = np.vstack([run_af_th_Bottom_round1, run_af_th_Bottom_round2])
	run_af_th_Middle = np.vstack([run_af_th_Middle_round1, run_af_th_Middle_round2])
	run_af_th_Top = np.vstack([run_af_th_Top_round1, run_af_th_Top_round2])

	#
	NN = np.shape(fly_num)[0]
	#
	run_Bottom_fly = np.zeros((NN,100))
	run_Middle_fly = np.zeros((NN,100))
	run_Top_fly = np.zeros((NN,100))

	for kk in range(NN):
		# Bottom
		run_temp = run_af_th_Bottom[12*kk+0:12*kk+12,:]
		run_temp_f = run_temp.flatten()
		run_temp_f_nz = run_temp_f[np.nonzero(run_temp_f)]
		#
		for yy in range(len(run_temp_f_nz)):
			run_Bottom_fly[kk,yy] = run_temp_f_nz[yy]

		# Middle
		run_temp = run_af_th_Middle[12*kk+0:12*kk+12,:]
		run_temp_f = run_temp.flatten()
		run_temp_f_nz = run_temp_f[np.nonzero(run_temp_f)]
		#
		for yy in range(len(run_temp_f_nz)):
			run_Middle_fly[kk,yy] = run_temp_f_nz[yy]
			
		# Top
		run_temp = run_af_th_Top[12*kk+0:12*kk+12,:]
		run_temp_f = run_temp.flatten()
		run_temp_f_nz = run_temp_f[np.nonzero(run_temp_f)]
		#
		for yy in range(len(run_temp_f_nz)):
			run_Top_fly[kk,yy] = run_temp_f_nz[yy]		



	########### Bottom
	NN = np.shape(run_Bottom_fly)[0]
	#
	run_Bottom_fly_b = run_Bottom_fly.flatten()
	run_Bottom_fly_b_nz = run_Bottom_fly_b[np.nonzero(run_Bottom_fly_b)]
	#
	[bins_circ, kde_circ] = vonmises_kde(run_Bottom_fly_b_nz, kappa, n_bins)  
	#
	kde_Bottom = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		run_th_temp = run_Bottom_fly[kk,:]
		#
		run_th_temp = run_th_temp[np.nonzero(run_th_temp)]	
		if len(run_th_temp) > min_mid:
			# wrapping
			run_th_temp = (run_th_temp + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_Bottom[kk,:]] = vonmises_kde(run_th_temp, kappa, n_bins) 

	kde_Bottom_m = np.zeros(len(kde_circ))
	kde_Bottom_u = np.zeros(len(kde_circ))
	kde_Bottom_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_Bottom_m)):
		kde_circ_temp = kde_Bottom[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_Bottom_m[ii], kde_Bottom_l[ii], kde_Bottom_u[ii]] = mean_confidence_interval(kde_circ_temp)
				
	kde_max = np.max(kde_Bottom_u)
	kde_min = np.min(kde_Bottom_l)
	#
	fig = plt.figure(1401)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure6N, s2_Figure6N)
	#
	ax.plot(bins_circ_af_temp, kde_Bottom_m, color = (r3, g3, b3), alpha = 0.5)
	ax.fill_between(bins_circ_af_temp, kde_Bottom_u, kde_Bottom_l, color = (r3, g3, b3), alpha = 0.5, linewidth = 0)
	#
	adjust_spines(ax, ['left', 'bottom'])
	#

	########### Middle
	NN = np.shape(run_Middle_fly)[0]
	#
	run_Middle_fly_b = run_Middle_fly.flatten()
	run_Middle_fly_b_nz = run_Middle_fly_b[np.nonzero(run_Middle_fly_b)]
	#
	[bins_circ, kde_circ] = vonmises_kde(run_Middle_fly_b_nz, kappa, n_bins)  
	#
	kde_Middle = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		run_th_temp = run_Middle_fly[kk,:]
		#
		run_th_temp = run_th_temp[np.nonzero(run_th_temp)]	
		if len(run_th_temp) > min_mid:
			# wrapping
			run_th_temp = (run_th_temp + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_Middle[kk,:]] = vonmises_kde(run_th_temp, kappa, n_bins) 

	kde_Middle_m = np.zeros(len(kde_circ))
	kde_Middle_u = np.zeros(len(kde_circ))
	kde_Middle_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_Middle_m)):
		kde_circ_temp = kde_Middle[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_Middle_m[ii], kde_Middle_l[ii], kde_Middle_u[ii]] = mean_confidence_interval(kde_circ_temp)
				
	kde_max = np.max(kde_Middle_u)
	kde_min = np.min(kde_Middle_l)
	#
	fig = plt.figure(1401)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure6N, s2_Figure6N)
	#
	ax.plot(bins_circ_af_temp, kde_Middle_m, color = (r2, g2, b2), alpha = 0.5)
	ax.fill_between(bins_circ_af_temp, kde_Middle_u, kde_Middle_l, color = (r2, g2, b2), alpha = 0.5, linewidth = 0)
	#
	adjust_spines(ax, ['left', 'bottom'])
	#

	########### Top
	NN = np.shape(run_Top_fly)[0]
	#
	run_Top_fly_b = run_Top_fly.flatten()
	run_Top_fly_b_nz = run_Top_fly_b[np.nonzero(run_Top_fly_b)]
	#
	[bins_circ, kde_circ] = vonmises_kde(run_Top_fly_b_nz, kappa, n_bins)  
	#
	kde_Top = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		run_th_temp = run_Top_fly[kk,:]
		#
		run_th_temp = run_th_temp[np.nonzero(run_th_temp)]	
		if len(run_th_temp) > min_mid:
			# wrapping
			run_th_temp = (run_th_temp + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_Top[kk,:]] = vonmises_kde(run_th_temp, kappa, n_bins) 

	kde_Top_m = np.zeros(len(kde_circ))
	kde_Top_u = np.zeros(len(kde_circ))
	kde_Top_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_Top_m)):
		kde_circ_temp = kde_Top[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_Top_m[ii], kde_Top_l[ii], kde_Top_u[ii]] = mean_confidence_interval(kde_circ_temp)
				
	kde_max = np.max(kde_Top_u)
	kde_min = np.min(kde_Top_l)
	#

	fig = plt.figure(1401)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure6N, s2_Figure6N)
	#
	ax.plot(bins_circ_af_temp, kde_Top_m, color = (r1, g1, b1), alpha = 0.5)
	ax.fill_between(bins_circ_af_temp, kde_Top_u, kde_Top_l, color = (r1, g1, b1), alpha = 0.5, linewidth = 0)
	#	
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 2.2])
	plt.yticks([0, 1.1, 2.2])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure6_plots/Figure6_N_KDE_run_mid_fly_all_'+str(kappa)+'.pdf')


