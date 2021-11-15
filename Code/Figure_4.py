############ Figure 4
# The FR model fails to predict Drosophila search behavior around multiple fictive food sites.

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
#
panel_B = 1
#
panel_C = 1
#
panel_D = 1
#


##############
cond = 2 # 2F_60
#
up = 0.2
DEF_size = 1.1
#
kappa = 200
n_bins = 2000
min_mid = 3


##### panel A real fly sample trace
if panel_A == 1:
	s1_Figure4A = 0.9*5.7/1.25
	s2_Figure4A = 0.9*2.3/1.1

	lw = 0.25 
	al = 1
	#
	fly_2F_60 = [14]
	#
	####
	for kk in range(len(fly_2F_60)):
		t_be_temp = np.loadtxt('../Data/real_fly_data/Non_trial_two_foods/2F_60/time_theta/time_theta_be/t_be_2_'+str(fly_2F_60[kk])+'.txt')
		theta_be_temp = np.loadtxt('../Data/real_fly_data/Non_trial_two_foods/2F_60/time_theta/time_theta_be/theta_be_2_'+str(fly_2F_60[kk])+'.txt')
		#
		t_du_temp = np.loadtxt('../Data/real_fly_data/Non_trial_two_foods/2F_60/time_theta/time_theta_du/t_du_2_'+str(fly_2F_60[kk])+'.txt')
		theta_du_temp = np.loadtxt('../Data/real_fly_data/Non_trial_two_foods/2F_60/time_theta/time_theta_du/theta_du_2_'+str(fly_2F_60[kk])+'.txt')
		#
		t_af_temp = np.loadtxt('../Data/real_fly_data/Non_trial_two_foods/2F_60/time_theta/time_theta_af/t_af_2_'+str(fly_2F_60[kk])+'.txt')
		theta_af_temp = np.loadtxt('../Data/real_fly_data/Non_trial_two_foods/2F_60/time_theta/time_theta_af/theta_af_2_'+str(fly_2F_60[kk])+'.txt')
		#
		LED_time = np.loadtxt('../Data/real_fly_data/Non_trial_two_foods/2F_60/LED_time/LED_time_2_'+str(fly_2F_60[kk])+'.txt')
		LED_time = LED_time[0::2]
		LED_time = LED_time[LED_time < 51]
		
		t_all = np.hstack([t_be_temp, t_du_temp, t_af_temp])
		theta_all = np.hstack([theta_be_temp, theta_du_temp, theta_af_temp])

		ind_s = find_nearest(t_all, 8)
		ind_e = find_nearest(t_all, 55)
		#
		t_all = t_all[ind_s:ind_e]
		theta_all = theta_all[ind_s:ind_e]


		fig = plt.figure(201+kk)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_Figure4A, s2_Figure4A)
		#
		ax.plot(t_all, theta_all*sc, color = 'k', linewidth = lw, alpha = al)
		#
		for uu in range(len(LED_time)):
			ax.plot(LED_time[uu]*np.ones(2), np.linspace(25.5,26,2), color = 'r', linewidth = 0.5)
		#
		plt.xlim([8,55])
		plt.ylim([-26,26])
		#
		plt.xticks([])
		plt.yticks(13*np.arange(5)-26)
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/Figure4_plots/Figure4_A_sample_real_fly_2F_60_'+str(fly_2F_60[kk])+'.pdf')

##### real fly data 2F_60
if panel_B == 1:
	s1_Figure4B = 1.1*0.9
	s2_Figure4B = 0.8
	
	# loading data
	RE_t_du1_fly_2F_60 = np.loadtxt('../Data/real_fly_data/Non_trial_two_foods/2F_60/reversals/RE_t_du1_fly_2F_60.txt')
	RE_th_du1_fly_2F_60 = np.loadtxt('../Data/real_fly_data/Non_trial_two_foods/2F_60/reversals/RE_th_du1_fly_2F_60.txt')
	#
	RE_t_du2_fly_2F_60 = np.loadtxt('../Data/real_fly_data/Non_trial_two_foods/2F_60/reversals/RE_t_du2_fly_2F_60.txt')
	RE_th_du2_fly_2F_60 = np.loadtxt('../Data/real_fly_data/Non_trial_two_foods/2F_60/reversals/RE_th_du2_fly_2F_60.txt')
	#
	RE_t_af_fly_2F_60 = np.loadtxt('../Data/real_fly_data/Non_trial_two_foods/2F_60/reversals/RE_t_af1_fly_2F_60.txt')
	RE_th_af_fly_2F_60 = np.loadtxt('../Data/real_fly_data/Non_trial_two_foods/2F_60/reversals/RE_th_af1_fly_2F_60.txt')
	#
	which_food_2F_60 = np.loadtxt('../Data/real_fly_data/Non_trial_two_foods/2F_60/reversals/which_food_2F_60.txt')

	
	NN = np.shape(RE_t_du1_fly_2F_60)[0]
	#
	min_RE = 5
	#
	mid_2F_60_du1 = np.zeros((NN,1000))
	#
	mid_2F_60_du2 = np.zeros((NN,1000))
	#
	mid_2F_60_du2_fil = np.zeros((NN,1000))
	#
	mid_2F_60_af = np.zeros((NN,1000))
	#
	mid_2F_60_du1_avg = np.zeros(NN)
	mid_2F_60_du2_avg = np.zeros(NN)
	mid_2F_60_du2_fil_avg = np.zeros(NN)
	mid_2F_60_af_avg = np.zeros(NN)
	#
	ind_du1 = np.zeros(NN, dtype = int)
	ind_du2 = np.zeros(NN, dtype = int)
	ind_du2_fil = np.zeros(NN, dtype = int)
	ind_af = np.zeros(NN, dtype = int)
	#
	co_du1 = 0
	co_du2 = 0
	co_du2_fil = 0
	co_af = 0
	#
	for kk in range(NN):
		####### du1
		rev_th_temp = RE_th_du1_fly_2F_60[kk,:]
		if which_food_2F_60[kk,0] == 1:
			rev_th_temp = -rev_th_temp[np.nonzero(rev_th_temp)]
		if which_food_2F_60[kk,0] == -1:
			rev_th_temp = rev_th_temp[np.nonzero(rev_th_temp)]
		#
		if len(rev_th_temp) > min_RE:
			ind_du1[co_du1] = kk
			co_du1 = co_du1 + 1
			for hh in range(len(rev_th_temp)-1):
				mid_2F_60_du1[kk,hh] = 0.5*(rev_th_temp[hh+1] + rev_th_temp[hh])
			#

		####### du2
		rev_th_temp = RE_th_du2_fly_2F_60[kk,:]
		if which_food_2F_60[kk,0] == 1:
			rev_th_temp = -rev_th_temp[np.nonzero(rev_th_temp)]
		if which_food_2F_60[kk,0] == -1:
			rev_th_temp = rev_th_temp[np.nonzero(rev_th_temp)]
		#
		if len(rev_th_temp) > min_RE:
			ind_du2[co_du2] = kk
			co_du2 = co_du2 + 1
			for hh in range(len(rev_th_temp)-1):
				mid_2F_60_du2[kk,hh] = 0.5*(rev_th_temp[hh+1] + rev_th_temp[hh])
				

		####### af
		rev_th_temp = RE_th_af_fly_2F_60[kk,:]
		if which_food_2F_60[kk,0] == 1:
			rev_th_temp = -rev_th_temp[np.nonzero(rev_th_temp)]
		if which_food_2F_60[kk,0] == -1:
			rev_th_temp = rev_th_temp[np.nonzero(rev_th_temp)]
		#
		if len(rev_th_temp) > min_RE:
			ind_af[co_af] = kk
			co_af = co_af + 1
			for hh in range(len(rev_th_temp)-1):
				mid_2F_60_af[kk,hh] = 0.5*(rev_th_temp[hh+1] + rev_th_temp[hh])
			

	# AP (before finding the second food)
	NN = np.shape(mid_2F_60_du1)[0]
	mid_2F_60_du1_f = mid_2F_60_du1.flatten()
	mid_2F_60_du1_f_nz = mid_2F_60_du1_f[np.nonzero(mid_2F_60_du1_f)]

	#
	[bins_circ, kde_circ] = vonmises_kde(mid_2F_60_du1_f_nz, kappa, n_bins)  
	#
	kde_circ_all = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		mid_th_temp = mid_2F_60_du1[kk,:]
		mid_th_temp = mid_th_temp[np.nonzero(mid_th_temp)]
		if len(mid_th_temp) > min_mid:
			# wrapping
			mid_th_temp = (mid_th_temp + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_du1_temp, kde_circ_all[kk,:]] = vonmises_kde(mid_th_temp, kappa, n_bins) 

	kde_circ_du1_m = np.zeros(len(kde_circ))
	kde_circ_du1_u = np.zeros(len(kde_circ))
	kde_circ_du1_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_circ_du1_m)):
		kde_circ_temp = kde_circ_all[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		[kde_circ_du1_m[ii], kde_circ_du1_l[ii], kde_circ_du1_u[ii]] = mean_confidence_interval(kde_circ_temp)
			
	kde_max = np.max(kde_circ_du1_u)
	kde_min = np.min(kde_circ_du1_l)
	#
	fig = plt.figure(401)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure4B, s2_Figure4B)
	#
	ax.plot(bins_circ_du1_temp, kde_circ_du1_m, color = 'red', alpha = 1)
	ax.fill_between(bins_circ_du1_temp, kde_circ_du1_u, kde_circ_du1_l, color = 'red', alpha = 0.5, linewidth = 0)

	#
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 3])
	plt.yticks([0, 1.5, 3])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure4_plots/Figure4_B_run_midpoint_du1_KDE_kappa_'+str(kappa)+'.pdf')	


	# AP (after finding the second food)
	NN = np.shape(mid_2F_60_du2)[0]
	mid_2F_60_du2_f = mid_2F_60_du2.flatten()
	mid_2F_60_du2_f_nz = mid_2F_60_du2_f[np.nonzero(mid_2F_60_du2_f)]

	#
	[bins_circ, kde_circ] = vonmises_kde(mid_2F_60_du2_f_nz, kappa, n_bins)  
	#
	kde_circ_all = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		mid_th_temp = mid_2F_60_du2[kk,:]
		mid_th_temp = mid_th_temp[np.nonzero(mid_th_temp)]
		if len(mid_th_temp) > min_mid:
			# wrapping
			mid_th_temp = (mid_th_temp + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_du2_temp, kde_circ_all[kk,:]] = vonmises_kde(mid_th_temp, kappa, n_bins) 

	kde_circ_du2_m = np.zeros(len(kde_circ))
	kde_circ_du2_u = np.zeros(len(kde_circ))
	kde_circ_du2_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_circ_du2_m)):
		kde_circ_temp = kde_circ_all[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_circ_du2_m[ii], kde_circ_du2_l[ii], kde_circ_du2_u[ii]] = mean_confidence_interval(kde_circ_temp)
			
	kde_max = np.max(kde_circ_du2_u)
	kde_min = np.min(kde_circ_du2_l)
	#
	fig = plt.figure(402)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure4B, s2_Figure4B)
	#
	ax.plot(bins_circ_du2_temp, kde_circ_du2_m, color = 'red', alpha = 1)
	ax.fill_between(bins_circ_du2_temp, kde_circ_du2_u, kde_circ_du2_l, color = 'red', alpha = 0.5, linewidth = 0)

	#
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 3])
	plt.yticks([0, 1.5, 3])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure4_plots/Figure4_B_run_midpoint_du2_KDE_kappa_'+str(kappa)+'.pdf')	

	# post-AP (pre-departure)
	NN = np.shape(mid_2F_60_af)[0]
	mid_2F_60_af_f = mid_2F_60_af.flatten()
	mid_2F_60_af_f_nz = mid_2F_60_af_f[np.nonzero(mid_2F_60_af_f)]

	#
	[bins_circ, kde_circ] = vonmises_kde(mid_2F_60_af_f_nz, kappa, n_bins)  
	#
	kde_circ_all = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		mid_th_temp = mid_2F_60_af[kk,:]
		mid_th_temp = mid_th_temp[np.nonzero(mid_th_temp)]
		if len(mid_th_temp) > min_mid:
			# wrapping
			mid_th_temp = (mid_th_temp + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_circ_all[kk,:]] = vonmises_kde(mid_th_temp, kappa, n_bins) 

	kde_circ_af_m = np.zeros(len(kde_circ))
	kde_circ_af_u = np.zeros(len(kde_circ))
	kde_circ_af_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_circ_af_m)):
		kde_circ_temp = kde_circ_all[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_circ_af_m[ii], kde_circ_af_l[ii], kde_circ_af_u[ii]] = mean_confidence_interval(kde_circ_temp)
			
	kde_max = np.max(kde_circ_af_u)
	kde_min = np.min(kde_circ_af_l)
	#
	fig = plt.figure(403)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure4B, s2_Figure4B)
	#
	ax.plot(bins_circ_af_temp, kde_circ_af_m, color = 'blue', alpha = 1)
	ax.fill_between(bins_circ_af_temp, kde_circ_af_u, kde_circ_af_l, color = 'blue', alpha = 0.5, linewidth = 0)

	#
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 3])
	plt.yticks([0, 1.5, 3])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure4_plots/Figure4_B_run_midpoint_af_KDE_kappa_'+str(kappa)+'.pdf')

##### panel C model sample trace FR
if panel_C == 1:
	s1_Figure4C = 0.9*5.7/1.25
	s2_Figure4C = 0.9*2.3/1.1


	Name_FR = '../Data/simulations/modeling_extracted_data/FR_extracted_data/FR_extracted_data_single/'


	lw = 0.25 
	al = 1
	#
	ind_FR = [4]
	#
	####
	for kk in range(len(ind_FR)):
		food_temp_nz = np.loadtxt(Name_FR+'food_FR_'+str(cond)+'_'+str(kk)+'.txt')
		#
		t_be_temp = np.loadtxt(Name_FR+'t_be_FR_'+str(cond)+'_'+str(ind_FR[kk])+'.txt')
		theta_be_temp = np.loadtxt(Name_FR+'theta_be_FR_'+str(cond)+'_'+str(ind_FR[kk])+'.txt')
		#
		t_du_temp = np.loadtxt(Name_FR+'t_du_FR_'+str(cond)+'_'+str(ind_FR[kk])+'.txt')
		theta_du_temp = np.loadtxt(Name_FR+'theta_du_FR_'+str(cond)+'_'+str(ind_FR[kk])+'.txt')
		#
		t_af_temp = np.loadtxt(Name_FR+'t_af_FR_'+str(cond)+'_'+str(ind_FR[kk])+'.txt')
		theta_af_temp = np.loadtxt(Name_FR+'theta_af_FR_'+str(cond)+'_'+str(ind_FR[kk])+'.txt')
		# wrapping 
		theta_be_temp = (theta_be_temp + np.pi) % (2 * np.pi) - np.pi
		theta_du_temp = (theta_du_temp + np.pi) % (2 * np.pi) - np.pi
		theta_af_temp = (theta_af_temp + np.pi) % (2 * np.pi) - np.pi
		
		t_all = np.hstack([t_be_temp, t_du_temp, t_af_temp])
		theta_all = np.hstack([theta_be_temp, theta_du_temp, theta_af_temp])

		ind_s = find_nearest(t_all/60, 3.5)
		ind_e = find_nearest(t_all/60, 51)
		# #
		t_all = t_all[ind_s:ind_e]
		theta_all = theta_all[ind_s:ind_e]

		fig = plt.figure(101+kk)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_Figure4C, s2_Figure4C)
		#
		ax.plot(t_all/60, (theta_all*sc), color = 'k', linewidth = lw, alpha = al)
		#
		for uu in range(len(food_temp_nz)):
			ax.plot((food_temp_nz[uu]/60)*np.ones(2), np.linspace(25.5,26,2), color = 'r', linewidth = 0.5)
		#
		plt.xlim([3.5,51])
		plt.ylim([-26,26])
		#
		plt.xticks([])
		plt.yticks(13*np.arange(5)-26)
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		#
		fig.savefig('../Plots/Figure4_plots/Figure4_C_sample_FR_sample_'+str(cond)+'_'+str(ind_FR[kk])+'.pdf')

		 
##### 2F_60 FR
if panel_D == 1:
	s1_Figure4D = 1.1*0.9
	s2_Figure4D = 0.8
	#
	# cond = 2
	Name_FR = '../Data/simulations/modeling_extracted_data/FR_extracted_data/FR_extracted_data_population/'

	# loading data
	bins_circ_du1_temp = np.loadtxt(Name_FR+'bins_circ_du1_FR_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du1_m = np.loadtxt(Name_FR+'kde_circ_du1_m_FR_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du1_l = np.loadtxt(Name_FR+'kde_circ_du1_l_FR'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du1_u = np.loadtxt(Name_FR+'kde_circ_du1_u_FR'+str(cond)+'_'+str(kappa)+'.txt')
	#
	bins_circ_du2_temp = np.loadtxt(Name_FR+'bins_circ_du2_FR_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du2_m = np.loadtxt(Name_FR+'kde_circ_du2_m_FR_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du2_l = np.loadtxt(Name_FR+'kde_circ_du2_l_FR'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du2_u = np.loadtxt(Name_FR+'kde_circ_du2_u_FR'+str(cond)+'_'+str(kappa)+'.txt')
	#
	bins_circ_b_aba_temp = np.loadtxt(Name_FR+'bins_circ_b_aba_FR_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_b_aba_m = np.loadtxt(Name_FR+'kde_circ_b_aba_m_FR_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_b_aba_l = np.loadtxt(Name_FR+'kde_circ_b_aba_l_FR'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_b_aba_u = np.loadtxt(Name_FR+'kde_circ_b_aba_u_FR'+str(cond)+'_'+str(kappa)+'.txt')

	# du1 AP before finding the second food
	fig = plt.figure(301)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure4D, s2_Figure4D)
	#
	ax.plot(bins_circ_du1_temp, kde_circ_du1_m, color = 'red', alpha = 1)
	ax.fill_between(bins_circ_du1_temp, kde_circ_du1_u, kde_circ_du1_l, color = 'red', alpha = 0.5, linewidth = 0)
	#
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 3])
	plt.yticks([0, 1.5, 3])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure4_plots/Figure4_D_FR_run_midpoint_du1_KDE_kappa_'+str(kappa)+'.pdf')	

	# du2 AP after finding the second food
	fig = plt.figure(302)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure4D, s2_Figure4D)
	#
	ax.plot(bins_circ_du2_temp, kde_circ_du2_m, color = 'red', alpha = 1)
	ax.fill_between(bins_circ_du2_temp, kde_circ_du2_u, kde_circ_du2_l, color = 'red', alpha = 0.5, linewidth = 0)
	#
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 3])
	plt.yticks([0, 1.5, 3])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure4_plots/Figure4_D_FR_run_midpoint_du2_KDE_kappa_'+str(kappa)+'.pdf')	

	# post-AP pre-departure
	fig = plt.figure(303)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure4D, s2_Figure4D)
	#
	ax.plot(bins_circ_b_aba_temp, kde_circ_b_aba_m, color = 'blue', alpha = 1)
	ax.fill_between(bins_circ_b_aba_temp, kde_circ_b_aba_u, kde_circ_b_aba_l, color = 'blue', alpha = 0.5, linewidth = 0)
	#
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 3])
	plt.yticks([0, 1.5, 3])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure4_plots/Figure4_D_FR_run_midpoint_b_aba_KDE_kappa_'+str(kappa)+'.pdf')		
