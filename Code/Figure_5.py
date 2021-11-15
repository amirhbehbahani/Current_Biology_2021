############ Figure 5
# Two modified versions of the FR model recapitulate Drosophila search behavior around multiple fictive food sites.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from custom_fcns import adjust_spines
from custom_fcns import safe_div
from custom_fcns import find_nearest
from custom_fcns import mean_confidence_interval
from custom_fcns import vonmises_kde


sc = (180/np.pi)/6.87

panel_D = 1
panel_E = 1

panel_F = 1
panel_G = 1

kappa = 200
n_bins = 2000
min_mid = 3


up = 0.2
DEF_size = 1.1

###### panels A, B, and C: Cartoon for showing the differences among FR, FR', and CR -- no data processing

##### panel D 2D FR_p
if panel_D == 1:
	s1_Figure5D = 5.7/1.25
	s2_Figure5D = 2.3/1.1

	cond = 2 # 2F_60
	sc = (180/np.pi)/6.87

	Name_FR = '../Data/simulations/modeling_extracted_data/FR_p_extracted_data/FR_p_extracted_data_single/'


	lw = 0.25 
	al = 1
	#
	ind_FR_p = [9]
	#
	####
	for kk in range(len(ind_FR_p)):
		food_temp_nz = np.loadtxt(Name_FR+'/food_FR_p_'+str(cond)+'_'+str(kk)+'.txt')
		#
		t_be_temp = np.loadtxt(Name_FR+'/t_be_FR_p_'+str(cond)+'_'+str(ind_FR_p[kk])+'.txt')
		theta_be_temp = np.loadtxt(Name_FR+'/theta_be_FR_p_'+str(cond)+'_'+str(ind_FR_p[kk])+'.txt')
		#
		t_du_temp = np.loadtxt(Name_FR+'/t_du_FR_p_'+str(cond)+'_'+str(ind_FR_p[kk])+'.txt')
		theta_du_temp = np.loadtxt(Name_FR+'/theta_du_FR_p_'+str(cond)+'_'+str(ind_FR_p[kk])+'.txt')
		#
		t_af_temp = np.loadtxt(Name_FR+'/t_af_FR_p_'+str(cond)+'_'+str(ind_FR_p[kk])+'.txt')
		theta_af_temp = np.loadtxt(Name_FR+'/theta_af_FR_p_'+str(cond)+'_'+str(ind_FR_p[kk])+'.txt')
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

		fig = plt.figure(401+kk)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_Figure5D, s2_Figure5D)
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
		fig.savefig('../Plots/Figure5_plots/Figure5_D_FR_p_sample_'+str(cond)+'_'+str(ind_FR_p[kk])+'.pdf')


##### panel E 2D CR
if panel_E == 1:
	s1_Figure5E = 5.7/1.25
	s2_Figure5E = 2.3/1.1

	cond = 2 # 2F_60
	sc = (180/np.pi)/6.87

	Name_CR = '../Data/simulations/modeling_extracted_data/CR_extracted_data/CR_extracted_data_single/'


	lw = 0.25 
	al = 1
	#
	ind_CR = [35]
	#
	####
	for kk in range(len(ind_CR)):
		food_temp_nz = np.loadtxt(Name_CR+'/food_CR_'+str(cond)+'_'+str(kk)+'.txt')
		#
		t_be_temp = np.loadtxt(Name_CR+'/t_be_CR_'+str(cond)+'_'+str(ind_CR[kk])+'.txt')
		theta_be_temp = np.loadtxt(Name_CR+'/theta_be_CR_'+str(cond)+'_'+str(ind_CR[kk])+'.txt')
		#
		t_du_temp = np.loadtxt(Name_CR+'/t_du_CR_'+str(cond)+'_'+str(ind_CR[kk])+'.txt')
		theta_du_temp = np.loadtxt(Name_CR+'/theta_du_CR_'+str(cond)+'_'+str(ind_CR[kk])+'.txt')
		#
		t_af_temp = np.loadtxt(Name_CR+'/t_af_CR_'+str(cond)+'_'+str(ind_CR[kk])+'.txt')
		theta_af_temp = np.loadtxt(Name_CR+'/theta_af_CR_'+str(cond)+'_'+str(ind_CR[kk])+'.txt')
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

		fig = plt.figure(501+kk)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_Figure5E, s2_Figure5E)
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
		fig.savefig('../Plots/Figure5_plots/Figure5_E_CR_sample_'+str(cond)+'_'+str(ind_CR[kk])+'.pdf')


##### 2F_60 FR_p
if panel_F == 1:
	s1_Figure5F = 0.9*1.25
	s2_Figure5F = 0.8*1.25
	#
	cond = 2
	Name_FR_p = '../Data/simulations/modeling_extracted_data/FR_p_extracted_data/FR_p_extracted_data_population/'

	# loading data
	bins_circ_du1_temp = np.loadtxt(Name_FR_p+'bins_circ_du1_FR_p_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du1_m = np.loadtxt(Name_FR_p+'kde_circ_du1_m_FR_p_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du1_l = np.loadtxt(Name_FR_p+'kde_circ_du1_l_FR_p'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du1_u = np.loadtxt(Name_FR_p+'kde_circ_du1_u_FR_p'+str(cond)+'_'+str(kappa)+'.txt')
	#
	bins_circ_du2_temp = np.loadtxt(Name_FR_p+'bins_circ_du2_FR_p_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du2_m = np.loadtxt(Name_FR_p+'kde_circ_du2_m_FR_p_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du2_l = np.loadtxt(Name_FR_p+'kde_circ_du2_l_FR_p'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du2_u = np.loadtxt(Name_FR_p+'kde_circ_du2_u_FR_p'+str(cond)+'_'+str(kappa)+'.txt')
	#
	bins_circ_b_aba_temp = np.loadtxt(Name_FR_p+'bins_circ_b_aba_FR_p_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_b_aba_m = np.loadtxt(Name_FR_p+'kde_circ_b_aba_m_FR_p_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_b_aba_l = np.loadtxt(Name_FR_p+'kde_circ_b_aba_l_FR_p'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_b_aba_u = np.loadtxt(Name_FR_p+'kde_circ_b_aba_u_FR_p'+str(cond)+'_'+str(kappa)+'.txt')

	# du1 AP before finding the second food
	fig = plt.figure(601)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure5F, s2_Figure5F)
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
	fig.savefig('../Plots/Figure5_plots/Figure5_F_FR_p_run_midpoint_du1_KDE_kappa_'+str(kappa)+'.pdf')	

	# du2 AP after finding the second food
	fig = plt.figure(602)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure5F, s2_Figure5F)
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
	fig.savefig('../Plots/Figure5_plots/Figure5_F_FR_p_run_midpoint_du2_KDE_kappa_'+str(kappa)+'.pdf')	

	# post-AP pre-departure
	fig = plt.figure(603)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure5F, s2_Figure5F)
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
	fig.savefig('../Plots/Figure5_plots/Figure5_F_FR_p_run_midpoint_b_aba_KDE_kappa_'+str(kappa)+'.pdf')	

##### 2F_60 CR
if panel_G == 1:
	s1_Figure5G = 0.9*1.25
	s2_Figure5G = 0.8*1.25
	#
	cond = 2
	Name_CR = '../Data/simulations/modeling_extracted_data/CR_extracted_data/CR_extracted_data_population/'

	# loading data
	bins_circ_du1_temp = np.loadtxt(Name_CR+'bins_circ_du1_CR_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du1_m = np.loadtxt(Name_CR+'kde_circ_du1_m_CR_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du1_l = np.loadtxt(Name_CR+'kde_circ_du1_l_CR'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du1_u = np.loadtxt(Name_CR+'kde_circ_du1_u_CR'+str(cond)+'_'+str(kappa)+'.txt')
	#
	bins_circ_du2_temp = np.loadtxt(Name_CR+'bins_circ_du2_CR_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du2_m = np.loadtxt(Name_CR+'kde_circ_du2_m_CR_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du2_l = np.loadtxt(Name_CR+'kde_circ_du2_l_CR'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_du2_u = np.loadtxt(Name_CR+'kde_circ_du2_u_CR'+str(cond)+'_'+str(kappa)+'.txt')
	#
	bins_circ_b_aba_temp = np.loadtxt(Name_CR+'bins_circ_b_aba_CR_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_b_aba_m = np.loadtxt(Name_CR+'kde_circ_b_aba_m_CR_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_b_aba_l = np.loadtxt(Name_CR+'kde_circ_b_aba_l_CR'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_b_aba_u = np.loadtxt(Name_CR+'kde_circ_b_aba_u_CR'+str(cond)+'_'+str(kappa)+'.txt')

	# du1 AP before finding the second food
	fig = plt.figure(701)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure5G, s2_Figure5G)
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
	fig.savefig('../Plots/Figure5_plots/Figure5_G_CR_run_midpoint_du1_KDE_kappa_'+str(kappa)+'.pdf')	

	# du2 AP after finding the second food
	fig = plt.figure(702)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure5G, s2_Figure5G)
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
	fig.savefig('../Plots/Figure5_plots/Figure5_G_CR_run_midpoint_du2_KDE_kappa_'+str(kappa)+'.pdf')	

	# post-AP pre-departure
	fig = plt.figure(703)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure5G, s2_Figure5G)
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
	fig.savefig('../Plots/Figure5_plots/Figure5_G_CR_run_midpoint_b_aba_KDE_kappa_'+str(kappa)+'.pdf')

