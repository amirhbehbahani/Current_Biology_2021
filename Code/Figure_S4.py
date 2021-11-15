############ Figure S4
# The CR model recapitulates fly re-initiation of local search at a former fictive food site after circling the arena.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from custom_fcns import adjust_spines
from custom_fcns import safe_div
from custom_fcns import find_nearest
from custom_fcns import mean_confidence_interval


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

sc = (180/np.pi)/(2*6.87)

kappa = 200
n_bins = 2000
cond = 1 # small arena - circular 
#
lw = 1
fil = 180 # filtering threshold for run length in degrees
no = 15

###### panel A CR trial sample trace
if panel_A == 1:
	ind_CR = [50, 52, 72, 73, 5, 14]

	#
	s1_FigureS4A = 4.5
	s2_FigureS4A = 3

	co1 = 0
	Name_CR = '../Data/simulations/modeling_extracted_data/CR_extracted_data/CR_extracted_data_single/'
	
	for kk in range(len(ind_CR)):
		#
		t_af_temp_all = np.loadtxt(Name_CR+'t_af_CR_'+str(cond)+'_'+str(ind_CR[kk])+'.txt')
		theta_af_temp_all = np.loadtxt(Name_CR+'theta_af_CR_'+str(cond)+'_'+str(ind_CR[kk])+'.txt')
		#
		t_b_aba_ind = np.loadtxt(Name_CR+'t_b_aba_CR_'+str(cond)+'_'+str(ind_CR[kk])+'.txt')
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
		#
		fig = plt.figure(101)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_FigureS4A, s2_FigureS4A)
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
		fig.savefig('../Plots/FigureS4_plots/FigureS4_A_CR_sample_'+str(cond)+'.pdf')


###### panel B transits CR
if panel_B == 1:
	cond = 1
	no = 15
	#
	s1_FigureS4B = 1.5
	s2_FigureS4B = 3*0.9936
	#
	Name_CR = '../Data/simulations/modeling_extracted_data/CR_extracted_data/CR_extracted_data_population'

	theta = np.loadtxt(Name_CR+'/theta_CR_'+str(cond)+'.txt')
	m_co_aba_norm = np.loadtxt(Name_CR+'/m_co_aba_norm_CR_'+str(cond)+'.txt')
	l_co_aba_norm = np.loadtxt(Name_CR+'/l_co_aba_norm_CR_'+str(cond)+'.txt')
	h_co_aba_norm = np.loadtxt(Name_CR+'/h_co_aba_norm_CR_'+str(cond)+'.txt')

	fig = plt.figure(201)
	fig.set_size_inches(s2_FigureS4B, s1_FigureS4B)
	ax = fig.add_subplot(111, polar=False)
	#
	#
	ax.plot(theta, m_co_aba_norm, color = 'k', alpha = 1, linewidth = 0.5)
	ax.fill_between(theta, h_co_aba_norm, l_co_aba_norm, color = 'black', alpha = 0.3, linewidth = 0)
	#
	plt.xticks(360*np.arange(no-1)-2520, ('', '', '', '', '', '', '', '', '', '', '', '', '', '', ''))
	plt.yticks(0.02*np.arange(3))
	#
	#
	plt.xlim([-7*360, 7*360])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/FigureS4_plots/FigureS4_B_CR_transits_'+str(cond)+'.pdf')

###### panel C KDE for run midpoint
if panel_C == 1:
	s1_FigureS4C = 1.75
	s2_FigureS4C = 2.5/2
	#	
	Name_CR = '../Data/simulations/modeling_extracted_data/CR_extracted_data/CR_extracted_data_population/'
	#	
	bins_circ_temp = np.loadtxt(Name_CR+'/bins_circ_CR_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_m = np.loadtxt(Name_CR+'/kde_circ_m_CR_'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_l = np.loadtxt(Name_CR+'/kde_circ_l_CR'+str(cond)+'_'+str(kappa)+'.txt')
	kde_circ_u = np.loadtxt(Name_CR+'/kde_circ_u_CR'+str(cond)+'_'+str(kappa)+'.txt')


	fig = plt.figure(401)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_FigureS4C, s2_FigureS4C)
	#
	ax.plot(bins_circ_temp, kde_circ_m, color = 'black', alpha = 1)
	ax.fill_between(bins_circ_temp, kde_circ_u, kde_circ_l, color = 'black', alpha = 0.3, linewidth = 0)
	#
	plt.plot(0*np.ones(2), np.linspace(0, 1, 2), '--k', alpha = 0.5)
	# #
	#
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.yticks([0, 0.5, 1])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/FigureS4_plots/FigureS4_C_CR_run_midpoint_KDE_kappa_'+str(kappa)+'_'+str(cond)+'.pdf')
	
######### panel D number of midruns in quadrants
if panel_D == 1:
	s1_FigureS4D = 1.6
	s2_FigureS4D = 2.5/2
	#
	cond = 1
	Name_CR = '../Data/simulations/modeling_extracted_data/CR_extracted_data/CR_extracted_data_population/'
	#
	mid_quad_no = np.loadtxt(Name_CR+'/mid_quad_no_CR_'+str(cond)+'.txt')
	NN = np.shape(mid_quad_no)[0]
	#
	fig = plt.figure(301)
	ax = fig.add_subplot(111, polar=False)
	ax.grid(False)
	fig.set_size_inches(s1_FigureS4D, s2_FigureS4D)
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
		adjust_spines(ax, ['left', 'bottom'])
		#
	#
	fig.savefig('../Plots/FigureS4_plots/FigureS4_D_CR_run_midpoint_dual_plot_'+str(cond)+'.pdf')



