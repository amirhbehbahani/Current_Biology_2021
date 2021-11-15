############ Figure 1 
# Repeated back-and-forth excursions constitute a local search around a fictive food location.


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
import seaborn as sns

from custom_fcns import find_nearest
from custom_fcns import adjust_spines
from custom_fcns import mean_confidence_interval
from custom_fcns import vonmises_kde

sc = (180/np.pi)/6.87

panel_B = 1
panel_C = 1
panel_E = 1
panel_F = 1
panel_G = 1
panel_H = 1
panel_I = 1
panel_J = 1
panel_K = 1
panel_L = 1
panel_M = 1

###### panel A: Experimental arena -- no data processing

###### panel B: sample trace from non-trial single food
if panel_B == 1:
	#### plotting parameters
	s1_Figure1B = 5.3*1.2
	s2_Figure1B = 2.3

	lw = 0.25
	####
	case = 1 # 1: single-food
	fly_1F = list([1])
	#
	for kk in range(len(fly_1F)):
		Nam = '../Data/real_fly_data/Non_trial_single_food/'
		# baseline
		t_be_temp = np.loadtxt(Nam+'time_theta/time_theta_be/t_be_'+str(case)+'_'+str(fly_1F[kk])+'.txt')
		theta_be_temp = np.loadtxt(Nam+'time_theta/time_theta_be/theta_be_'+str(case)+'_'+str(fly_1F[kk])+'.txt')
		theta_be_temp = (theta_be_temp + np.pi) % (2 * np.pi) - np.pi
		# activation period
		t_du_temp = np.loadtxt(Nam+'time_theta/time_theta_du/t_du_'+str(case)+'_'+str(fly_1F[kk])+'.txt')
		theta_du_temp = np.loadtxt(Nam+'time_theta/time_theta_du/theta_du_'+str(case)+'_'+str(fly_1F[kk])+'.txt')
		theta_du_temp = (theta_du_temp + np.pi) % (2 * np.pi) - np.pi
		# post-AP
		t_af_temp = np.loadtxt(Nam+'time_theta//time_theta_af/t_af_'+str(case)+'_'+str(fly_1F[kk])+'.txt')
		theta_af_temp = np.loadtxt(Nam+'time_theta/time_theta_af/theta_af_'+str(case)+'_'+str(fly_1F[kk])+'.txt')
		theta_af_temp = (theta_af_temp + np.pi) % (2 * np.pi) - np.pi

		# activation events
		t_adj_temp = np.loadtxt(Nam+'LED_time/LED_time_'+str(case)+'_'+str(fly_1F[kk])+'.txt')
		t_adj_temp = t_adj_temp[0::2] # removing the end of the activation event

		# adjusting the starting and end points
		i_be = find_nearest(t_be_temp, 5)
		t_be_temp_adj = t_be_temp[i_be:]
		theta_be_temp_adj = theta_be_temp[i_be:]
		#
		i_du = find_nearest(t_du_temp, 50)
		t_du_temp_adj = t_du_temp[0:i_du]
		theta_du_temp_adj = theta_du_temp[0:i_du]

		t_adj_temp = t_adj_temp[t_adj_temp>9]
		t_adj_temp = t_adj_temp[t_adj_temp<51]
		#
		fig = plt.figure(201+kk)
		fig.set_size_inches(s1_Figure1B, s2_Figure1B)
		#
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		#
		plt.plot(t_be_temp, theta_be_temp*sc, color = 'black', linewidth = lw, alpha = 1)
		plt.plot(t_du_temp_adj, theta_du_temp_adj*sc, color = 'black', linewidth = lw, alpha = 1)
		plt.plot(t_af_temp, theta_af_temp*sc, color = 'black', linewidth = lw, alpha = 1)
		#
		for ii in range(len(t_adj_temp)):
			plt.plot(t_adj_temp[ii]*np.ones(2), np.linspace(25.5,26,2), color = 'r', markersize = 0, linewidth = 0.5)
		#
		ax.set_xticks([])
		ax.set_yticks(13*np.arange(5)-26)
		#
		adjust_spines(ax, ['left', 'bottom'])	
		#
		fig.savefig('../Plots/Figure1_plots/Figure1_B_Non_trial_single_food_trace_'+str(fly_1F[kk])+'.pdf')

###### panel C: sample trace from trial single food
if panel_C == 1:
	s1_Figure1C = 9.3*0.97
	s2_Figure1C = 2.3

	####
	fly = list([2])
	tri = list([1])
	#
	for kk in range(len(fly)):
		Nam = '../Data/real_fly_data/Trial_single_food/'
		# baseline
		t_be_temp = np.loadtxt(Nam+'time_theta/time_theta_be/t_be_'+str(fly[kk])+'_'+str(tri[kk])+'.txt')
		theta_be_temp = np.loadtxt(Nam+'time_theta/time_theta_be/theta_be_'+str(fly[kk])+'_'+str(tri[kk])+'.txt')
		theta_be_temp = (theta_be_temp + np.pi) % (2 * np.pi) - np.pi

		for ii in range(6):
			# AP
			exec('t_du'+str(ii+1)+'_temp = np.loadtxt(Nam+"time_theta/time_theta_du/t_du_"+str(fly[kk])+"_"+str(tri[kk])+"_"+str(ii+1)+".txt")')
			exec('theta_du'+str(ii+1)+'_temp = np.loadtxt(Nam+"time_theta/time_theta_du/theta_du_"+str(fly[kk])+"_"+str(tri[kk])+"_"+str(ii+1)+".txt")')
			# post-AP
			exec('t_af'+str(ii+1)+'_temp = np.loadtxt(Nam+"time_theta/time_theta_af/t_af_"+str(fly[kk])+"_"+str(tri[kk])+"_"+str(ii+1)+".txt")')
			exec('theta_af'+str(ii+1)+'_temp = np.loadtxt(Nam+"time_theta/time_theta_af/theta_af_"+str(fly[kk])+"_"+str(tri[kk])+"_"+str(ii+1)+".txt")')

		# wrapping
		for ii in range(6):
			exec('theta_du'+str(ii+1)+'_temp = (theta_du'+str(ii+1)+'_temp  + np.pi) % (2 * np.pi) - np.pi')
			exec('theta_af'+str(ii+1)+'_temp = (theta_af'+str(ii+1)+'_temp  + np.pi) % (2 * np.pi) - np.pi')

		# activation events
		t_LED_temp = np.loadtxt(Nam+'LED_time/LED_time_'+str(fly[kk])+'_'+str(tri[kk])+'.txt')
		#######

		fig = plt.figure(301+kk)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		#
		fig.set_size_inches(s1_Figure1C, s2_Figure1C) 
		#
		plt.plot(t_be_temp  - t_be_temp[0], theta_be_temp*sc, color = 'black', linewidth = lw, markersize = 0, alpha = 1)
		#
		for ii in range(6):
			exec('plt.plot(t_du'+str(ii+1)+'_temp  - t_be_temp[0], theta_du'+str(ii+1)+'_temp*sc, color = "black", linewidth = lw, markersize = 0, alpha = 1)')
			exec('plt.plot(t_af'+str(ii+1)+'_temp  - t_be_temp[0], theta_af'+str(ii+1)+'_temp*sc, color = "black", linewidth = lw, markersize = 0, alpha = 1)')
		##
		for jj in range(len(t_LED_temp)):
			plt.plot((t_LED_temp[jj] - t_be_temp[0])*np.ones(2), np.linspace(25.5,26,2), color = 'r', markersize = 0, linewidth = 0.5)
		#
		ax.set_xticks([])
		ax.set_yticks(13*np.arange(5)-26)
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/Figure1_plots/Figure1_C_Trial_single_food_trace_'+str(fly[kk])+'.pdf')

###### panel D: Schematic for defining terms -- no data processing

###### panel E: excursion distance histogram -- non-trial single-food
if panel_E == 1:
	s1_Figure1E = 1.425*1.5
	s2_Figure1E = 1.2

	# loading data
	C_f_f_nz = np.loadtxt('../Data/real_fly_data/Non_trial_single_food/excursion_distance/C_f_f_nz.txt')
	le = len(C_f_f_nz)
	[m_C_f, h_C_f, l_C_f] = mean_confidence_interval(C_f_f_nz*sc)

	fig = plt.figure(501)
	ax = fig.add_subplot(1, 1, 1)
	#
	fig.set_size_inches(s1_Figure1E, s2_Figure1E)
	#
	plt.hist(C_f_f_nz*sc, bins = 'auto', color = 'red', alpha = 1)
	#
	plt.plot(m_C_f*np.ones(2), np.linspace(0,0.3*le,2), linewidth = 0.25, color = 'k')
	#
	#
	plt.xlim([-1,26])
	plt.ylim([0,250])
	#
	plt.xticks(13*np.arange(3))
	plt.yticks(125*np.arange(3))
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure1_plots/Figure1_E_exc_non_trial_1F_hist.pdf')


##### panel F: r_N vs N AP violin plot -- non-trial single-food
if panel_F == 1:
	s1_Figure1F = 2.5
	s2_Figure1F = 1.2

	no_du = 16
	no_af = 1
	# loading data
	r_N_du_all = np.loadtxt('../Data/real_fly_data/Non_trial_single_food/run_length/r_N_du_all.txt')

	m_r_N_du = np.zeros(no_du+no_af-1)
	h_r_N_du = np.zeros(no_du+no_af-1)
	l_r_N_du = np.zeros(no_du+no_af-1)
	#
	for ii in range(no_du+no_af-1):
		r_N_temp = r_N_du_all[:,ii]
		[m_r_N_du[ii], h_r_N_du[ii], l_r_N_du[ii]] = mean_confidence_interval(r_N_temp*sc)

	fig = plt.figure(601)
	ax = fig.add_subplot(1, 1, 1)
	#
	fig.set_size_inches(s1_Figure1F, s2_Figure1F)
	#
	violin_parts = plt.violinplot(r_N_du_all*sc, np.arange(-no_du+1,no_af), showmeans=False,showmedians=False,showextrema=False)
	#
	for pc in violin_parts['bodies']:
		pc.set_facecolor('red')
	#
	ax.plot(np.arange(-no_du+1,no_af), m_r_N_du, '-', color = 'red', alpha = 1, markersize = 0, linewidth = 0.5)
	#
	for ii in range(no_du+no_af-1):
		ax.plot((-no_du+ii+1)*np.ones(3), np.linspace(l_r_N_du[ii], h_r_N_du[ii], 3), color = 'red', alpha = 1, linewidth = 0.25) 	
	#
	plt.xlim([-15.5,0.5])
	plt.ylim([0,40])
	#
	plt.xticks(5*np.arange(4)-15)
	plt.yticks(20*np.arange(3))
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure1_plots/Figure1_F_r_N_vs_N_du_af_CI_violin.pdf')

###### panel G: Run lengths histogram for post-AP trial
if panel_G == 1:	
	s1_Figure1G = 1.425*1.1
	s2_Figure1G = 1.2

	# loading data
	r_N_all = np.loadtxt('../Data/real_fly_data/Trial_single_food/runs_matrix/r_N_all.txt') # run lengths
	r_N_all_copy = np.zeros(np.shape(r_N_all))
	for kk in range(np.shape(r_N_all)[0]):
		r_N_temp = r_N_all[kk,:]
		r_N_temp = r_N_temp[np.nonzero(r_N_temp)]
		for jj in range(len(r_N_temp)-1):
			r_N_all_copy[kk,jj] = r_N_temp[jj]

	r_N_all_f = r_N_all_copy.flatten()
	r_N_all_f_nz = r_N_all_f[np.nonzero(r_N_all_f)]


	le = len(r_N_all_f_nz)
	[m_r_N, h_r_N, l_r_N] = mean_confidence_interval(r_N_all_f_nz*sc)


	fig = plt.figure(701)
	ax = fig.add_subplot(1, 1, 1)
	#
	fig.set_size_inches(s1_Figure1G, s2_Figure1G)
	#
	plt.hist(r_N_all_f_nz*sc, bins = 'auto', color = 'blue', alpha = 1)
	#
	plt.plot(m_r_N*np.ones(2), np.linspace(0,0.3*le,2), linewidth = 0.5, color = 'k')
	#
	plt.xlim([0,40])
	plt.ylim([0,172])
	#
	plt.xticks([0, 20, 40])
	plt.yticks([0, 86, 172])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure1_plots/Figure1_G_run_post_AP_trial_hist.pdf')
	#

##### panel H: r_N vs N AP and post-AP violin plot -- Trial single-food
if panel_H == 1:
	s1_Figure1H = 2
	s2_Figure1H = 1.2

	no_du = 6
	no_af = 11

	r_N_all_fly = np.loadtxt('../Data/real_fly_data/Trial_single_food/run_length/r_N_all_fly.txt')

	for ii in range(16):
		exec('r_N_'+str(ii)+'_fly= np.loadtxt("../Data/real_fly_data/Trial_single_food/run_length/r_N_"+str(ii)+"_fly.txt")')

	m_r_N_all_fly = np.zeros(no_du+no_af-1)
	h_r_N_all_fly = np.zeros(no_du+no_af-1)
	l_r_N_all_fly = np.zeros(no_du+no_af-1)
	#
	for ii in range(no_du+no_af-1):
		r_N_temp = r_N_all_fly[:,ii]
		r_N_temp_nz = r_N_temp[np.nonzero(r_N_temp)]
		r_N_temp_nz = r_N_temp[~np.isnan(r_N_temp)]
		#
		exec('r_N_'+str(ii)+'=r_N_temp_nz*sc')
		#
		[m_r_N_all_fly[ii], h_r_N_all_fly[ii], l_r_N_all_fly[ii]] = mean_confidence_interval(r_N_temp_nz*sc)

	data_to_plot = [r_N_0_fly, r_N_1_fly, r_N_2_fly, r_N_3_fly, r_N_4_fly, r_N_5_fly, r_N_6_fly, r_N_7_fly, r_N_8_fly, r_N_9_fly, r_N_10_fly, r_N_11_fly, r_N_12_fly, r_N_13_fly, r_N_14_fly, r_N_15_fly]
	
	fig = plt.figure(802)
	ax = fig.add_subplot(111, polar=False)
	ax.grid(False)
	#
	fig.set_size_inches(s1_Figure1H, s2_Figure1H) 
	#
	ax = sns.violinplot(data = data_to_plot , scale='area', linewidth=0, cut = 0, inner = None, palette = ['r', 'r','r','r','r', 'r', 'b', 'b','b','b','b','b', 'b','b','b', 'b'])
	for violin, alpha in zip(ax.collections[::1], [0.5,0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5, 0.5, 0.5]):
	    violin.set_alpha(alpha) 
	ax.plot(np.arange(0,no_du), m_r_N_all_fly[0:no_du], '-', color = 'red', alpha = 1, markersize = 0, linewidth = 0.5)
	ax.plot(np.arange(no_du,no_du+no_af-1), m_r_N_all_fly[no_du:], '-', color = 'blue', alpha = 1, markersize = 0, linewidth = 0.5)
	#
	cl = ['r']*no_du + ['b']*no_af

	for ii in range(no_du+no_af-1):
		ax.plot((ii)*np.ones(3), np.linspace(l_r_N_all_fly[ii], h_r_N_all_fly[ii], 3), color = cl[ii], alpha = 1, linewidth = 0.5) 	
	#
	plt.xlim([-2,no_du+no_af-1])
	plt.ylim([0,40])
	#
	ax.set_xticks([0,5,10, 15])
	ax.set_xticklabels(['-5','0', '5', '10'])
	#
	plt.yticks(20*np.arange(3))
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure1_plots/Figure1_H_r_N_vs_N_AP_postAP_fly.pdf')

###### panel I: r_1 vs final excursion highlighted for one of the flies - trial-single food
if panel_I == 1:
	s1_Figure1I = 1.3
	s2_Figure1I = 1.3

	LF_plot = np.linspace(0,26,2)

	# loading data
	# LF and r1 for the sample fly
	LF_fly = np.loadtxt('../Data/real_fly_data/Trial_single_food/r1_vs_LF/LF_fly.txt')
	r1_fly = np.loadtxt('../Data/real_fly_data/Trial_single_food/r1_vs_LF/r1_fly.txt')
	# regression for the sample fly
	fit_fly = np.loadtxt('../Data/real_fly_data/Trial_single_food/r1_vs_LF/fit_fly_LF.txt')
	m_fly = fit_fly[0]
	b_fly = fit_fly[1]
	# regression for all flies
	m_all = np.loadtxt('../Data/real_fly_data/Trial_single_food/r1_vs_LF/m_all_LF.txt')
	b_all = np.loadtxt('../Data/real_fly_data/Trial_single_food/r1_vs_LF/b_all_LF.txt')


	fig = plt.figure(902)
	ax = fig.add_subplot(111, polar=False)
	ax.grid(False)
	fig.set_size_inches(s1_Figure1I, s2_Figure1I)
	ax.plot(LF_fly*sc, r1_fly*sc, 'ok', alpha = 0.8, markersize = 2)
	ax.plot(LF_plot, m_fly*LF_plot + b_fly*np.ones(len(LF_plot)), color = 'k', alpha = 0.8, linewidth = 1)
	#
	plt.xlim([-1,27])
	plt.ylim([-1,27])
	#
	plt.xticks(13*np.arange(3))
	plt.yticks(13*np.arange(3))
	#
	adjust_spines(ax, ['left', 'bottom'])
	#

	for kk in range(len(m_all)):
		fig = plt.figure(902)
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		fig.set_size_inches(s1_Figure1I, s2_Figure1I)
		ax.plot(LF_plot, m_all[kk]*LF_plot + b_all[kk]*np.ones(len(LF_plot)), color = 'k', alpha = 0.1, linewidth = 1)
		#
		plt.xlim([-1,27])
		plt.ylim([-1,27])
		#
		plt.xticks(13*np.arange(3))
		plt.yticks(13*np.arange(3))
		#
		adjust_spines(ax, ['left', 'bottom'])
		#
		fig.savefig('../Plots/Figure1_plots/Figure1_I_r1_vs_LF_all.pdf')
	

######## panel J: ticket-to-ride (heatmap for Trial - single food)
if panel_J == 1:
	# heatmap parameters
	s1_Figure1J = 6
	s2_Figure1J = 3.5
	#####

	pad = 0
	he = 1
	max_rN = 52
	al_e = 0.05
	lw = 1
	lw2 = 0.1*0
	lw3 = 0.1 # vertical linewidth
	lw4 = 1 # end of the first search
	lw_e = 0.1

	##### loading data
	r_N_all = np.loadtxt('../Data/real_fly_data/Trial_single_food/runs_matrix/r_N_all.txt') # run lengths
	time_start_all = np.loadtxt('../Data/real_fly_data/Trial_single_food/runs_matrix/time_start_all.txt') # start time for runs
	time_width_all = np.loadtxt('../Data/real_fly_data/Trial_single_food/runs_matrix/time_width_all.txt') # time duration for runs
	post_search_t = np.loadtxt('../Data/real_fly_data/Trial_single_food/runs_matrix/post_search_t.txt') # duration of post-AP search for trials
	co_r_all = np.loadtxt('../Data/real_fly_data/Trial_single_food/runs_matrix/co_r_all.txt') # the counter number for trials
	#
	si = np.shape(r_N_all)
	si0 = int(si[0])
	si1 = int(si[1])

	post_search_t = np.reshape(post_search_t, (si0,1))
	co_r_all = np.reshape(co_r_all, (si0,1))

	r_N_all_time = np.hstack([r_N_all, -post_search_t])
	time_start_all_time = np.hstack([time_start_all, -post_search_t]) 
	time_width_all_time = np.hstack([time_width_all, -post_search_t])
	co_r_all_time = np.hstack([co_r_all, -post_search_t]) 

	### sorting based on the first post AP duration
	r_N_all_time = r_N_all_time[r_N_all_time[:,-1].argsort()]
	time_start_all_time = time_start_all_time[time_start_all_time[:,-1].argsort()]
	time_width_all_time = time_width_all_time[time_width_all_time[:,-1].argsort()]
	co_r_all_time = co_r_all_time[co_r_all_time[:,-1].argsort()]

	### removing the time column
	co_r_all_time = co_r_all_time[:,0:-1]

	no_tri = len(co_r_all[np.nonzero(co_r_all)])
	########## crop at the end of the first post AP search
	for kk in range(no_tri):
		####### extracting the relevant vectors from the matrix
		dur = -r_N_all_time[kk,-1]
		co_r = int(co_r_all_time[kk])
		r_N_temp = 	r_N_all_time[kk,0:co_r]	
		time_start = time_start_all_time[kk,0:co_r]
		time_width = time_width_all_time[kk,0:co_r]
		####### heatmap
		fig = plt.figure(1001)
		#					
		fig.set_size_inches(s1_Figure1J, s2_Figure1J)
		#
		ax = fig.add_subplot(111, polar=False)
		ax.grid(False)
		# adding the initial box on the left
		x_start_temp = 0
		y_start_temp = pad + kk*he
		#
		ax.add_patch(Rectangle((x_start_temp, y_start_temp), time_start[0], he, fc = 'black', alpha = al_e, ec ='none', lw = lw_e))
		# add a vertical line
		ax.plot(0*np.ones(2), np.linspace(y_start_temp,y_start_temp+he,2), color = 'black', linewidth = lw3)	
		#
		for tt in range(len(time_start)):
			if time_start[tt] < dur+1:
				if r_N_temp[tt]*sc < max_rN:
					al = ((r_N_temp[tt]*sc)/max_rN)
				else:
					al = 1	
				#
				x_start_temp = time_start[tt]
				y_start_temp = pad + kk*he
				#
				ax.add_patch(Rectangle((x_start_temp, y_start_temp), time_width[tt], he, fc ='blue', alpha = al, ec ='none', lw = lw2))
				# add a vertical line
				ax.plot(x_start_temp*np.ones(2), np.linspace(y_start_temp,y_start_temp+he,2), color = 'black', linewidth = lw3)
						
	plt.xlim([0, 6])
	plt.ylim([-1, no_tri*he+5])
	#		
	ax.set_yticks([])
	adjust_spines(ax, ['left', 'bottom'])	
	#
	fig.savefig('../Plots/Figure1_plots/Figure1_J_heatmap_rN_all_search_sorted.pdf')
		

	no = 200
	fig = plt.figure(1002)
	ax = fig.add_subplot(111, polar=False)
	ax.grid(False)
	#					
	fig.set_size_inches(0.5, 4)
	#
	for ii in range(no):
		ax.add_patch(Rectangle((0, ii*0.05), 1, 0.05, fc ='blue', alpha = ii/200, ec ='none', lw = 0))
	#
	plt.xlim([0, 1])
	plt.ylim([0, 10])
	#
	ax.set_xticks([])
	ax.set_yticks([])
	adjust_spines(ax, ['left', 'bottom'])	
	#
	fig.savefig('../Plots/Figure1_plots/Figure1_J_color_map.pdf')
				
####### panel K
if panel_K == 1:
	s1_Figure1K = 2.5
	s2_Figure1K = 1.2
	#
	NN = 11
	lw = 1

	# loading data
	r_fly_dep_0_fil_nz = np.loadtxt('../Data/real_fly_data/Trial_single_food/runs_pre_departure/r_fly_dep_0_fil_nz.txt')
	[m_r_fly_dep_0_fil, h_r_fly_dep_0_fil, l_r_fly_dep_0_fil] = mean_confidence_interval(r_fly_dep_0_fil_nz)
	#
	for ii in range(NN-1):
		exec('r_fly_dep_'+str(ii+1)+'_fil_nz = np.loadtxt("../Data/real_fly_data/Trial_single_food/runs_pre_departure/r_fly_dep_"+str(ii+1)+"_fil_nz.txt")')
		exec('[m_r_fly_dep_'+str(ii+1)+'_fil'+',h_r_fly_dep_'+str(ii+1)+'_fil'+',l_r_fly_dep_'+str(ii+1)+'_fil] = mean_confidence_interval(r_fly_dep_'+str(ii+1)+'_fil_nz)')		


	fig = plt.figure(1102)
	ax = fig.add_subplot(111, polar=False)
	ax.grid(False)
	fig.set_size_inches(s1_Figure1K, s2_Figure1K)
	#
	plt.subplot(1,1,1)
	#
	data_to_plot = [r_fly_dep_10_fil_nz, r_fly_dep_9_fil_nz, r_fly_dep_8_fil_nz, r_fly_dep_7_fil_nz, r_fly_dep_6_fil_nz, r_fly_dep_5_fil_nz, r_fly_dep_4_fil_nz, r_fly_dep_3_fil_nz, r_fly_dep_2_fil_nz, r_fly_dep_1_fil_nz, r_fly_dep_0_fil_nz]	
	#
	ax = sns.violinplot(data = data_to_plot , scale='area', linewidth=0, cut = 0, inner = None, palette = ['b', 'b', 'b','b','b','b', 'b', 'b','b','b','b'])
	for violin, alpha in zip(ax.collections[::1], [0.5, 0.5,0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5,0.5]):
	    violin.set_alpha(alpha)  
	#
	m_avg = [m_r_fly_dep_10_fil, m_r_fly_dep_9_fil, m_r_fly_dep_8_fil, m_r_fly_dep_7_fil, m_r_fly_dep_6_fil, m_r_fly_dep_5_fil, m_r_fly_dep_4_fil, m_r_fly_dep_3_fil, m_r_fly_dep_2_fil, m_r_fly_dep_1_fil, m_r_fly_dep_0_fil]
	ax.plot(np.arange(NN), m_avg, linewidth = 0.5*lw, ms = 0, color = 'b')
	#
	ax.plot(10*np.ones(2), np.linspace(h_r_fly_dep_0_fil, l_r_fly_dep_0_fil, 2), color = 'b', linewidth = 0.5*lw)
	ax.plot(9*np.ones(2), np.linspace(h_r_fly_dep_1_fil, l_r_fly_dep_1_fil, 2), color = 'b', linewidth = 0.5*lw)
	ax.plot(8*np.ones(2), np.linspace(h_r_fly_dep_2_fil, l_r_fly_dep_2_fil, 2), color = 'b', linewidth = 0.5*lw)
	ax.plot(7*np.ones(2), np.linspace(h_r_fly_dep_3_fil, l_r_fly_dep_3_fil, 2), color = 'b', linewidth = 0.5*lw)
	ax.plot(6*np.ones(2), np.linspace(h_r_fly_dep_4_fil, l_r_fly_dep_4_fil, 2), color = 'b', linewidth = 0.5*lw)
	ax.plot(5*np.ones(2), np.linspace(h_r_fly_dep_5_fil, l_r_fly_dep_5_fil, 2), color = 'b', linewidth = 0.5*lw)
	ax.plot(4*np.ones(2), np.linspace(h_r_fly_dep_6_fil, l_r_fly_dep_6_fil, 2), color = 'b', linewidth = 0.5*lw)
	ax.plot(3*np.ones(2), np.linspace(h_r_fly_dep_7_fil, l_r_fly_dep_7_fil, 2), color = 'b', linewidth = 0.5*lw)
	ax.plot(2*np.ones(2), np.linspace(h_r_fly_dep_8_fil, l_r_fly_dep_8_fil, 2), color = 'b', linewidth = 0.5*lw)
	ax.plot(1*np.ones(2), np.linspace(h_r_fly_dep_9_fil, l_r_fly_dep_9_fil, 2), color = 'b', linewidth = 0.5*lw)
	ax.plot(0*np.ones(2), np.linspace(h_r_fly_dep_10_fil, l_r_fly_dep_10_fil, 2), color = 'b', linewidth = 0.5*lw)
	#
	ax.set_xticks([0, 5, 10])
	#
	ax.set_yticks([0,60,120])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure1_plots/Figure1_K_violin_pre_departure_fly.pdf')


####### panel L
if panel_L == 1:
	s1_Figure1L = 0.54*1.2
	s2_Figure1L = 1.2
	#

	# loading data
	first_run_1F_nz = np.loadtxt('../Data/real_fly_data/Trial_single_food/max_vs_departure/first_run_1F_nz.txt')
	max_inter_1F_nz = np.loadtxt('../Data/real_fly_data/Trial_single_food/max_vs_departure/max_inter_1F_nz.txt')


	fig = plt.figure(1201)
	ax = fig.add_subplot(111, polar=False)
	ax.grid(False)
	#
	fig.set_size_inches(s1_Figure1L, s2_Figure1L)
	#
	for kk in range(len(first_run_1F_nz)):
		temp = [max_inter_1F_nz[kk], first_run_1F_nz[kk]]
		#
		plt.plot([0,1], temp, '-o', color = 'blue', alpha = 1, linewidth = 0.15, markersize = 0, clip_on = False)
	#
	plt.ylim([0, 200])
	plt.xlim([-0.1, 1.1])
	#
	ax.set_xticks([])
	ax.set_yticks([0, 50, 100, 150, 200])
	#		
	adjust_spines(ax, ['left', 'bottom'])	
	#
	fig.savefig('../Plots/Figure1_plots/Figure1_L_pair_plot_first_run.pdf')


############# panel M KDE
if panel_M == 1:
	s1_Figure1M = 1
	s2_Figure1M = 1.2
	#
	min_mid = 0
	kappa = 200
	n_bins = 1000
	# baseline

	mid_th_all_adj = np.loadtxt('../Data/real_fly_data/KDE_run_midpoint/mid_th_all_adj.txt')
	# 
	NN = np.shape(mid_th_all_adj)[0]
	#
	mid_th_all_adj_f = mid_th_all_adj.flatten()
	mid_th_all_adj_f_nz = mid_th_all_adj_f[np.nonzero(mid_th_all_adj_f)]
	#
	[bins_circ, kde_circ] = vonmises_kde(mid_th_all_adj_f_nz, kappa, n_bins)  
	#
	kde_be = np.zeros((NN, len(kde_circ)))
	#
	for kk in range(NN):
		run_th_temp = mid_th_all_adj[kk,:]
		#
		run_th_temp = run_th_temp[np.nonzero(run_th_temp)]	
		if len(run_th_temp) > min_mid:
			# wrapping
			run_th_temp = (run_th_temp + np.pi) % (2 * np.pi) - np.pi
			#
			[bins_circ_af_temp, kde_be[kk,:]] = vonmises_kde(run_th_temp, kappa, n_bins) 

	kde_be_m = np.zeros(len(kde_circ))
	kde_be_u = np.zeros(len(kde_circ))
	kde_be_l = np.zeros(len(kde_circ))

	for ii in range(len(kde_be_m)):
		kde_circ_temp = kde_be[:,ii]
		kde_circ_temp = kde_circ_temp[~np.isnan(kde_circ_temp)]
		kde_circ_temp = kde_circ_temp[np.nonzero(kde_circ_temp)]
		[kde_be_m[ii], kde_be_l[ii], kde_be_u[ii]] = mean_confidence_interval(kde_circ_temp)
				
	kde_max = np.max(kde_be_u)
	kde_min = np.min(kde_be_l)
	#
	fig = plt.figure(1301)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure1M, s2_Figure1M)
	#
	ax.plot(bins_circ_af_temp, kde_be_m, color = 'black', alpha = 0.5)
	ax.fill_between(bins_circ_af_temp, kde_be_u, kde_be_l, color = 'black', alpha = 0.5, linewidth = 0)
	# 
	#	
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 2.5])
	plt.yticks([0, 1.25, 2.5])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure1_plots/Figure1_M_KDE_run_mid_be.pdf')


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
	fig = plt.figure(1302)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure1M, s2_Figure1M)
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
	fig.savefig('../Plots/Figure1_plots/Figure1_M_KDE_run_mid_du.pdf')


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
	fig = plt.figure(1303)
	ax = fig.add_subplot(111, polar=False)
	fig.set_size_inches(s1_Figure1M, s2_Figure1M)
	#
	ax.plot(bins_circ_af_temp, kde_af_m, color = 'blue', alpha = 0.5)
	ax.fill_between(bins_circ_af_temp, kde_af_u, kde_af_l, color = 'blue', alpha = 0.5, linewidth = 0)
	# 
	#	
	plt.xlim([-np.pi, np.pi])
	plt.xticks([-np.pi, 0, np.pi])
	#
	plt.ylim([0, 2.5])
	plt.yticks([0, 1.25, 2.5])
	#
	adjust_spines(ax, ['left', 'bottom'])
	#
	fig.savefig('../Plots/Figure1_plots/Figure1_M_KDE_run_mid_af.pdf')

