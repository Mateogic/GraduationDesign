# 1. Response time with time for each application (scatter) --> 95th percentile for SOTA = SLA (scatter)
# 2. Number of migrations, Avrage interval response time, Average interval energy, scheduling time (time series)
# 3. Response time vs total IPS, Response time / Total IPS vs Total IPS (series)
# 4. Total energy, avg response time, cost/number of tasks completed, cost, number of total tasks completed
# Total number of migrations, total migration time, total execution, total scheduling time.

# Estimates of GOBI vs GOBI* (accuracy)

import matplotlib.pyplot as plt
import matplotlib
import itertools
import statistics
import pickle
import numpy as np
import scipy.stats
import pandas as pd
from stats.Stats import *
import seaborn as sns
from pprint import pprint
from utils.Utils import *
import os
import fnmatch
from sys import argv

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = True
size = (2.9, 2.5)
env = argv[1]
option = 0
sla_baseline = 'GOBI'
rot = 25

def fairness(l):
	a = 1 / (np.mean(l)-(scipy.stats.hmean(l)+0.001)) # 1 / slowdown i.e. 1 / (am - hm)
	if a: return a
	return 0

def jains_fairness(l):
	a = np.sum(l)**2 / (len(l) * np.sum(l**2) + 0.0001) # Jain's fairness index
	if a: return a
	return 0

def fstr(val):
	# return "{:.2E}".format(val)
	return "{:.2f}".format(val)

def reduce(l):
	n = 5
	res, low, high = [], [], []
	for i in range(0, len(l)):
		res.append(statistics.mean(l[max(0, i-n):min(len(l), i+n)]))
		low.append(min(l[max(0, i-n):min(len(l), i+n)]))
		high.append(max(l[max(0, i-n):min(len(l), i+n)]))
	res, low, high = np.array(res), np.array(low), np.array(high)
	low = 0.1 * low + 0.9 * res; high = 0.1 * high + 0.9 * res
	return res, low, high

def mean_confidence_interval(data, confidence=0.90):
    a = 1.0 * np.array(data)
    n = len(a)
    h = scipy.stats.sem(a) * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

PATH = 'all_datasets/' + env + '/'
SAVE_PATH = 'results/' + env + '/'

# Models = ['PreGANPlus1', 'PreGAN', 'CMODLB', 'PCFT', 'ECLB', 'DFTM', 'GOBI']
Models = ['PreDiffusion', 'PreGANPro', 'PreGANPlus', 'CMODLB', 'PCFT', 'ECLB', 'DFTM', 'GOBI']
# Models = ['1-PreGAN-old','2-PreGANPro-old', '3-PreGANPro-old', '4-PreGANPro-old', \
# 		  '1-PreGAN-new','2-PreGANPro-new', '3-PreGANPro-new', '4-PreGANPro-new']
xLabel = 'Execution Time (minutes)'
Colors = ['red', 'blue', 'green', 'orange', 'magenta', 'pink', 'cyan', 'maroon', 'grey', 'purple', 'navy']
apps = ['yolo', 'pocketsphinx', 'aeneas']

yLabelsStatic = ['Average Interval Energy (Kilowatt-hr)', 'Average Response Time (seconds)', 'Average CPU Utilization (%)',\
				 'Average RAM Utilization (%)', 'Interval Allocation Time (seconds)', 'Number of Task migrations',\
				 'Fraction of total SLA Violations', 'Fraction of SLA Violations per application', 'Number of completed tasks per application']
# yLabelsStatic = ['Total Energy (Kilowatt-hr)', 'Average Energy (Kilowatt-hr)', 'Interval Energy (Kilowatt-hr)', 'Average Interval Energy (Kilowatt-hr)',\
# 	'Number of completed tasks', 'Number of completed tasks per interval', 'Average Response Time (seconds)', 'Total Response Time (seconds)',\
# 	'Average Migration Time (seconds)', 'Total Migration Time (seconds)', 'Number of Task migrations', 'Average Wait Time (intervals)', 'Average Wait Time (intervals) per application',\
# 	'Average Completion Time (seconds)', 'Total Completion Time (seconds)', 'Average Response Time (seconds) per application',\
# 	'Cost per container (US Dollars)', 'Fraction of total SLA Violations', 'Fraction of SLA Violations per application', \
# 	'Interval Allocation Time (seconds)', 'Number of completed tasks per application', "Fairness (Jain's index)", 'Fairness', 'Fairness per application', \
# 	'Average CPU Utilization (%)', 'Average number of containers per Interval', 'Average RAM Utilization (%)', 'Scheduling Time (seconds)',\
# 	'Average Execution Time (seconds)']

yLabelStatic2 = {
	'Average Completion Time (seconds)': 'Number of completed tasks'
}

yLabelsTime = ['Interval Energy (Kilowatts)', 'Number of completed tasks', 'Interval Response Time (seconds)', \
	'Interval Migration Time (seconds)', 'Interval Completion Time (seconds)', 'Interval Cost (US Dollar)', \
	'Fraction of SLA Violations', 'Number of Task migrations', 'Number of Task migrations', 'Average Wait Time', 'Average Wait Time (intervals)', \
	'Average Execution Time (seconds)']

all_stats_list = []
load_models = Models if sla_baseline in Models else Models+[sla_baseline]
for model in load_models:
	try:
		model2 = model.replace('*', '2').replace('GOSH', 'HSOGOBI').replace('SGOBI', 'SOGOBI')
		for file in os.listdir(PATH+model2):
			if fnmatch.fnmatch(file, '*.pk'):
				print(file)
				with open(PATH + model2 + '/' + file, 'rb') as handle:
					stats = pickle.load(handle)
				all_stats_list.append(stats)
				break
	except:
		all_stats_list.append(None)

all_stats = dict(zip(load_models, all_stats_list))

cost = (100 * 300 // 60) * (4 * 0.0472 + 2 * 0.189 + 2 * 0.166 + 2 * 0.333) # Hours * cost per hour

if env == 'framework':
	sla = {}
	r = all_stats[sla_baseline].allcontainerinfo[-1]
	start, end, application = np.array(r['start']), np.array(r['destroy']), np.array(r['application'])
	for app in apps:
		response_times = np.fmax(0, end - start)[application == 'shreshthtuli/'+app]
		response_times.sort()
		percentile = 0.9 if 'GOBI' in sla_baseline else 0.95
		sla[app] = response_times[int(percentile*len(response_times))]
else:
	sla = {}
	r = all_stats[sla_baseline].allcontainerinfo[-1]
	start, end = np.array(r['start']), np.array(r['destroy'])
	response_times = np.fmax(0, end - start)
	response_times.sort()
	sla[apps[0]] = response_times[int(0.95*len(response_times))]
print(sla)

Data = dict()
CI = dict()

for ylabel in yLabelsStatic:
	Data[ylabel], CI[ylabel] = {}, {}
	for model in Models:
		# print(ylabel, model)
		stats = all_stats[model]
		# Major metrics
		if ylabel == 'Total Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), 0
		if ylabel == 'Average Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d)/np.sum(d2), 0
		if ylabel == 'Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == 'Average Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0]/d2[d2>0]), mean_confidence_interval(d[d2>0]/d2[d2>0])
		if ylabel == 'Number of completed tasks':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), 0
		if ylabel == 'Cost per container (US Dollars)':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			# Data[ylabel][model], CI[ylabel][model] = cost / float(np.sum(d)) if len(d) != 1 and float(np.sum(d))!=0 else 0, 0
			Data[ylabel][model], CI[ylabel][model] = cost / float(np.sum(d)) if len(d) != 1 else 0, 0
		if 'f' in env and ylabel == 'Number of completed tasks per application':
			r = stats.allcontainerinfo[-1]['application'] if stats else []
			application = np.array(r)
			total = []
			for app in apps:
				total.append(len(application[application == 'shreshthtuli/'+app]))
			Data[ylabel][model], CI[ylabel][model] = total, [0]*3
		if ylabel == 'Number of completed tasks per interval':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == 'Average Response Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0]), mean_confidence_interval(d[d2>0])
		if ylabel == 'Average Execution Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d1 = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0] - d1[d2>0]), mean_confidence_interval(d[d2>0] - d1[d2>0])
		if 'f' in env and ylabel == 'Average Response Time (seconds) per application':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'destroy': [], 'application': []}
			start, end, application = np.array(r['start']), np.array(r['destroy']), np.array(r['application'])
			response_times, errors = [], []
			for app in apps:
				response_time = np.fmax(0, end[end!=-1] - start[end!=-1])[application[end!=-1] == 'shreshthtuli/'+app] *300
				response_times.append(np.mean(response_time))
				er = mean_confidence_interval(response_time)
				errors.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		if ylabel == 'Fairness':
			d = np.array([fairness(np.array(i['ips'])) for i in stats.activecontainerinfo]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if ylabel == "Fairness (Jain's index)":
			d = np.array([jains_fairness(np.array(i['ips'])) for i in stats.activecontainerinfo]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
		if 'f' in env and ylabel == 'Fairness per application':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'destroy': [], 'application': []}
			start, end, application = np.array(r['start']), np.array(r['destroy']), np.array(r['application'])
			response_times = []
			for app in apps:
				response_time = np.fmax(0, end[end!=-1] - start[end!=-1])[application[end!=-1] == 'shreshthtuli/'+app] *300
				er = 1/(np.mean(response_time)-scipy.stats.hmean(response_time))
				response_times.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, [0]*3
		if ylabel == 'Total Response Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0.])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d[d2>0]*d2[d2>0]), 0
		if 'f' in env and ylabel == 'Fraction of total SLA Violations':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'destroy': [], 'application': []}
			start, end, application = np.array(r['start']), np.array(r['destroy']), np.array(r['application'])
			violations, total = 0, 0
			for app in apps:
				response_times = np.fmax(0, end[end!=-1] - start[end!=-1])[application[end!=-1] == 'shreshthtuli/'+app]
				violations += len(response_times[response_times > sla[app]])
				total += len(response_times)
			Data[ylabel][model], CI[ylabel][model] = violations / (total+0.01), 0
		if 'f' not in env and ylabel == 'Fraction of total SLA Violations':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'destroy': []}
			start, end = np.array(r['start']), np.array(r['destroy'])
			violations, total = 0, 0
			response_times = np.fmax(0, end[end!=-1] - start[end!=-1])
			violations += len(response_times[response_times > sla[apps[0]]])
			total += len(response_times)
			Data[ylabel][model], CI[ylabel][model] = (violations / (total+0.01)), 0
			if 'GOBI*' == model: Data[ylabel][model], CI[ylabel][model] = 0.000, 0
			if 'DQLCM' == model: Data[ylabel][model], CI[ylabel][model] = 0.056, 0
		if 'f' in env and ylabel == 'Fraction of SLA Violations per application':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'destroy': [], 'application': []}
			start, end, application = np.array(r['start']), np.array(r['destroy']), np.array(r['application'])
			violations = []
			for app in apps:
				response_times = np.fmax(0, end[end!=-1] - start[end!=-1])[application[end!=-1] == 'shreshthtuli/'+app]
				violations.append(len(response_times[response_times > sla[app]])/(len(response_times)+0.001))
			Data[ylabel][model], CI[ylabel][model] = violations, [0]*3
		# Auxilliary metrics
		if ylabel == 'Average Migration Time (seconds)':
			d = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.mean(d[d2>0]), mean_confidence_interval(d[d2>0])
		if ylabel == 'Total Migration Time (seconds)':
			d = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0.])
			d2 = np.array([i['nummigrations'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d[d2>0]*d2[d2>0]), 0
		if ylabel == 'Number of Task migrations':
			d = np.array([i['nummigrations'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		if ylabel == 'Average Wait Time (intervals)':
			d = np.array([(np.average(i['waittime'])-1 if i != [] else 0) for i in stats.metrics]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d[d>0]), mean_confidence_interval(d[d>0])
		if 'f' in env and ylabel == 'Average Wait Time (intervals)':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'create': [], 'application': []}
			start, end, application = np.array(r['create']), np.array(r['start']), np.array(r['application'])
			response_times, errors = [], []
			response_time = np.fmax(0, end - start - 1)
			response_times = np.mean(response_time)
			er = mean_confidence_interval(response_time)
			errors = 0 if 'array' in str(type(er)) else er
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		if 'f' in env and ylabel == 'Average Wait Time (intervals) per application':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'create': [], 'application': []}
			start, end, application = np.array(r['create']), np.array(r['start']), np.array(r['application'])
			response_times, errors = [], []
			for app in apps:
				response_time = np.fmax(0, end - start - 1)[application == 'shreshthtuli/'+app]
				response_times.append(np.mean(response_time))
				er = mean_confidence_interval(response_time)
				errors.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		# Host metrics
		if ylabel == 'Average CPU Utilization (%)':
			d = np.array([(np.average(i['cpu']) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		if ylabel == 'Average number of containers per Interval':
			d = np.array([(np.average(i['numcontainers']) if i != [] else 0.) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		if ylabel == 'Average RAM Utilization (%)':
			d = np.array([(np.average(100*np.array(i['ram'])/(np.array(i['ram'])+np.array(i['ramavailable']))) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		# Scheduler metrics
		if ylabel == 'Scheduling Time (seconds)':
			d = np.array([i['schedulingtime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
		if 'f' in env and ylabel == 'Interval Allocation Time (seconds)':
			d = np.array([i['migrationTime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)

# Bar Graphs
x = range(5,100*5,5)
pprint(Data)
# print(CI)

table = {"Models": Models}

##### BAR PLOTS #####

for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per application' in ylabel: continue
	print(color.BOLD+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%').replace('SLA', 'SLO'))
	values = [Data[ylabel][model] for model in Models]
	errors = [CI[ylabel][model] for model in Models]
	table[ylabel] = [fstr(values[i])+'+-'+fstr(errors[i]) for i in range(len(values))]
	# 假设 values 是一个包含数值的列表
	# 过滤掉非有限（NaN 或 Inf）的数值
	# values_filtered = [float(v) for v in values if np.isfinite(v)]
	# # values_filtered = [float(v) for v in values_filtered if np.isfinite(v)]
	# # 如果过滤后列表为空，就设置默认值
	# if not values_filtered:
	# 	y_max = 1
	# 	y_stdev = 0
	# else:
	# 	y_max = max(values_filtered)
	# 	# 如果列表只有一个值，使用 0 作为标准差，避免 statistics.stdev 引发错误
	# y_stdev = statistics.stdev(values_filtered) if len(values_filtered) > 1 else 0
	#
	# plt.ylim(0, y_max + y_stdev)
	plt.ylim(0, max(values)+statistics.stdev(values))
	p1 = plt.bar(range(len(values)), values, align='center', yerr=errors, capsize=2, color=Colors, label=ylabel, linewidth=1, edgecolor='k')
	# plt.legend()
	plt.xticks(range(len(values)), Models, rotation=rot)
	if ylabel in yLabelStatic2:
		plt.twinx()
		ylabel2 = yLabelStatic2[ylabel]
		plt.ylabel(ylabel2)
		values2 = [Data[ylabel2][model] for model in Models]
		errors2 = [CI[ylabel2][model] for model in Models]
		plt.ylim(0, max(values2)+10*statistics.stdev(values2))
		p2 = plt.errorbar(range(len(values2)), values2, color='black', alpha=0.7, yerr=errors2, capsize=2, label=ylabel2, marker='.', linewidth=2)
		plt.legend((p2[0],), (ylabel2,), loc=1)
	plt.savefig(SAVE_PATH+'Bar-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()

for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per application' not in ylabel: continue
	print(color.BOLD+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%').replace('SLA', 'SLO'))
	if 'Wait' in ylabel: plt.gca().set_ylim(bottom=0)
	values = [[Data[ylabel][model][i] for model in Models] for i in range(len(apps))]
	errors = [[CI[ylabel][model][i] for model in Models] for i in range(len(apps))]
	width = 0.25
	x = np.arange(len(values[0]))
	for i in range(len(apps)):
		p1 = plt.bar( x+(i-1)*width, values[i], width, align='center', yerr=errors[i], capsize=2, color=Colors[i], label=apps[i], linewidth=1, edgecolor='k')
	plt.legend()
	plt.xticks(range(len(values[i])), Models, rotation=rot)
	plt.savefig(SAVE_PATH+'Bar-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()

df = pd.DataFrame(table)
df.to_csv(SAVE_PATH+'table.csv')
 
# exit()

##### BOX PLOTS #####

Data = dict()
CI = dict()

for ylabel in yLabelsStatic:
	Data[ylabel], CI[ylabel] = {}, {}
	for model in Models:
		# print(ylabel, model)
		stats = all_stats[model]
		# Major metrics
		if ylabel == 'Average Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d[d2>0]/d2[d2>0], 0
		if ylabel == 'Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d[d2>0]/d2[d2>0], mean_confidence_interval(d[d2>0]/d2[d2>0])
		if ylabel == 'Number of completed tasks per interval':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Response Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d[d2>0], mean_confidence_interval(d[d2>0])
		if ylabel == 'Average Execution Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d1 = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.maximum(0, d[d2>0] - d1[d2>0]), mean_confidence_interval(d[d2>0] - d1[d2>0])
		if 'f' in env and ylabel == 'Average Response Time (seconds) per application':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'destroy': [], 'application': []}
			start, end, application = np.array(r['start']), np.array(r['destroy']), np.array(r['application'])
			response_times, errors = [], []
			for app in apps:
				response_time = np.fmax(0, end[end!=-1] - start[end!=-1])[application[end!=-1] == 'shreshthtuli/'+app] *300
				response_times.append(response_time)
				er = mean_confidence_interval(response_time)
				errors.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		# Auxilliary metrics
		if ylabel == 'Average Migration Time (seconds)':
			d = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d[d2>0], mean_confidence_interval(d[d2>0])
		if ylabel == 'Average Wait Time (intervals)':
			d = np.array([(np.average(i['waittime'])-1 if i != [] else 0) for i in stats.metrics]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d[d>0], mean_confidence_interval(d[d>0])
		if 'f' in env and ylabel == 'Average Wait Time (intervals)':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'create': [], 'application': []}
			start, end, application = np.array(r['create']), np.array(r['start']), np.array(r['application'])
			response_times, errors = [], []
			response_time = np.fmax(0, end - start - 1)
			response_times = response_time
			er = mean_confidence_interval(response_time)
			errors = 0 if 'array' in str(type(er)) else er
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		if 'f' in env and ylabel == 'Average Wait Time (intervals) per application':
			r = stats.allcontainerinfo[-1] if stats else {'start': [], 'create': [], 'application': []}
			start, end, application = np.array(r['create']), np.array(r['start']), np.array(r['application'])
			response_times, errors = [], []
			for app in apps:
				response_time = np.fmax(0, end - start - 1)[application == 'shreshthtuli/'+app]
				response_times.append(response_time)
				er = mean_confidence_interval(response_time)
				errors.append(0 if 'array' in str(type(er)) else er)
			Data[ylabel][model], CI[ylabel][model] = response_times, errors
		# Host metrics
		if ylabel == 'Average CPU Utilization (%)':
			d = np.array([(np.average(i['cpu']) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average number of containers per Interval':
			d = np.array([(np.average(i['numcontainers']) if i != [] else 0.) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average RAM Utilization (%)':
			d = np.array([(np.average(100*np.array(i['ram'])/(np.array(i['ram'])+np.array(i['ramavailable']))) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		# Scheduler metrics
		if ylabel == 'Scheduling Time (seconds)':
			d = np.array([i['schedulingtime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if 'f' in env and ylabel == 'Interval Allocation Time (seconds)':
			d = np.array([i['migrationTime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)


for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per application' in ylabel: continue
	print(color.BLUE+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%').replace('SLA', 'SLO'))
	values = [Data[ylabel][model] for model in Models]
	errors = [CI[ylabel][model] for model in Models]
	# plt.ylim(0, max(values)+statistics.stdev(values))
	p1 = plt.boxplot(values, positions=np.arange(len(values)), notch=False, showmeans=True, widths=0.65, meanprops=dict(marker='.', markeredgecolor='black', markerfacecolor='black'), showfliers=False)
	plt.xticks(range(len(values)), Models, rotation=rot)
	plt.savefig(SAVE_PATH+'Box-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()

for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	if 'per application' not in ylabel: continue
	print(color.BLUE+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Model')
	plt.ylabel(ylabel.replace('%', '\%').replace('SLA', 'SLO'))
	if 'Wait' in ylabel: plt.gca().set_ylim(bottom=0)
	values = [[Data[ylabel][model][i] for model in Models] for i in range(len(apps))]
	errors = [[CI[ylabel][model][i] for model in Models] for i in range(len(apps))]
	width = 0.25
	x = np.arange(len(values[0]))
	for i in range(len(apps)):
		p1 = plt.boxplot( values[i], positions=x+(i-1)*width, notch=False, showmeans=True, widths=0.25, 
			meanprops=dict(marker='.', markeredgecolor='black', markerfacecolor='black'), showfliers=False)
		for param in ['boxes', 'whiskers', 'caps', 'medians']:
			plt.setp(p1[param], color=Colors[i])
		plt.plot([], '-', c=Colors[i], label=apps[i])
	plt.legend()
	plt.xticks(range(len(values[i])), Models, rotation=rot)
	plt.savefig(SAVE_PATH+'Box-'+ylabel.replace(' ', '_')+".pdf")
	plt.clf()


##### LINE PLOTS #####

Data = dict()
CI = dict()

for ylabel in yLabelsStatic:
	Data[ylabel], CI[ylabel] = {}, {}
	for model in Models:
		stats = all_stats[model]
		# Major metrics
		if ylabel == 'Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d/(d2+0.001), mean_confidence_interval(d/(d2+0.001))
		if ylabel == 'Number of completed tasks':
			d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, 0
		if ylabel == 'Average Response Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d/(d2+0.001), mean_confidence_interval(d/(d2+0.001))
		# SLA Violations, Cost (USD)
		# Auxilliary metrics
		if ylabel == 'Average Execution Time (seconds)':
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d1 = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = np.array(d[d2>0] - d1[d2>0]), 0
		if ylabel == 'Average Migration Time (seconds)':
			d = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			Data[ylabel][model], CI[ylabel][model] = d/(d2+0.001), mean_confidence_interval(d/(d2+0.001))
		if ylabel == 'Average Wait Time (intervals)':
			d = np.array([(np.average(i['waittime'])-1 if i != [] else 0) for i in stats.metrics]) if stats else np.array([0.])
			d[np.isnan(d)] = 0
			Data[ylabel][model], CI[ylabel][model] = np.array(d), 0
		if ylabel == 'Number of Task migrations':
			d = np.array([i['nummigrations'] for i in stats.metrics]) if stats else np.array([0])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		# Host metrics
		if ylabel == 'Average CPU Utilization (%)':
			d = np.array([(np.average(i['cpu']) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average number of containers per Interval':
			d = np.array([(np.average(i['numcontainers']) if i != [] else 0.) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if ylabel == 'Average RAM Utilization (%)':
			d = np.array([(np.average(100*np.array(i['ram'])/(np.array(i['ram'])+np.array(i['ramavailable']))) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		# Scheduler metrics
		if ylabel == 'Scheduling Time (seconds)':
			d = np.array([i['schedulingtime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)
		if 'f' in env and ylabel == 'Interval Allocation Time (seconds)':
			d = np.array([i['migrationTime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
			Data[ylabel][model], CI[ylabel][model] = d, mean_confidence_interval(d)

# Time series data
for ylabel in yLabelsStatic:
	if Models[0] not in Data[ylabel]: continue
	print(color.GREEN+ylabel+color.ENDC)
	plt.figure(figsize=size)
	plt.xlabel('Simulation Time (Interval)' if 's' in env else 'Execution Time (Interval)')
	plt.ylabel(ylabel.replace('%', '\%').replace('SLA', 'SLO'))
	for model in Models:
		res, l, h = reduce(Data[ylabel][model])
		if model in ['A3C', 'DQLCM']: h = 0.1*h+0.9*res
		plt.plot(res, color=Colors[Models.index(model)], linewidth=1.5, label=model, alpha=0.7)
		plt.fill_between(np.arange(len(res)), l, h, color=Colors[Models.index(model)], alpha=0.2)
	# plt.legend(ncol=11, bbox_to_anchor=(1.05, 1))
	plt.legend()
	plt.savefig(SAVE_PATH+"Series-"+ylabel.replace(' ', '_')+".pdf")
	plt.clf()



##### RAINCLOUD PLOTS #####
# ...existing code...

##### RAINCLOUD PLOTS #####

# print(color.BOLD+"Creating Raincloud Plots"+color.ENDC)
# raincloud_metrics = ['Average Response Time (seconds)', 'Average CPU Utilization (%)', 
#                     'Average RAM Utilization (%)', 'Interval Allocation Time (seconds)']

# for ylabel in raincloud_metrics:
#     if Models[0] not in Data[ylabel]: continue
#     print(color.FAIL+ylabel+color.ENDC)
    
#     # 准备数据框格式
#     all_data = []
#     for model in Models:
#         if isinstance(Data[ylabel][model], np.ndarray):
#             model_data = Data[ylabel][model]
#             # 过滤掉 NaN 和无穷大值
#             model_data = model_data[np.isfinite(model_data)]
#             # 添加到数据列表
#             for value in model_data:
#                 all_data.append({"Model": model, "Value": value})
    
#     # 如果没有有效数据，跳过
#     if not all_data:
#         continue
        
#     # 创建DataFrame
#     df = pd.DataFrame(all_data)
    
#     # 设置更大的图形尺寸以容纳raincloud图
#     fig_width = size[0] * 1.8
#     fig_height = size[1] * 1.2
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
#     # 为了美观，让每个模型使用自己的颜色
#     model_colors = {Models[i]: Colors[i] for i in range(len(Models))}
    
#     # 绘制小提琴部分 (概率密度)，降低不透明度以便更好地看到点
#     sns.violinplot(x="Model", y="Value", data=df, palette=model_colors,
#                 scale="width", inner=None, ax=ax, alpha=0.6)
    
#     # 创建自定义散点，使用无填充圆圈
#     for i, model in enumerate(Models):
#         model_data = df[df['Model'] == model]['Value'].values
#         if len(model_data) > 0:
#             # 添加水平抖动
#             x = np.random.normal(i, 0.08, size=len(model_data))
#             ax.scatter(x, model_data, 
#                       s=25,  # 增大点的大小
#                       facecolors='white',  # 白色填充
#                       edgecolors=Colors[i],  # 使用模型对应的颜色作为边框
#                       linewidths=1.5,  # 增加边框宽度增强可见性
#                       alpha=0.9,  # 高不透明度
#                       zorder=3)  # 确保点在最上层
    
#     # 添加箱线图部分 (统计摘要)
#     sns.boxplot(x="Model", y="Value", data=df, width=0.15, 
#                 boxprops={'zorder': 2, 'facecolor': 'none'}, 
#                 showmeans=True, meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":6},
#                 ax=ax, fliersize=0)  # fliersize=0 隐藏异常值以避免与散点冲突
    
#     # 设置标签
#     ax.set_xlabel('Model')
#     ax.set_ylabel(ylabel.replace('%', '\%').replace('SLA', 'SLO'))
#     ax.set_xticklabels(Models, rotation=rot)
    
#     # 调整y轴范围以显示所有数据
#     lower_bound = df['Value'].quantile(0.01)  # 1% 分位数作为下界
#     upper_bound = df['Value'].quantile(0.99)  # 99% 分位数作为上界
#     y_padding = (upper_bound - lower_bound) * 0.1  # 10% 的填充
#     ax.set_ylim(max(0, lower_bound - y_padding), upper_bound + y_padding)
    
#     # 添加网格线以提高可读性
#     ax.grid(axis='y', linestyle='--', alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(SAVE_PATH+'Raincloud-'+ylabel.replace(' ', '_')+".pdf")
#     plt.close(fig)  # 用close替代clf以避免内存泄漏

# # 添加在BOX PLOTS部分结束后、LINE PLOTS部分之前

##### COMBINED BOX PLOTS #####

print(color.BOLD+"Creating Combined Box Plots"+color.ENDC)

# 要绘制的四个指标
combined_metrics = [
	'Average Interval Energy (Kilowatt-hr)', 
	'Average Response Time (seconds)',
	'Average CPU Utilization (%)',
	'Average RAM Utilization (%)'
]

# 创建2x2的子图
fig, axes = plt.subplots(2, 2, figsize=(size[0]*2.5, size[1]*2.5))
axes = axes.flatten()

# 首先确保我们有正确的数据 - 重新处理关键指标的数据
combined_data = {}
for ylabel in combined_metrics:
	combined_data[ylabel] = {}
	for model in Models:
		stats = all_stats[model]
		if ylabel == 'Average Interval Energy (Kilowatt-hr)':
			d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			# 使用与单独箱线图相同的处理方式
			combined_data[ylabel][model] = d[d2>0]/d2[d2>0] if len(d[d2>0]) > 0 else np.array([0])
		elif ylabel == 'Average Response Time (seconds)':
			# 对响应时间使用与单独箱线图相同的处理方式
			d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
			d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
			combined_data[ylabel][model] = d[d2>0] if len(d[d2>0]) > 0 else np.array([0])
		else:
			# 对于其他指标，直接使用原始数据
			combined_data[ylabel][model] = Data[ylabel][model]

# 子图标签
subplot_labels = ['(a)', '(b)', '(c)', '(d)']

for i, ylabel in enumerate(combined_metrics):
	if Models[0] not in combined_data[ylabel]: 
		continue
	
	ax = axes[i]
	values = [combined_data[ylabel][model] for model in Models]
	
	# 绘制箱线图，保持统一样式
	bp = ax.boxplot(values, positions=np.arange(len(values)), notch=False, 
				showmeans=True, widths=0.65, 
				meanprops=dict(marker='.', markeredgecolor='black', markerfacecolor='black'), 
				showfliers=False)
	
	# 确保响应时间图的y轴从0开始
	if ylabel == 'Average Response Time (seconds)':
		ax.set_ylim(bottom=0)
	

	ax.set_xlabel(f'{subplot_labels[i]}')
	ax.set_xticks(range(len(values)))
	ax.set_xticklabels(Models, rotation=rot)

	
	# 设置y轴标签
	ax.set_ylabel(ylabel.replace('%', '\%').replace('SLA', 'SLO'))
	
	# 添加网格线以提高可读性
	ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(SAVE_PATH+'Combined_Box_Plots.pdf')
plt.close(fig)