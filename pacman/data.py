# date.py module helps user in disaggregating the AC energy consumption
# from the smart meter data. 

# System library
import os
import sys
import glob
import numpy
import pandas
import matplotlib.pyplot as plt

# Thresholds to extract AC cycles
USAGETHRESHOLD = 2000
INITIALCUTOFF = 300	# Padding across the estimated on and off times of AC

# extract_usage function is designed to take out an AC data from the complete data frame 
# for a given set of parameters which are as follows:
# data - complete data
# current_usage_on_time - estimated time when user turned on the AC
# current_usage_off_time - estimated time when user turned off the AC
# start_index - 
# usage_count - counter to track the number of AC usages encountered in the dataset
# i - 
# usage_csv_files - 
# usage_png_files - 
# This function returns the updated start_index and the usage_count after extracting AC usage data.
def extract_usage(data, current_usage_on_time, current_usage_off_time, \
				start_index, usage_count, i, usage_csv_files, usage_png_files):
	
	# Padding for estimated on and off time for an AC cycle
	start_time = current_usage_on_time - INITIALCUTOFF #- initTime - 300 (Because of NILM error)
	end_time =  current_usage_off_time + INITIALCUTOFF  #- initTime + 300 (Because of NILM error)

	# Seprate out AC usage data from the complete data frame
	usage_data = data.loc[start_time:end_time]
	
	# Update start_index & usage_count
	start_index.append(i)
	usage_count = usage_count + 1

	# Get AC set temperature from the data - assuming it to be closer to minimum temperature achieved.
	usage_data['setTemperature'] = numpy.round(numpy.amin(usage_data['int_temperature'].values))

	# CSV (for data) and PNG (for image) file names
	if usage_count < 10: # To align it with two-digits
		usage_csv_file = usage_csv_files + "0" + str(usage_count) + ".csv"
		usage_png_file = usage_png_files + "0" + str(usage_count) + ".png"
	else:
		usage_csv_file = usage_csv_files + str(usage_count) + ".csv"    
		usage_png_file = usage_png_files + str(usage_count) + ".png"

	# Save data to CSV file
	usage_data.to_csv(usage_csv_file)

	# Save internal temperature data to a PNG file
	plt.plot(usage_data["int_temperature"].index, usage_data["int_temperature"].values)
	plt.savefig(usage_png_file, format='png')
	plt.clf()

	# Return the updated counters
	return start_index, usage_count

# Extract AC usages from a specific file
def extract_AC_usages_file(data_file):
	
	# WARNING! - Pending validation of file and its data

	# Read data from the file
	data = pandas.read_csv(data_file, index_col=0)
	filename = os.path.basename(data_file).split(".")[0]
	
	# IMPROVEMENT: Instead of passing single invalid entries, we can provide a list and program can match all 
	# those entries present in the list for both on and off time
	
	# Timestamps at which AC compressor turns On. In second line, we are extracting invalid entries such as 0
	on_time_raw = numpy.unique(data['onTime'].values)
	on_time = numpy.delete(on_time_raw, numpy.where(on_time_raw==0)[0])
	
	# Timestamps at which AC compressor turns Off. In second line, we are extracting invalid entries such as 0
	off_time_raw = numpy.unique(data['offTime'].values)
	off_time = numpy.delete(off_time_raw, numpy.where(off_time_raw==0)[0])
	
	# Timestamp at which AC turns on first time in the dataset
	init_time = data.index.values[0]

	# USELESS: Currently looks like that this information is not required
#	d = {'OnTime': on_time, 'OffTime': off_time}
#	on_off_data = pandas.DataFrame(data = d)
#	on_off_data.to_csv(OnOffTimeFile,index_label="S.No")

	# Index to maintain start time of each AC usage - A usage is a period for which AC remain turned on. 
	start_index = []
	start_index.append(0)
	
	# Counter for number of usages - A usage is a period for which AC remain turned on. 
	usage_count = 0

	# Output directories. Create if doesn't exist. 
	result_csv_dir = "Results/Usages/" + filename + "/CSV/"
	result_png_dir = "Results/Usages/" + filename + "/PNG/"

	if not os.path.exists(result_csv_dir):
		os.makedirs(result_csv_dir)

	if not os.path.exists(result_png_dir):
		os.makedirs(result_png_dir)

	usage_csv_files = result_csv_dir + "usage_"
	usage_png_files = result_png_dir + "usage_"

	# Loop to iterate over each AC compressor cycle on time. Algorithm says, whenever compressor takes more than
	# USAGETHRESHOLD (in Seconds) to turn on after being turned off, then it is a seprate AC usage. 

	for i in range(1, len(on_time)):
		
		next_on_time = on_time[i]
		current_off_time = off_time[i-1]
		
		# If next compressor on time is more than USAGETHRESHOLD then, it is a separate AC usage
		if next_on_time - current_off_time > USAGETHRESHOLD:

			# Get AC On and Off time from the data
			current_usage_index = start_index[usage_count]
			
			current_usage_on_time = on_time[current_usage_index]
			current_usage_off_time = current_off_time

			# Extract usage from the data and save in CSV and PNG formats.
			start_index, usage_count = extract_usage(data, current_usage_on_time, current_usage_off_time, \
				start_index, usage_count, i, usage_csv_files, usage_png_files)
	
	# For last cycle which can't be included in the loop
	current_off_time = off_time[i]
	current_usage_index = start_index[usage_count]
	
	current_usage_on_time = on_time[current_usage_index]
	current_usage_off_time = current_off_time

	start_index, usage_count = extract_usage(data, current_usage_on_time, current_usage_off_time, \
				start_index, usage_count, i, usage_csv_files, usage_png_files)

# WARNING! - Check for input validation
# Iterate over all data files present in a directory if given as a path
def extract_AC_usages_dir(data_dir):
	# Get files as a list present in provided directory
	path_files = data_dir + "*.csv"

	files = glob.glob(path_files)
	files.sort()
	
	# Iterate through each file
	for data_file in files:
		extract_AC_usages_file(data_file)

def get_actual_data(prediction_file, r):
	actual_data = pandas.read_csv(prediction_file, index_col=0)
	
	# Preprocessing
	temp = len(actual_data)
	actual_data = actual_data[r:]
	actual_data = actual_data[:temp-r]
	
	Tint = actual_data["int_temperature"].values
	S = actual_data["status"].values

	Tset = numpy.unique(actual_data["setTemperature"].values)[0]
	
	return Tint, S, Tset

def filer_less_varying(data):
	int_temp = data["int_temperature"].values
	Tsmooth = pandas.ewma(int_temp, span=15)

	usage_var = numpy.var(Tsmooth)*100
	print usage_var
	if usage_var > 20:
		return 1
	
	return 0

def filer_less_running(data):
	status = data["status"].values
	
	usage_energy = numpy.sum(status)
	print usage_energy
	if usage_energy < 1800:
		return 0
	
	return 1

def filter_ac_usages_dir(data_dir, output_dir):
	# Get files as a list present in provided directory
	path_files = data_dir + "*.csv"

	files = glob.glob(path_files)
	files.sort()
	
	usage_count = 0
	
	result_csv_dir = output_dir + "/CSV/"
	result_png_dir = output_dir + "/PNG/"

	if not os.path.exists(result_csv_dir):
		os.makedirs(result_csv_dir)

	if not os.path.exists(result_png_dir):
		os.makedirs(result_png_dir)

	# Iterate through each file
	for data_file in files:
		data = pandas.read_csv(data_file, index_col=0)
		
		c = filer_less_running(data)
		usage_count = usage_count + c

		if c == 1:
			if usage_count < 10:
				usage_csv_file = result_csv_dir + "usage_0" + str(usage_count) + ".csv"
				usage_png_file = result_png_dir + "usage_0" + str(usage_count) + ".png"
			else:
				usage_csv_file = result_csv_dir + "usage_" + str(usage_count) + ".csv"    
				usage_png_file = result_png_dir + "usage_" + str(usage_count) + ".png"

			data.to_csv(usage_csv_file)
			# Save internal temperature data to a PNG file
			plt.plot(data["int_temperature"].index, data["int_temperature"].values)
			plt.savefig(usage_png_file, format='png')
			plt.clf()