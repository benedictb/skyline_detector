import os
import sys
import shutil

out = open('skyline_data.txt','w+')
out.write('')
for i,file in enumerate(['shanghai_out.txt', 'london_out.txt','nyc_out.txt','chicago_out.txt']):
	with open(file) as f:
		for line in f:
			fid = line.split(',')[1]
			direct = line.split(',')[0]
			out.write('{},{}\n'.format(direct + '/' + fid + '_' +direct , i))

