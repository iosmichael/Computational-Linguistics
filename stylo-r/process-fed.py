import os

source_file = open("federalist-all", 'r')

dest_file = None

paper = 0

map_fnames = {}

for line in source_file :
    if line.startswith('FEDERALIST') :
        if dest_file :
            dest_file.close()
        paper += 1
        dest_file = open("federalist" + str(paper), 'w')
    if dest_file and all([not line.startswith(x) for x in ["HAMILTON AND MADISON", "HAMILTON", "MADISON", "JAY"]]) :
        dest_file.write(line)
    else:
    	if line.rstrip() == "HAMILTON OR MADISON":
    		line = "DISPUTED"
    	map_fnames["federalist" + str(paper)] = line.rstrip()

if dest_file :
    dest_file.close()

print map_fnames
# for k in map_fnames.keys():
# 	os.rename(k, map_fnames[k]+"_"+k)