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
    if dest_file and all([not line.startswith(x) for x in ["HAMILTON OR MADISON", "HAMILTON", "MADISON", "JAY"]]) :
        dest_file.write(line)
    else:
    	if line[:-2] == "HAMILTON OR MADISON":
    		line = "DISPUTED\r\n"
        if paper == 0:
            continue
    	map_fnames["federalist" + str(paper)] = line[:-2]

if dest_file :
    dest_file.close()

for file_path in map_fnames.keys():
    print file_path
    os.rename(file_path, map_fnames[file_path]+"_"+file_path)