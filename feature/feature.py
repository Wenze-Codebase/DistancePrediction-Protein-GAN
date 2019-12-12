import os

with open('XXXX.fasta' , 'r') as protein_fa_file:
	contents=protein_fa_file.readlines()
protein_name=contents[0].split()[0][1:]
sequence=contents[1].split()[0]
logfile=open('logfile','w')
output = os.popen('./3rd_party '+protein_name+' out_final -d 0 -t 40').read()
logfile.write('\n\n'+protein_name+'\n\n')
logfile.write(output+'\n')
# logfile.flush()
logfile.close()

if not os.path.exists('out_final/'+protein_name[iii]+'.spd33'):
	print("3rd-party running failed, please check details in logfile!!")
	exit(1)




