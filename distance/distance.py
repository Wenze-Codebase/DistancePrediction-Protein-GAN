import os
import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="2"

l2l1hot = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "E": 5, "Q": 6, "G": 7, "H": 8, "I": 9, "L": 10,
           "K": 11, "M": 12, "F": 13, "P": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19, "X": 20}
l2l = { "A": [np.arange(0, 1), 1],
        "R": [np.arange(1, 2), 1],
        "N": [np.arange(2, 3), 1],
        "D": [np.arange(3, 4), 1],
        "C": [np.arange(4, 5), 1],
        "E": [np.arange(5, 6), 1],
        "Q": [np.arange(6, 7), 1],
        "G": [np.arange(7, 8), 1],
        "H": [np.arange(8, 9), 1],
        "I": [np.arange(9, 10), 1],
        "L": [np.arange(10, 11), 1],
        "K": [np.arange(11, 12), 1],
        "M": [np.arange(12, 13), 1],
        "F": [np.arange(13, 14), 1],
        "P": [np.arange(14, 15), 1],
        "S": [np.arange(15, 16), 1],
        "T": [np.arange(16, 17), 1],
        "W": [np.arange(17, 18), 1],
        "Y": [np.arange(18, 19), 1],
        "V": [np.arange(19, 20), 1],
        "U": [np.arange(20, 21), 1],
        "O": [np.arange(20, 21), 1],
        "-": [np.arange(20, 21), 1],
        'X': [np.arange(0, 20), 0.05],
        'B': [np.arange(2, 4), 0.5],
        'Z': [np.arange(5, 7), 0.5],
        'J': [np.arange(9, 11), 0.5]}

with open('XXXX.fasta' , 'r') as protein_fa_file:
    contents=protein_fa_file.readlines()
name=contents[0].split()[0][1:]
sequence=contents[1].split()[0]
length = len(sequence)
deepcnf1 = np.loadtxt(
    'feature/'+name+'.ss3', dtype=float, usecols=range(3, 6))
deepcnf2 = np.loadtxt(
    'feature/'+name+'.ss8', dtype=float, usecols=range(3, 11))
deepcnf = np.hstack([deepcnf1, deepcnf2])
seq = np.array([l2l1hot[i] for i in sequence], dtype=float)
seql = np.zeros((length, 20))
seql[seq != 20, seq[seq != 20].astype(int)] = 1.
seql[seq == 20, :] = 1./20
f = open('feature/'+name+'.aln', 'r')
count = 0
freq = np.zeros((length, 21))
gap = np.zeros((length, length))
for line in f:
    line = line.strip()
    if line == '':
        continue
    for j in range(length):
        place, addv = l2l[line[j]]
        freq[j, place] += addv
    temp = np.array(list(line)) == '-'
    gap[np.ix_(temp, temp)] += 1
    count += 1
freq /= count
gap /= count
f.close()
spd = np.loadtxt('feature/'+name+'.spd33',dtype=float, usecols=range(3, 13)) #using all infos in spd33
ccmpred = np.loadtxt('feature/'+name+'-ccmpred.result')
mi = np.loadtxt('feature/'+name+'-mi.result')
pos = abs(np.arange(length)[np.newaxis]-np.arange(length)[:, np.newaxis])

f0d = np.array([length, count]).astype(np.float32)
f1d = np.concatenate((deepcnf, seql, freq, spd), axis=1).astype(np.float32)
f2d = np.stack((ccmpred, mi, gap, pos), axis=2).astype(np.float32)

featurefeed2d = np.concatenate([
  f2d,
  np.tile(f1d[np.newaxis], [length, 1, 1]),
  np.tile(f1d[:, np.newaxis], [1, length, 1]),
  np.tile(f0d[np.newaxis, np.newaxis], [length, length, 1]),
], axis=2)[np.newaxis]

max_list=np.loadtxt('max_130_channel')
min_list=np.loadtxt('min_130_channel')
for i in range(130):
    featurefeed2d[:,:,:,i]=featurefeed2d[:,:,:,i]*(2./(max_list[i]-min_list[i]))+(1-2.*max_list[i]/(max_list[i]-min_list[i]))

tf.reset_default_graph()
with tf.Session() as sess:        
    saver = tf.train.import_meta_graph('export.meta')
    saver.restore(sess,'export')
    graph = tf.get_default_graph()
    x=graph.get_operation_by_name('Input_without_preprocess').outputs[0]
    realpred=graph.get_operation_by_name('generator/Final_output').outputs[0]
    results = sess.run(realpred,feed_dict = {x:featurefeed2d})
    results=np.array(results)
    results_relative=results[0,:,:,0].copy()
    xx,yy=np.indices(results_relative.shape)
    results_relative[abs(xx-yy)<6]=0
    np.savetxt('result.npy',results_relative)            
        
