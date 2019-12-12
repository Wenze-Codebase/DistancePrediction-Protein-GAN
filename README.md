# DistancePrediction-Protein-GAN

# Tips for usage:
## "GAN"
The folder 'GAN' contains training and model exportation codes. Once the training data (features and lables) were prepared (usually the "TFRecord format"), training process could be executed via "python main.py --mode train --XXX", where related parameters could be controled by --XXX (see main.py for details). The model exportation could be conduced through "python main.py --mode export --XXX".        
## "distance"
This folder is for the final inference once the model (generator) is exported.
## "feature"
The folder 'feature' contains codes for feature generation. There are some 3rd-party software and databases which are need to be set by users. Notice:
1. Package needs to be installed: BioPython & prody;
2. Blast package is also needed, install-command recommended by us is "sudo apt-get install ncbi-blast+";
3. Make sure you have got the tensorflow package of the newest version;
4. You should config your own addresses of HHLIB, CCMPRED, DEEPCNF, SPIDER3 and UNIPROT20. Blanks filled with 'XXXX' should be overrided.

