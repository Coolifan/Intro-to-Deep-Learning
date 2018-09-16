These files give an example of a simple Tensorflow script working on the cluster using a GPU.  I recommend you test it, and modify the code to your own purposes to get use to running the cluster.

To test the files, first upload the files to your share in the cluster. Login using:  
ssh netid@dcc-slogin.oit.duke.edu  
Update run01.q with your path information.
And then test this with:  
sbatch run01.q  
After it finishes running, it should produce two files:  
01.err  
01.out  
The 01.out file contains the model performance.
