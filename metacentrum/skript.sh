#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=16gb:scratch_local=8gb:ngpus=1:gpu_cap=cuda60:cuda_version=12.4
#PBS -l walltime=00:50:00
#PBS -N TensorFlow
# initialize the required application (e.g. Python, version 3.4.1, compiled by gcc)

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR=/storage/brno2/home/adam2024/pokus # substitute username and path to to your real username and path

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# copy input file "h2o.com" to scratch directory
# if the copy operation fails, issue error message and exit
cp $DATADIR/proj.py  $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }

mkdir $SCRATCHDIR/data
mkdir $SCRATCHDIR/model

cp -r $DATADIR/data/* $SCRATCHDIR/data || { echo >&2 "Error while copying input file(s) data!"; exit 2; }

cd $SCRATCHDIR

singularity exec --nv /cvmfs/singularity.metacentrum.cz/NGC/TensorFlow\:23.03-tf2-py3.SIF python $DATADIR/proj.py > skuska.out

# move the output to user's DATADIR or exit in case of failure
cp skuska.out $DATADIR/maybe.out || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }
cp model/siamese_model.h5 $DATADIR/model/siamese_model.h5 || { echo >&2 "Result file copying failed (with  a code $?) !!"; exit 4; }

# clean the SCRATCH directory
clean_scratch
#install python libraries -> https://docs.metacentrum.cz/software/install-software/#python-packages