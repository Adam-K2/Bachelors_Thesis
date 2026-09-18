# Face recognition resilience against deepfake README

## Overview
This folder contains all files related to facial recognition experiments, including used dataset for evaluation, scripts to get results and finaly scripts to plot results into graphs.

## Structure
- **data**: Contains images from dataset CFP FP, which were used for evaluation of each face recognition algorithm and also our best performing siamese model.  
- **graphs**: Graphs showing the values from each tested face recognition algorithm, based on vector distances.
- **experiment_cos**: Script for evaluation of facial recognition experiments for cos metric.  
- **experiment_cos**: Script for evaluation of facial recognition experiments for Euklidean l2 metric.
- **results_cos**: Csv file of results for cos metric.
- **results_l2**: Csv file of results for Euclidean l2 metric.
- **table**: Script for evaluation for each tested face recognition algorithm. These results were used to created table in thesis.