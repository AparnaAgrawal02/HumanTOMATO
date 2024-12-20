#!/bin/bash

path1="/scratch/aparna/experiments/temos/BSL/embeddings/val/epoch_199/"
path2="/scratch/aparna/experiments/temos/BSL/embeddings/val/epoch_1199/"
path3="/scratch/aparna/experiments/temos/BSL/embeddings/val/epoch_1299/"


for protocal in A B D
do
    echo "**protocal" $protocal"**"
    for retrieval_type in T2M M2T
    do
        echo $retrieval_type
        python retrieval.py --retrieval_type $retrieval_type --protocal $protocal --expdirs $path1 $path2 $path3 
    done
done
