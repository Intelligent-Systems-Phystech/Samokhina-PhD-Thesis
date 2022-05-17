#!/bin/bash

mkdir data

wget --no-check-certificate https://gin.g-node.org/v-goncharenko/neiry-demons/raw/master/nery_demons_dataset.zip
unzip nery_demons_dataset.zip -d data/demons
