#!/bin/bash
#
# Automatically downloads the rrsg_challenge data from Zenodo

if [ ! -d rrsg_data ]; then
    mkdir rrsg_data
fi

files=("rawdata_brain_radial_96proj_12ch.h5" "rawdata_spiral_ETH.h5")

for f in ${files[@]}; do
    if [ ! -e rrsg_data/$f ]; then
        echo "Downloading ${f} from Zenodo"
        curl -L "https://zenodo.org/record/3975887/files/${f}" --output rrsg_data/${f}
    fi
done
