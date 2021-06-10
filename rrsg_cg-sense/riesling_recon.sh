#!/bin/bash
# Reconstruct brain data from the CG-SENSE reproducibility challenge
# Assumes you have the riesling binary in your path

if [ ! -d riesling_recon ]; then
    mkdir riesling_recon
fi

### Challenge data ###
# Brain
f=riesling_data/riesling_rawdata_brain_radial_96proj_12ch.h5
oname=riesling_recon/rrsg_challenge_brain
FOV=234
base="--fov=${FOV} --sdc=pipe -v --os 2.5 --magnitude"
riesling cg $base -i 10 -o ${oname}_cgsense $f

### Reference data ###
# Brain
f=riesling_data/riesling_rawdata_spiral_ETH.h5
oname=riesling_recon/rrsg_reference_brain
FOV=240
base="--fov=${FOV} --sdc=pipe -v --os 2.5 --magnitude"
riesling cg $base -i 10 -o ${oname}_cgsense $f
