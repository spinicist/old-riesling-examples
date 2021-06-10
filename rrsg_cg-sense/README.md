# Example - CG-SENSE Challenge

Scripts to test `riesling` on the ISMRM CG-SENSE Reproducibility challenge. (_CG-SENSE revisited: Results from the first ISMRM reproducibility challenge, [10.1002/mrm.28569](https://onlinelibrary.wiley.com/doi/10.1002/mrm.28569)_). Note that this is 2D data and while useful for quick demonstration, it does not show the true power of `riesling` which is 3D reconstruction.

## 1. Download data
You can automatically download the data from Zenodo using
```sh
bash download_rrsg_data.sh
```

Or download it manually and place it in a subfolder called `rrsg_data`. Download link: https://zenodo.org/record/3975887#.X9EE6l7gokg


## 2. Convert to riesling format
To convert the data to `riesling` .h5 format run
```sh
python3 convert_rrsg_data.py
```
This script will create a folder `riesling_data` 
The conversion script is also a useful reference to understand the `riesling` data format.

## 3. Run riesling recon
To perform CG-SENSE recon with `riesling` run
```sh
bash riesling_recon.sh
```

## 4. View the images
Have a look at the images in your preferred nifti viewer such as fsleyes
```sh
fsleyes riesling_recon/rrsg_challenge_brain_cgsense-cg.nii 
fsleyes riesling_recon/rrsg_reference_brain_cgsense-cg.nii
```