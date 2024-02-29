# train 
cd /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/diffusion/experiments/s20240209_oecorr/guided-diffusion
conda activate dps_mpi
python scripts/image_train.py

# files saved to /tmp in ankara
cat /tmp/openai-2024-02-28-05-12-16-486120/log.txt

# test 
python scripts/image_sample.py