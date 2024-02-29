# train 
cd /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/diffusion/experiments/s20240209_oecorr/guided-diffusion
conda activate dps_mpi
python scripts/image_train.py # training with f30840848f5fda925ad51227df51c3fdb52735e6 -> batch size of 1 
python scripts/image_train.py # training with next commit -> batch size of 32 









############################
# [archive] 3D training
############################
# train 
cd /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/diffusion/experiments/s20240209_oecorr/guided-diffusion
conda activate dps_mpi
python scripts/image_train.py

# files saved to /tmp in ankara
cat /tmp/openai-2024-02-28-05-12-16-486120/log.txt

# test 
python scripts/image_sample.py

# inferrence results: 
ls /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/diffusion/experiments/s20240209_oecorr/guided-diffusion/trained/openai-2024-02-28-05-25-58-668429/model070500
- 3D diffusion training with ~100 subjects does NOT work. 
- generated images are a gargle. 
- possible reasons of failure: 
    - not enough samples? 
    - the 3D network is too shallow to generate in 3D from noise? (possibly yes)
- future work: 
    - try to train a 2D network instead on cardiac images... (split my dataset into pngs...)
    - verify if 2D generation works... 
    - if it does -> this will support my hypothesis that 3D generation is too hard to generate. 

- how long will it take? 
    - 2hrs? 
