"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

import nibabel as nb 

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    savedir = args.model_path.replace(".pt", "/")
    os.makedirs(savedir,exist_ok=True)

    for sampling in range(0,args.num_samples//args.batch_size):

        
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 1, 32, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
        )
        
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # sample = sample.permute(0, 2, 3, 1)
        sample = sample.permute(0,3,4,2,1)
        sample = sample.contiguous()

        
        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        
        
        for i, sam in enumerate(sample): 
            
            # save 
            sam = sam.cpu().numpy()
                    
            if dist.get_rank() == 0:

                shape_str = "x".join([str(x) for x in sam.shape])
                out_path = os.path.join(savedir, f"samples_{shape_str}_{sampling}_{i}.npz")
                logger.log(f"saving to {out_path}")
                # if args.class_cond:
                #     np.savez(out_path, arr, label_arr)
                # else:
                #     np.savez(out_path, arr)
                    
                # save as nifti additionally

                imo = nb.Nifti1Image(sam, affine=np.eye(4))
                nb.save(imo, out_path.replace(".npz", ".nii.gz"))
                    
        th.cuda.empty_cache()
        
        
        # if args.class_cond:
        #     gathered_labels = [
        #         th.zeros_like(classes) for _ in range(dist.get_world_size())
        #     ]
        #     dist.all_gather(gathered_labels, classes)
        #     all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        # logger.log(f"created {len(all_images) * args.batch_size} samples")
        
        
            
    
    # arr = np.concatenate(all_images, axis=0)
    # arr = arr[: args.num_samples]
    # if args.class_cond:
    #     label_arr = np.concatenate(all_labels, axis=0)
    #     label_arr = label_arr[: args.num_samples]
    # if dist.get_rank() == 0:
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    #     logger.log(f"saving to {out_path}")
    #     if args.class_cond:
    #         np.savez(out_path, arr, label_arr)
    #     else:
    #         np.savez(out_path, arr)
            
        # # save as nifti additionally
        # for i, im in enumerate(arr):
        #     imo = nb.Nifti1Image(im, affine=np.eye(4))
        #     nb.save(imo, out_path.replace(".npz", f"_sample{i}.nii.gz"))
            

        

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=20, #10000,
        batch_size=5, #16,
        use_ddim=False,
        model_path="/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/diffusion/experiments/s20240209_oecorr/guided-diffusion/trained/openai-2024-02-28-05-25-58-668429/model070500.pt",  #"/tmp/openai-2024-02-28-05-25-58-668429/model012500.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
