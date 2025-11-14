"""
Generate image samples from a diffusion model for evaluation.
Simplified version with core functionality.
"""

import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.shaprloader import SHAPRDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ds = SHAPRDataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    data = iter(datal)
    
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        b, path = next(data)
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)
        slice_ID = path

        for i in range(args.num_ensemble):
            model_kwargs = {}
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            s = th.tensor(sample)
            output_path = f'./predictions/*****'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            th.save(s.cpu().detach(), output_path)

def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=5,
        batch_size=1,
        use_ddim=False,
        model_path="",
        image_size=64,
        num_channels=32,
        num_res_blocks=2,
        num_heads=1,
        num_ensemble=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()