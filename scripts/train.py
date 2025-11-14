"""
MPaDiffusion: Multi-modal Property-Aware Diffusion Model for 3D Reconstruction and On-demand Design
Enhanced implementation with multi-modal conditioning, physics consistency, and property-aware variance scheduling.
"""

import time
import torch as th
import os
import socket
import sys
import numpy as np
import argparse
import torch
import torch.distributed as dist
import torch.nn.parallel
from torch.utils.data.distributed import DistributedSampler

sys.path.append("")
sys.path.append("scripts")

# MPaDiffusion specific imports
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.shaprloader import SHAPRDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

# MPaDiffusion enhanced components
from guided_diffusion.mpa_components import (
    MPaTrainLoop,
    MultiModalConditioning,
    PhysicsConsistencyLoss,
    PropertyAwareVarianceScheduler
)


def main():
    """MPaDiffusion"""
    args = create_argparser().parse_args()
    
    # Enhanced distributed setup for MPaDiffusion
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    dist.init_process_group(
        backend='nccl', 
        init_method="env://", 
        rank=args.local_rank, 
        world_size=args.world_size
    )
    
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{args.local_rank}')
    
    logger.configure()
    
    # MPaDiffusion: Enhanced model creation with multi-modal support
    logger.log("Creating MPaDiffusion model and diffusion process...")
    
    # Create MPaDiffusion model with enhanced parameters
    model, diffusion = create_mpa_model_and_diffusion(
        **args_to_dict(args, mpa_model_and_diffusion_defaults().keys())
    )
    
    # Calculate and log model parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.log(f"Model parameters: {params:,}")
    
    # Enhanced device placement and distributed training
    model.to(device)
    model = th.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[rank],
        find_unused_parameters=args.multi_modal_conditioning  # Handle multi-modal branches
    )
    
    # MPaDiffusion: Property-aware schedule sampler
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, 
        diffusion,  
        maxt=args.diffusion_steps
    )
    
    # MPaDiffusion: Enhanced data loading with multi-modal support
    logger.log("Creating multi-modal data loader...")
    ds = SHAPRDataset(
        args.data_dir, 
        test_flag=False,
        multi_modal=args.multi_modal_conditioning,
        stress_strain_dim=args.stress_strain_dim
    )
    
    train_sampler = DistributedSampler(ds)
    dataloader = th.utils.data.DataLoader(
        ds,
        sampler=train_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    data_iter = iter(dataloader)
    
    # MPaDiffusion: Enhanced training loop with physics consistency
    logger.log("Starting MPaDiffusion training...")
    
    # Use MPaDiffusion enhanced training loop
    MPaTrainLoop(
        model=model,
        diffusion=diffusion,
        data=data_iter,
        dataloader=dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        # MPaDiffusion specific parameters
        use_mpa=args.use_mpa,
        property_aware_var=args.property_aware_var,
        physics_consistency=args.physics_consistency,
        multi_modal_conditioning=args.multi_modal_conditioning,
        stress_strain_dim=args.stress_strain_dim,
        gamma_t=args.gamma_t,
        classifier_free_guidance=args.classifier_free_guidance,
        guidance_weight=args.guidance_weight,
        anisotropy_regularization=args.anisotropy_regularization,
        property_loss_weight=args.property_loss_weight,
        physics_loss_weight=args.physics_loss_weight,
        grad_clip=args.grad_clip
    ).run_loop()


def create_mpa_model_and_diffusion(
    image_size,
    num_channels,
    num_res_blocks,
    # Standard parameters
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0.0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    # Diffusion parameters
    diffusion_steps=1000,
    noise_schedule="linear",
    timestep_respacing="",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    # MPaDiffusion specific parameters
    use_mpa=True,
    property_aware_var=True,
    physics_consistency=True,
    multi_modal_conditioning=True,
    stress_strain_dim=512,
    modal_fusion_dim=256,
    anisotropy_regularization=True,
    classifier_free_guidance=True,
    guidance_weight=1.0
):
    """
    Create MPaDiffusion model and diffusion process with enhanced capabilities.
    """
    # Enhanced model creation with MPaDiffusion modifications
    if use_mpa:
        from guided_diffusion.mpa_unet import MPaUNetModel
        from guided_diffusion.mpa_diffusion import MPaGaussianDiffusion
        
        # Create MPaDiffusion U-Net model
        model = MPaUNetModel(
            image_size=image_size,
            in_channels=3,
            model_channels=num_channels,
            out_channels=(3 if not learn_sigma else 6),
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(2 if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            # MPaDiffusion parameters
            use_multi_modal=multi_modal_conditioning,
            stress_strain_dim=stress_strain_dim,
            modal_fusion_dim=modal_fusion_dim
        )
        
        # Create MPaDiffusion process
        diffusion = MPaGaussianDiffusion(
            steps=diffusion_steps,
            learn_sigma=learn_sigma,
            noise_schedule=noise_schedule,
            use_kl=use_kl,
            predict_xstart=predict_xstart,
            rescale_timesteps=rescale_timesteps,
            rescale_learned_sigmas=rescale_learned_sigmas,
            timestep_respacing=timestep_respacing,
            # MPaDiffusion parameters
            use_property_aware=property_aware_var,
            physics_consistency=physics_consistency,
            stress_strain_dim=stress_strain_dim,
            classifier_free_guidance=classifier_free_guidance,
            guidance_weight=guidance_weight
        )
    else:
        # Fallback to standard implementation
        model, diffusion = create_model_and_diffusion(
            image_size=image_size,
            class_cond=class_cond,
            learn_sigma=learn_sigma,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            diffusion_steps=diffusion_steps,
            noise_schedule=noise_schedule,
            timestep_respacing=timestep_respacing,
            use_kl=use_kl,
            predict_xstart=predict_xstart,
            rescale_timesteps=rescale_timesteps,
            rescale_learned_sigmas=rescale_learned_sigmas,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_fp16=use_fp16,
            use_new_attention_order=use_new_attention_order,
        )
    
    return model, diffusion


def mpa_model_and_diffusion_defaults():
    """
    Default parameters for MPaDiffusion framework.
    """
    defaults = {
        # Core model parameters
        "image_size": 64,
        "num_channels": 32,
        "num_res_blocks": 2,
        "num_heads": 4,
        "num_heads_upsample": -1,
        "num_head_channels": -1,
        "attention_resolutions": "16,8",
        "channel_mult": "",
        "dropout": 0.0,
        "class_cond": False,
        "use_checkpoint": False,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_fp16": False,
        "use_new_attention_order": False,
        
        # Diffusion parameters
        "learn_sigma": True,
        "diffusion_steps": 1000,
        "noise_schedule": "linear",
        "timestep_respacing": "",
        "use_kl": False,
        "predict_xstart": False,
        "rescale_timesteps": False,
        "rescale_learned_sigmas": False,
        
        # MPaDiffusion specific parameters
        "use_mpa": True,
        "property_aware_var": True,
        "physics_consistency": True,
        "multi_modal_conditioning": True,
        "stress_strain_dim": 512,
        "modal_fusion_dim": 256,
        "gamma_t": 0.1,
        "classifier_free_guidance": True,
        "guidance_weight": 1.0,
        "anisotropy_regularization": True,
        "property_loss_weight": 0.1,
        "physics_loss_weight": 0.2,
        "grad_clip": 1.0,
        "num_workers": 4,
        "world_size": 4
    }
    return defaults


def create_argparser():
    """
    Create argument parser with MPaDiffusion enhanced parameters.
    """
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        use_fp16=True,
        fp16_scale_growth=1e-3,
    )
    
    # Merge with MPaDiffusion defaults
    mpa_defaults = mpa_model_and_diffusion_defaults()
    defaults.update(mpa_defaults)
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    # MPaDiffusion specific arguments
    parser.add_argument('--local-rank', type=int, default=0, 
                       help='Local rank for distributed training')
    parser.add_argument('--world-size', type=int, default=4,
                       help='World size for distributed training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    return parser


# MPaDiffusion enhanced training monitoring
class MPaTrainingMonitor:
    """
    Enhanced training monitor for MPaDiffusion with physics metrics tracking.
    """
    
    def __init__(self):
        self.losses = {
            'mse': [], 'kl': [], 'physics': [], 'property': [], 'total': []
        }
        self.anisotropy_indices = []
        self.stress_strain_accuracy = []
    
    def update(self, loss_dict, anisotropy_idx=None, stress_strain_acc=None):
        """Update training metrics."""
        for key, value in loss_dict.items():
            if key in self.losses:
                self.losses[key].append(value)
        
        if anisotropy_idx is not None:
            self.anisotropy_indices.append(anisotropy_idx)
        if stress_strain_acc is not None:
            self.stress_strain_accuracy.append(stress_strain_acc)
    
    def get_summary(self):
        """Get training summary with MPaDiffusion metrics."""
        summary = {}
        for key, values in self.losses.items():
            if values:
                summary[f'avg_{key}_loss'] = np.mean(values[-100:])  # Last 100 steps
        
        if self.anisotropy_indices:
            summary['avg_anisotropy'] = np.mean(self.anisotropy_indices[-100:])
        if self.stress_strain_accuracy:
            summary['avg_stress_strain_acc'] = np.mean(self.stress_strain_accuracy[-100:])
        
        return summary


# Enhanced SHAPRDataset for MPaDiffusion
class MPaSHAPRDataset(SHAPRDataset):
    """
    Enhanced SHAPR dataset with multi-modal support for MPaDiffusion.
    """
    
    def __init__(self, data_dir, test_flag=False, multi_modal=True, stress_strain_dim=512):
        super().__init__(data_dir, test_flag)
        self.multi_modal = multi_modal
        self.stress_strain_dim = stress_strain_dim
        
    def __getitem__(self, index):
        """Enhanced getitem with multi-modal data support."""
        data = super().__getitem__(index)
        
        if self.multi_modal:
            # Add multi-modal conditioning data
            data['modalities'] = {
                'slices': data.get('slices'),
                'masks': data.get('masks'),
                'stress_strain': self._process_stress_strain(data.get('properties'))
            }
        
        return data
    
    def _process_stress_strain(self, properties):
        """Process stress-strain data for MPaDiffusion conditioning."""
        if properties is None:
            return torch.zeros(self.stress_strain_dim)
        
        # Convert properties to stress-strain representation
        # Implementation depends on specific data format
        return torch.tensor(properties, dtype=torch.float32)


if __name__ == "__main__":
    main()