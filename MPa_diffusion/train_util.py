import copy
import functools
import os
from typing import Dict, List, Tuple, Optional, Union
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import blobfile as bf

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .modalities import ModalFusionProcessor
from .property_utils import PropertyConstraintHandler

# MPaDiffusion specific imports
from .stress_strain_encoder import StressStrainEncoder
from .physics_consistency import PhysicsConsistencyLoss

INITIAL_LOG_LOSS_SCALE = 20.0


class MPaMultiModal3DTrainLoop:
    """
    Enhanced training loop for MPaDiffusion: Multi-modal Property-Aware Diffusion Model
    for 3D Reconstruction and On-demand Design.

    Major extensions include:
    - Multi-modal conditioning (2D slices, segmentation masks, stress-strain data)
    - Property-aware variance scheduling
    - Physics-consistent loss functions
    - Enhanced evaluation metrics
    - Anisotropy-aware sampling
    """

    def __init__(
            self,
            *,
            model: th.nn.Module,
            classifier: Optional[th.nn.Module],
            diffusion: object,
            dataloader: th.utils.data.DataLoader,
            batch_size: int,
            microbatch: int = -1,
            base_lr: float = 1e-4,
            ema_rates: Union[List[float], str] = "0.999,0.9999",
            log_interval: int = 10,
            save_interval: int = 1000,
            resume_checkpoint: Optional[str] = None,
            use_fp16: bool = False,
            fp16_scale_growth: float = 1e-3,
            schedule_sampler: Optional[object] = None,
            weight_decay: float = 1e-4,
            lr_anneal_steps: int = 0,
            modal_types: List[str] = ["2d_slice", "segmentation_mask", "stress_strain"],
            property_specs: Optional[Dict] = None,
            grad_clip: float = 1.0,
            # MPaDiffusion specific parameters
            stress_strain_dim: int = 512,
            physics_config: Optional[Dict] = None,
            anisotropy_config: Optional[Dict] = None,
            multi_modal_config: Optional[Dict] = None
    ):
        # Core components initialization
        self.model = model
        self.classifier = classifier
        self.diffusion = diffusion
        self.dataloader = dataloader

        # Training configuration
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.base_lr = base_lr
        self.ema_rates = self._parse_ema_rates(ema_rates)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.grad_clip = grad_clip

        # MPaDiffusion: Enhanced multi-modal and property-aware components
        self.modal_processor = ModalFusionProcessor(modal_types)
        self.property_handler = PropertyConstraintHandler(property_specs) if property_specs else None

        # MPaDiffusion: Stress-strain encoder and physics enforcer
        self.stress_strain_encoder = StressStrainEncoder(
            input_dim=12,  # 6 stress + 6 strain components
            hidden_dims=[64, 128, 256],
            output_dim=stress_strain_dim
        )

        # MPaDiffusion: Physics consistency loss
        if physics_config:
            self.physics_enforcer = PhysicsConsistencyLoss(
                youngs_modulus=physics_config.get('youngs_modulus', 1.0),
                poissons_ratio=physics_config.get('poissons_ratio', 0.3),
                stress_strain_weight=physics_config.get('stress_strain_weight', 0.2),
                hookes_law_weight=physics_config.get('hookes_law_weight', 0.1)
            )
        else:
            self.physics_enforcer = None

        # MPaDiffusion: Anisotropy-aware configuration
        self.anisotropy_config = anisotropy_config or {}
        self.multi_modal_config = multi_modal_config or {}

        # Training state
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.device = dist_util.dev()
        self.use_ddp = th.cuda.is_available()

        # Initialize all components
        self._setup_parameters()
        self._init_mixed_precision()
        self._init_optimizer()
        self._init_ema_parameters()
        self._setup_ddp()
        self._init_lr_scheduler()

        # MPaDiffusion: Training monitor for enhanced metrics
        self.training_monitor = MPaTrainingMonitor()

    def _parse_ema_rates(self, ema_rates: Union[List[float], str]) -> List[float]:
        """Parse EMA rates from string or list with validation."""
        if isinstance(ema_rates, str):
            rates = [float(rate.strip()) for rate in ema_rates.split(",")]
        else:
            rates = ema_rates

        # Validate EMA rates
        for rate in rates:
            if not (0 < rate < 1):
                raise ValueError(f"Invalid EMA rate: {rate}. Must be between 0 and 1.")

        return rates

    def _setup_parameters(self) -> None:
        """Load and synchronize model parameters across distributed processes."""
        resume_checkpoint = self._find_resume_checkpoint()
        if resume_checkpoint:
            self.resume_step = self._parse_checkpoint_step(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"Loading MPaDiffusion model from checkpoint: {resume_checkpoint}")
                state_dict = dist_util.load_state_dict(resume_checkpoint, map_location=self.device)

                # Handle potential parameter name changes in MPaDiffusion
                state_dict = self._adapt_state_dict_for_mpadiffusion(state_dict)
                self.model.load_state_dict(state_dict)

        dist_util.sync_params(self.model.parameters())

    def _adapt_state_dict_for_mpadiffusion(self, state_dict: Dict) -> Dict:
        """Adapt state dictionary for MPaDiffusion-specific parameters."""
        # Handle parameter name changes or additions for MPaDiffusion
        adapted_dict = {}
        for key, value in state_dict.items():
            # Example adaptation: handle renamed parameters
            if key.startswith('old_prefix.'):
                new_key = key.replace('old_prefix.', 'mpadiffusion.')
                adapted_dict[new_key] = value
            else:
                adapted_dict[key] = value
        return adapted_dict

    def _init_mixed_precision(self) -> None:
        """Initialize mixed precision training handler for MPaDiffusion."""
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
            initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE
        )

    def _init_optimizer(self) -> None:
        """Initialize optimizer with parameter grouping for MPaDiffusion."""
        # Group parameters for different learning rates if needed
        param_groups = self._create_parameter_groups()

        self.opt = AdamW(
            param_groups,
            lr=self.base_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        if self.resume_step > 0:
            self._load_optimizer_state()

    def _create_parameter_groups(self) -> List[Dict]:
        """Create parameter groups for optimized training in MPaDiffusion."""
        # Separate parameters for different components
        main_params = []
        stress_strain_params = []
        physics_params = []

        for name, param in self.model.named_parameters():
            if 'stress_strain' in name or 'property_aware' in name:
                stress_strain_params.append(param)
            elif 'physics' in name or 'mechanical' in name:
                physics_params.append(param)
            else:
                main_params.append(param)

        param_groups = [
            {'params': main_params, 'lr': self.base_lr},
            {'params': stress_strain_params, 'lr': self.base_lr * 0.1},  # Lower LR for stability
            {'params': physics_params, 'lr': self.base_lr * 0.5}  # Moderate LR for physics
        ]

        return param_groups

    def _init_ema_parameters(self) -> None:
        """Initialize EMA parameters for MPaDiffusion."""
        if self.resume_step > 0:
            self.ema_params = [self._load_ema_params(rate) for rate in self.ema_rates]
        else:
            self.ema_params = [copy.deepcopy(self.mp_trainer.master_params) for _ in self.ema_rates]

    def _setup_ddp(self) -> None:
        """Setup distributed data parallel for MPaDiffusion."""
        if self.use_ddp:
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=256,
                find_unused_parameters=True  # Allow for multi-modal parameters
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn("Distributed training without CUDA - gradients may not sync properly")
            self.ddp_model = self.model

    def _init_lr_scheduler(self) -> None:
        """Initialize learning rate scheduler for MPaDiffusion."""
        if self.lr_anneal_steps > 0:
            self.lr_scheduler = LambdaLR(
                self.opt,
                lr_lambda=self._create_lr_lambda()
            )
        else:
            self.lr_scheduler = None

    def _create_lr_lambda(self):
        """Create learning rate lambda function with warmup for MPaDiffusion."""

        def lr_lambda(step):
            total_step = step + self.resume_step
            if total_step < 1000:  # Warmup phase
                return float(total_step) / 1000
            else:
                return max(0.0, 1.0 - total_step / self.lr_anneal_steps)

        return lr_lambda

    def _find_resume_checkpoint(self) -> Optional[str]:
        """Find latest resume checkpoint for MPaDiffusion."""
        auto_checkpoint = find_latest_checkpoint()
        return auto_checkpoint or self.resume_checkpoint

    def _parse_checkpoint_step(self, filename: str) -> int:
        """Parse training step from checkpoint filename."""
        try:
            return int(''.join(filter(str.isdigit, filename)))
        except ValueError:
            return 0

    def _load_ema_params(self, rate: float) -> List[th.Tensor]:
        """Load EMA parameters from checkpoint for MPaDiffusion."""
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = self._find_resume_checkpoint()
        if main_checkpoint:
            ema_checkpoint = self._find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
            if ema_checkpoint and dist.get_rank() == 0:
                logger.log(f"Loading EMA {rate} from checkpoint: {ema_checkpoint}")
                state_dict = dist_util.load_state_dict(ema_checkpoint, map_location=self.device)
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self) -> None:
        """Load optimizer state from checkpoint for MPaDiffusion."""
        main_checkpoint = self._find_resume_checkpoint()
        if main_checkpoint:
            opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"opt{self.resume_step:06d}.pt")
            if bf.exists(opt_checkpoint):
                logger.log(f"Loading optimizer state from: {opt_checkpoint}")
                state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=self.device)
                self.opt.load_state_dict(state_dict)

    def _find_ema_checkpoint(self, main_checkpoint: str, step: int, rate: float) -> Optional[str]:
        """Find EMA checkpoint corresponding to main checkpoint."""
        if not main_checkpoint:
            return None
        filename = f"ema_{rate:.4f}_{step:06d}.pt"
        path = bf.join(bf.dirname(main_checkpoint), filename)
        return path if bf.exists(path) else None

    def run_training(self) -> None:
        """Main training loop for MPaDiffusion with enhanced monitoring."""
        data_iter = iter(self.dataloader)
        loss_tracker = MPaLossTracker()

        while self._should_continue_training():
            try:
                batch_data = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch_data = next(data_iter)

            # Process batch and run training step with MPaDiffusion enhancements
            loss_components, generated_sample = self._run_training_iteration(batch_data)
            loss_tracker.update(loss_components)

            # MPaDiffusion: Enhanced logging with physics metrics
            if self.step % self.log_interval == 0:
                self._log_training_state(loss_tracker, generated_sample)
                loss_tracker.reset()

            # MPaDiffusion: Enhanced checkpointing
            if self.step % self.save_interval == 0:
                self._save_checkpoint()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            self.step += 1
            self._update_lr_scheduler()

        # Save final checkpoint if needed
        if (self.step - 1) % self.save_interval != 0:
            self._save_checkpoint()

    def _should_continue_training(self) -> bool:
        """Check if training should continue for MPaDiffusion."""
        if not self.lr_anneal_steps:
            return True
        return (self.step + self.resume_step) < self.lr_anneal_steps

    def _run_training_iteration(self, batch_data: Dict) -> Tuple[Dict[str, float], Optional[th.Tensor]]:
        """Run single training iteration with MPaDiffusion enhancements."""
        # Unpack and process multi-modal inputs with MPaDiffusion processing
        modalities = batch_data["modalities"]
        properties = batch_data.get("properties", None)

        # MPaDiffusion: Enhanced modal fusion with stress-strain encoding
        fused_input = self.modal_processor.fuse(modalities).to(self.device)

        # Prepare conditions including properties and stress-strain data
        cond = self._prepare_conditions(batch_data, properties)

        # Forward-backward pass with physics-aware loss
        loss_components, sample = self._forward_backward_pass(fused_input, cond)

        # Optimize and update EMA
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema_parameters()
            self._clip_gradients()

        return loss_components, sample

    def _prepare_conditions(self, batch_data: Dict, properties: Optional[Dict]) -> Dict:
        """Prepare condition dictionary including properties and stress-strain encoding for MPaDiffusion."""
        cond = {}

        # Basic conditions
        if "class_labels" in batch_data:
            cond["y"] = batch_data["class_labels"].to(self.device)

        # MPaDiffusion: Property constraints
        if properties is not None and self.property_handler is not None:
            cond["properties"] = self.property_handler.process_constraints(properties)

        # MPaDiffusion: Stress-strain encoding
        if "stress_strain" in batch_data and self.stress_strain_encoder is not None:
            stress_strain_data = batch_data["stress_strain"].to(self.device)
            encoded_stress_strain = self.stress_strain_encoder(stress_strain_data)
            cond["encoded_stress_strain"] = encoded_stress_strain

        # MPaDiffusion: Multi-modal fusion conditions
        if "multi_modal_data" in batch_data:
            cond["multi_modal"] = batch_data["multi_modal_data"].to(self.device)

        return cond

    def _forward_backward_pass(self, input_data: th.Tensor, cond: Dict) -> Tuple[Dict[str, float], Optional[th.Tensor]]:
        """Forward pass with loss calculation and backward pass for MPaDiffusion."""
        self.mp_trainer.zero_grad()

        # Initialize loss accumulators
        loss_accumulators = {
            "seg": 0.0, "cls": 0.0, "rec": 0.0,
            "prop": 0.0, "physics": 0.0, "anisotropy": 0.0
        }
        sample = None

        # Process microbatches
        for i in range(0, input_data.shape[0], self.microbatch):
            micro_input = input_data[i:i + self.microbatch]
            micro_cond = {k: self._prepare_micro_condition(v, i) for k, v in cond.items()}
            last_microbatch = (i + self.microbatch) >= input_data.shape[0]

            # Sample timesteps and calculate weights
            t, weights = self.schedule_sampler.sample(micro_input.shape[0], self.device)

            # Compute losses using MPaDiffusion-enhanced function
            compute_losses = functools.partial(
                self.diffusion.MPa_training_losses,  # MPaDiffusion enhanced method
                self.ddp_model,
                micro_input,
                t,
                multi_modal_conditions=micro_cond
            )

            # Forward pass with appropriate sync
            if last_microbatch or not self.use_ddp:
                loss_outputs = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    loss_outputs = compute_losses()

            # Unpack losses and sample
            losses_dict, generated_sample = loss_outputs
            sample = generated_sample if sample is None else sample

            self._update_sampler(t, losses_dict)

            # Calculate weighted total loss
            total_loss = self._compute_weighted_total_loss(losses_dict, weights, micro_input, generated_sample, micro_cond)

            # Accumulate individual losses for logging
            self._accumulate_losses(loss_accumulators, losses_dict, weights, micro_input, generated_sample, micro_cond)

            # Backward pass
            self.mp_trainer.backward(total_loss)

        # Average losses across microbatches
        num_microbatches = input_data.shape[0] // self.microbatch
        averaged_losses = {k: v / num_microbatches for k, v in loss_accumulators.items()}

        return averaged_losses, sample

    def _prepare_micro_condition(self, condition: th.Tensor, start_idx: int) -> th.Tensor:
        """Prepare condition for microbatch with proper device placement."""
        end_idx = start_idx + self.microbatch
        if isinstance(condition, th.Tensor):
            return condition[start_idx:end_idx].to(self.device)
        return condition

    def _compute_weighted_total_loss(self, losses_dict: Dict, weights: th.Tensor,
                                     input_data: th.Tensor, sample: th.Tensor,
                                     cond: Dict) -> th.Tensor:
        """Compute weighted total loss with MPaDiffusion enhancements."""
        # Base diffusion loss
        base_loss = (losses_dict["loss"] * weights).mean()

        # MPaDiffusion: Additional loss components
        additional_loss = 0.0

        # Reconstruction loss
        rec_loss = self._compute_reconstruction_loss(input_data, sample)
        additional_loss += 0.1 * rec_loss

        # Property-aware loss
        if self.property_handler and "properties" in cond:
            prop_loss = self._compute_property_loss(sample, cond)
            additional_loss += 0.1 * prop_loss

        # Physics consistency loss
        if self.physics_enforcer and "encoded_stress_strain" in cond:
            physics_loss = self._compute_physics_consistency_loss(sample, cond)
            additional_loss += 0.2 * physics_loss

        # Anisotropy regularization
        if self.anisotropy_config.get('enabled', False):
            anisotropy_loss = self._compute_anisotropy_regularization(sample)
            additional_loss += 0.05 * anisotropy_loss

        return base_loss + additional_loss

    def _accumulate_losses(self, accumulators: Dict, losses_dict: Dict, weights: th.Tensor,
                           input_data: th.Tensor, sample: th.Tensor, cond: Dict) -> None:
        """Accumulate individual losses for logging."""
        # Standard losses
        accumulators["seg"] += (losses_dict.get("mse", th.tensor(0.0)) * weights).mean().item()
        accumulators["cls"] += (losses_dict.get("kl", th.tensor(0.0)) * weights).mean().item()

        # MPaDiffusion: Additional losses
        accumulators["rec"] += self._compute_reconstruction_loss(input_data, sample).item()

        if self.property_handler and "properties" in cond:
            accumulators["prop"] += self._compute_property_loss(sample, cond).item()

        if self.physics_enforcer and "encoded_stress_strain" in cond:
            accumulators["physics"] += self._compute_physics_consistency_loss(sample, cond).item()

        if self.anisotropy_config.get('enabled', False):
            accumulators["anisotropy"] += self._compute_anisotropy_regularization(sample).item()

    def _update_sampler(self, t: th.Tensor, losses_dict: Dict) -> None:
        """Update loss-aware sampler if applicable."""
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, losses_dict["loss"].detach())

    def _compute_reconstruction_loss(self, input_data: th.Tensor, sample: th.Tensor) -> th.Tensor:
        """Compute 3D reconstruction loss for MPaDiffusion."""
        return F.l1_loss(self._normalize_3d_volume(sample), self._normalize_3d_volume(input_data[:, :1]))

    def _compute_property_loss(self, sample: th.Tensor, cond: Dict) -> th.Tensor:
        """Compute property constraint loss for MPaDiffusion."""
        if "properties" not in cond or self.property_handler is None:
            return th.tensor(0.0, device=sample.device)
        return self.property_handler.compute_loss(sample, cond["properties"])

    def _compute_physics_consistency_loss(self, sample: th.Tensor, cond: Dict) -> th.Tensor:
        """Compute physics consistency loss based on Hooke's law for MPaDiffusion."""
        if "encoded_stress_strain" not in cond or self.physics_enforcer is None:
            return th.tensor(0.0, device=sample.device)

        # Extract predicted stress-strain from sample (simplified)
        predicted_stress_strain = self._extract_stress_strain_from_sample(sample)
        target_stress_strain = cond["encoded_stress_strain"]

        return self.physics_enforcer(predicted_stress_strain, target_stress_strain)

    def _compute_anisotropy_regularization(self, sample: th.Tensor) -> th.Tensor:
        """Compute anisotropy regularization for MPaDiffusion."""
        anisotropy_index = self._calculate_anisotropy_index(sample)
        target_anisotropy = self.anisotropy_config.get('target', 30.0)
        return F.mse_loss(th.tensor(anisotropy_index, device=sample.device),
                          th.tensor(target_anisotropy, device=sample.device))

    def _extract_stress_strain_from_sample(self, sample: th.Tensor) -> th.Tensor:
        """Extract stress-strain information from generated sample."""
        # Simplified implementation - in practice, this should be model-specific
        return sample[:, -12:]  # Assuming last 12 channels represent stress-strain

    def _calculate_anisotropy_index(self, sample: th.Tensor) -> float:
        """Calculate anisotropy index for MPaDiffusion."""
        # Simplified implementation
        if sample.dim() == 5:  # 3D volume [batch, channels, depth, height, width]
            variances = []
            for dim in [2, 3, 4]:  # Depth, height, width dimensions
                slice_var = sample.var(dim=dim).mean()
                variances.append(slice_var.item())
            return max(variances) / (min(variances) + 1e-8)
        return 1.0  # Isotropic by default

    def _normalize_3d_volume(self, volume: th.Tensor) -> th.Tensor:
        """Normalize 3D volume data to [0, 1] range."""
        vol_min = volume.min()
        vol_max = volume.max()
        return (volume - vol_min) / (vol_max - vol_min + 1e-8)

    def _update_ema_parameters(self) -> None:
        """Update EMA parameters for all rates."""
        for rate, params in zip(self.ema_rates, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)


    def _clip_gradients(self) -> None:
        """Clip gradients to prevent explosion using global norm clipping.

        This method applies gradient clipping to all master parameters in the mixed precision trainer.
        Clipping is based on the global norm of gradients, which helps stabilize training by preventing
        gradient explosion in deep networks and multi-modal models like MPaDiffusion.

        Notes:
            - Only clips if grad_clip > 0 is set during initialization
            - Uses PyTorch's clip_grad_norm_ which computes the norm of all gradients together
            - Particularly important for property-aware diffusion models with complex loss functions
        """
        if self.grad_clip > 0:
            # Apply gradient clipping to all master parameters
            th.nn.utils.clip_grad_norm_(
                self.mp_trainer.master_params,  # Parameters from mixed precision trainer
                max_norm=self.grad_clip,  # Maximum allowed norm value
                norm_type=2.0  # L2 norm (Euclidean norm) for stable clipping
            )
            # Log clipping event for debugging and monitoring
            if self.step % self.log_interval == 0:
                logger.logkv_mean("grad_norm", self._compute_gradient_norm())
                logger.logkv_mean("grad_clip_applied", 1.0)
