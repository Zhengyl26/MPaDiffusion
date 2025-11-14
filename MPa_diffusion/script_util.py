"""
MPaDiffusion: Multi-modal Property-Aware Diffusion Framework
Enhanced implementation for 3D reconstruction and on-demand design
"""

import argparse
import inspect
from typing import Dict, List, Optional, Tuple, Union
import torch as th
import torch.nn as nn

from . import mpa_gaussian_diffusion as mgd
from .mpa_respace import MPaSpacedDiffusion, mpa_space_timesteps
from .mpa_unet import (
    MPaUNetModel,
    MPaEncoderUNetModel,
    MPaSuperResModel,
    MultiModalFusionEncoder,
    PropertyAwareAttention
)

# Constants
MPA_NUM_CLASSES = 2
DEFAULT_STRESS_STRAIN_DIM = 512
MPA_INITIAL_LOG_LOSS_SCALE = 20.0


def create_mpa_diffusion_defaults() -> Dict:
    """Default parameters for MPaDiffusion framework."""
    return {
        # Core diffusion parameters
        "learn_sigma": True,
        "diffusion_steps": 1000,
        "noise_schedule": "linear",
        "timestep_respacing": "",
        "use_kl": False,
        "predict_xstart": False,
        "rescale_timesteps": False,
        "rescale_learned_sigmas": False,

        # MPaDiffusion enhancements
        "use_property_aware": True,
        "physics_consistency": True,
        "multi_modal_conditioning": True,
        "stress_strain_dim": DEFAULT_STRESS_STRAIN_DIM,
        "modal_fusion_dim": 256,
        "anisotropy_regularization": True,
        "classifier_free_guidance": True
    }


def create_mpa_model_and_diffusion(
        image_size: int,
        num_channels: int,
        num_res_blocks: int,
        # Standard parameters
        channel_mult: str = "",
        learn_sigma: bool = False,
        class_cond: bool = False,
        use_checkpoint: bool = False,
        attention_resolutions: str = "16",
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        dropout: float = 0.0,
        resblock_updown: bool = False,
        use_fp16: bool = False,
        use_new_attention_order: bool = False,
        # Diffusion parameters
        diffusion_steps: int = 1000,
        noise_schedule: str = "linear",
        timestep_respacing: str = "",
        use_kl: bool = False,
        predict_xstart: bool = False,
        rescale_timesteps: bool = False,
        rescale_learned_sigmas: bool = False,
        # MPaDiffusion specific parameters
        use_property_aware: bool = True,
        physics_consistency: bool = True,
        multi_modal_conditioning: bool = True,
        stress_strain_dim: int = DEFAULT_STRESS_STRAIN_DIM,
        modal_fusion_dim: int = 256,
        anisotropy_regularization: bool = True,
        classifier_free_guidance: bool = True,
        guidance_weight: float = 1.0
) -> Tuple[nn.Module, object]:
    """
    Create MPaDiffusion model and diffusion process.
    """
    # Create MPaDiffusion model
    model = create_mpa_model(
        image_size=image_size,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        # MPaDiffusion parameters
        use_property_aware=use_property_aware,
        multi_modal_conditioning=multi_modal_conditioning,
        stress_strain_dim=stress_strain_dim,
        modal_fusion_dim=modal_fusion_dim
    )

    # Create MPaDiffusion process
    diffusion = create_mpa_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        # MPaDiffusion parameters
        use_property_aware=use_property_aware,
        physics_consistency=physics_consistency,
        multi_modal_conditioning=multi_modal_conditioning,
        stress_strain_dim=stress_strain_dim,
        anisotropy_regularization=anisotropy_regularization,
        classifier_free_guidance=classifier_free_guidance,
        guidance_weight=guidance_weight
    )

    return model, diffusion


def create_mpa_model(
        image_size: int,
        num_channels: int,
        num_res_blocks: int,
        channel_mult: str = "",
        learn_sigma: bool = False,
        class_cond: bool = False,
        use_checkpoint: bool = False,
        attention_resolutions: str = "16",
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        dropout: float = 0.0,
        resblock_updown: bool = False,
        use_fp16: bool = False,
        use_new_attention_order: bool = False,
        # MPaDiffusion specific parameters
        use_property_aware: bool = True,
        multi_modal_conditioning: bool = True,
        stress_strain_dim: int = DEFAULT_STRESS_STRAIN_DIM,
        modal_fusion_dim: int = 256
) -> nn.Module:
    """
    Create MPaDiffusion U-Net model with multi-modal conditioning.
    """
    # Determine channel multiplier
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 4)
        else:
            raise ValueError(f"Unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(x) for x in channel_mult.split(","))

    # Calculate attention resolutions
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    # Create MPaDiffusion U-Net model
    return MPaUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(MPA_NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        # MPaDiffusion parameters
        use_property_aware=use_property_aware,
        multi_modal_conditioning=multi_modal_conditioning,
        stress_strain_dim=stress_strain_dim,
        modal_fusion_dim=modal_fusion_dim
    )


def create_mpa_gaussian_diffusion(
        *,
        steps: int = 1000,
        learn_sigma: bool = False,
        noise_schedule: str = "linear",
        use_kl: bool = False,
        predict_xstart: bool = False,
        rescale_timesteps: bool = False,
        rescale_learned_sigmas: bool = False,
        timestep_respacing: str = "",
        # MPaDiffusion specific parameters
        use_property_aware: bool = True,
        physics_consistency: bool = True,
        multi_modal_conditioning: bool = True,
        stress_strain_dim: int = DEFAULT_STRESS_STRAIN_DIM,
        anisotropy_regularization: bool = True,
        classifier_free_guidance: bool = True,
        guidance_weight: float = 1.0
) -> object:
    """
    Create MPaDiffusion Gaussian diffusion process.
    """
    # Get beta schedule
    betas = mgd.get_named_beta_schedule(noise_schedule, steps)

    # Determine loss type
    if use_property_aware and physics_consistency:
        loss_type = mgd.LossType.MPaPHYSICS
    elif use_kl:
        loss_type = mgd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = mgd.LossType.RESCALED_MSE
    else:
        loss_type = mgd.LossType.MSE

    # Handle timestep respacing
    if not timestep_respacing:
        timestep_respacing = [steps]

    # Determine model variance type
    if use_property_aware:
        model_var_type = mgd.ModelVarType.PROPERTY_AWARE
    elif not learn_sigma:
        model_var_type = mgd.ModelVarType.FIXED_LARGE
    else:
        model_var_type = mgd.ModelVarType.LEARNED_RANGE

    # Determine model mean type
    model_mean_type = (
        mgd.ModelMeanType.EPSILON if not predict_xstart else mgd.ModelMeanType.START_X
    )

    # Create MPaDiffusion process
    return MPaSpacedDiffusion(
        use_timesteps=mpa_space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        # MPaDiffusion parameters
        use_property_aware=use_property_aware,
        physics_consistency=physics_consistency,
        multi_modal_conditioning=multi_modal_conditioning,
        stress_strain_dim=stress_strain_dim,
        anisotropy_regularization=anisotropy_regularization,
        classifier_free_guidance=classifier_free_guidance,
        guidance_weight=guidance_weight
    )


def create_mpa_classifier_and_diffusion(
        image_size: int,
        classifier_use_fp16: bool,
        classifier_width: int,
        classifier_depth: int,
        classifier_attention_resolutions: str,
        classifier_use_scale_shift_norm: bool,
        classifier_resblock_updown: bool,
        classifier_pool: str,
        learn_sigma: bool,
        diffusion_steps: int,
        noise_schedule: str,
        timestep_respacing: str,
        use_kl: bool,
        predict_xstart: bool,
        rescale_timesteps: bool,
        rescale_learned_sigmas: bool,
        # MPaDiffusion parameters
        use_property_aware: bool = True,
        physics_consistency: bool = True,
        multi_modal_conditioning: bool = True,
        stress_strain_dim: int = DEFAULT_STRESS_STRAIN_DIM
) -> Tuple[nn.Module, object]:
    """
    Create MPaDiffusion classifier and diffusion process.
    """
    classifier = create_mpa_classifier(
        image_size=image_size,
        classifier_use_fp16=classifier_use_fp16,
        classifier_width=classifier_width,
        classifier_depth=classifier_depth,
        classifier_attention_resolutions=classifier_attention_resolutions,
        classifier_use_scale_shift_norm=classifier_use_scale_shift_norm,
        classifier_resblock_updown=classifier_resblock_updown,
        classifier_pool=classifier_pool,
        use_property_aware=use_property_aware,
        stress_strain_dim=stress_strain_dim
    )

    diffusion = create_mpa_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        use_property_aware=use_property_aware,
        physics_consistency=physics_consistency,
        multi_modal_conditioning=multi_modal_conditioning,
        stress_strain_dim=stress_strain_dim
    )

    return classifier, diffusion


def create_mpa_classifier(
        image_size: int,
        classifier_use_fp16: bool,
        classifier_width: int,
        classifier_depth: int,
        classifier_attention_resolutions: str,
        classifier_use_scale_shift_norm: bool,
        classifier_resblock_updown: bool,
        classifier_pool: str,
        use_property_aware: bool = True,
        stress_strain_dim: int = DEFAULT_STRESS_STRAIN_DIM
) -> nn.Module:
    """
    Create MPaDiffusion classifier with property-aware capabilities.
    """
    # Determine channel multiplier
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 4)
    else:
        raise ValueError(f"Unsupported image size: {image_size}")

    # Calculate attention resolutions
    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return MPaEncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=2,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
        use_property_aware=use_property_aware,
        stress_strain_dim=stress_strain_dim
    )


def create_mpa_sr_model_and_diffusion(
        large_size: int,
        small_size: int,
        class_cond: bool,
        learn_sigma: bool,
        num_channels: int,
        num_res_blocks: int,
        num_heads: int,
        num_head_channels: int,
        num_heads_upsample: int,
        attention_resolutions: str,
        dropout: float,
        diffusion_steps: int,
        noise_schedule: str,
        timestep_respacing: str,
        use_kl: bool,
        predict_xstart: bool,
        rescale_timesteps: bool,
        rescale_learned_sigmas: bool,
        use_checkpoint: bool,
        use_scale_shift_norm: bool,
        resblock_updown: bool,
        use_fp16: bool,
        # MPaDiffusion parameters
        use_property_aware: bool = True,
        multi_modal_conditioning: bool = True,
        stress_strain_dim: int = DEFAULT_STRESS_STRAIN_DIM
) -> Tuple[nn.Module, object]:
    """
    Create MPaDiffusion super-resolution model and diffusion.
    """
    model = create_mpa_sr_model(
        large_size=large_size,
        small_size=small_size,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_property_aware=use_property_aware,
        multi_modal_conditioning=multi_modal_conditioning,
        stress_strain_dim=stress_strain_dim
    )

    diffusion = create_mpa_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        use_property_aware=use_property_aware,
        multi_modal_conditioning=multi_modal_conditioning,
        stress_strain_dim=stress_strain_dim
    )

    return model, diffusion


def create_mpa_sr_model(
        large_size: int,
        small_size: int,
        num_channels: int,
        num_res_blocks: int,
        learn_sigma: bool,
        class_cond: bool,
        use_checkpoint: bool,
        attention_resolutions: str,
        num_heads: int,
        num_head_channels: int,
        num_heads_upsample: int,
        use_scale_shift_norm: bool,
        dropout: float,
        resblock_updown: bool,
        use_fp16: bool,
        use_property_aware: bool = True,
        multi_modal_conditioning: bool = True,
        stress_strain_dim: int = DEFAULT_STRESS_STRAIN_DIM
) -> nn.Module:
    """
    Create MPaDiffusion super-resolution model.
    """
    # Determine channel multiplier
    if large_size == 512:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif large_size == 32:
        channel_mult = (1, 2, 4)
    else:
        raise ValueError(f"Unsupported large size: {large_size}")

    # Calculate attention resolutions
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return MPaSuperResModel(
        image_size=large_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(MPA_NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_property_aware=use_property_aware,
        multi_modal_conditioning=multi_modal_conditioning,
        stress_strain_dim=stress_strain_dim
    )


# Utility functions
def add_dict_to_argparser(parser: argparse.ArgumentParser, default_dict: Dict):
    """Add dictionary of arguments to argument parser."""
    for key, value in default_dict.items():
        value_type = type(value)
        if value is None:
            value_type = str
        elif isinstance(value, bool):
            value_type = str2bool
        parser.add_argument(f"--{key}", default=value, type=value_type)


def args_to_dict(args, keys):
    """Convert arguments to dictionary."""
    return {key: getattr(args, key) for key in keys}


def str2bool(value):
    """
    Convert string to boolean with enhanced error handling and support for multiple input types.

    This function robustly converts various string representations to boolean values,
    supporting common true/false patterns with comprehensive error handling.

    Args:
        value: Input value to convert. Can be bool, str, int, or float.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If value cannot be converted to boolean.

    Examples:
        >>> str2bool("yes")
        True
        >>> str2bool("false")
        False
        >>> str2bool(1)
        True
        >>> str2bool(0)
        False
    """
    # Handle boolean input directly
    if isinstance(value, bool):
        return value

    # Handle numeric input
    if isinstance(value, (int, float)):
        if value == 0:
            return False
        elif value == 1:
            return True
        else:
            raise argparse.ArgumentTypeError(
                f"Boolean value expected for numeric input, got {value}. "
                "Only 0 and 1 are accepted."
            )

    # Handle string input with case-insensitive comparison
    if isinstance(value, str):
        value_lower = value.lower().strip()

        # Extended true values
        true_values = {
            "yes", "true", "t", "y", "1", "on", "enable", "enabled",
            "positive", "affirmative", "ok", "okay", "si", "da"
        }

        # Extended false values
        false_values = {
            "no", "false", "f", "n", "0", "off", "disable", "disabled",
            "negative", "negative", "not", "none", "null", "nein", "nyet"
        }

        if value_lower in true_values:
            return True
        elif value_lower in false_values:
            return False
        else:
            # Provide helpful error message with accepted values
            raise argparse.ArgumentTypeError(
                f"Invalid boolean value: '{value}'. "
                f"Accepted true values: {sorted(true_values)}. "
                f"Accepted false values: {sorted(false_values)}."
            )

    # Handle unsupported types
    raise argparse.ArgumentTypeError(
        f"Unsupported type for boolean conversion: {type(value)}. "
        "Supported types: bool, str, int, float."
    )


# Enhanced boolean utilities for MPaDiffusion framework
class BooleanConverter:
    """
    Advanced boolean conversion utility with configuration support.
    Provides enhanced functionality for MPaDiffusion parameter handling.
    """

    def __init__(self, strict_mode: bool = True, case_sensitive: bool = False):
        """
        Initialize boolean converter with configuration options.

        Args:
            strict_mode: If True, raises exceptions on invalid input.
            case_sensitive: If True, performs case-sensitive comparisons.
        """
        self.strict_mode = strict_mode
        self.case_sensitive = case_sensitive

        # Configurable true/false values
        self.true_values = {
            "yes", "true", "t", "y", "1", "on", "enable", "enabled"
        }
        self.false_values = {
            "no", "false", "f", "n", "0", "off", "disable", "disabled"
        }

    def convert(self, value, default=None):
        """
        Convert value to boolean with optional default.

        Args:
            value: Input value to convert.
            default: Default value to return if conversion fails and not in strict mode.

        Returns:
            bool: Converted value or default.
        """
        try:
            return self._convert_strict(value)
        except (ValueError, argparse.ArgumentTypeError):
            if not self.strict_mode and default is not None:
                return default
            raise

    def _convert_strict(self, value):
        """Strict conversion with comprehensive type handling."""
        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            if value == 0:
                return False
            elif value == 1:
                return True
            else:
                raise ValueError(f"Invalid numeric value for boolean: {value}")

        if isinstance(value, str):
            processed_value = value if self.case_sensitive else value.lower()
            processed_value = processed_value.strip()

            if processed_value in self.true_values:
                return True
            elif processed_value in self.false_values:
                return False
            else:
                raise ValueError(f"Unrecognized boolean string: {value}")

        raise TypeError(f"Unsupported type for boolean conversion: {type(value)}")

    def add_true_value(self, value: str):
        """Add custom true value to accepted patterns."""
        if self.case_sensitive:
            self.true_values.add(value)
        else:
            self.true_values.add(value.lower())

    def add_false_value(self, value: str):
        """Add custom false value to accepted patterns."""
        if self.case_sensitive:
            self.false_values.add(value)
        else:
            self.false_values.add(value.lower())


# MPaDiffusion specific boolean configuration
class MPaBooleanConfig:
    """
    Boolean configuration manager for MPaDiffusion framework.
    Handles specialized boolean parameters for multi-modal diffusion models.
    """

    def __init__(self):
        self.converter = BooleanConverter(strict_mode=True, case_sensitive=False)

        # Add MPaDiffusion specific boolean values
        self._add_mpa_specific_values()

    def _add_mpa_specific_values(self):
        """Add MPaDiffusion-specific boolean patterns."""
        # Physics and property-aware terms
        self.converter.add_true_value("physics")
        self.converter.add_true_value("property")
        self.converter.add_true_value("multi_modal")
        self.converter.add_true_value("anisotropy")
        self.converter.add_true_value("stress_strain")

        self.converter.add_false_value("no_physics")
        self.converter.add_false_value("no_property")
        self.converter.add_false_value("single_modal")
        self.converter.add_false_value("isotropy")
        self.converter.add_false_value("no_stress_strain")

    def parse_mpa_boolean(self, value, parameter_name=None):
        """
        Parse boolean value with MPaDiffusion context.

        Args:
            value: Value to parse.
            parameter_name: Optional parameter name for error messages.

        Returns:
            bool: Parsed boolean value.
        """
        try:
            return self.converter.convert(value)
        except (ValueError, TypeError) as e:
            if parameter_name:
                raise argparse.ArgumentTypeError(
                    f"Invalid value for {parameter_name}: {value}. {str(e)}"
                )
            else:
                raise argparse.ArgumentTypeError(str(e))


# Utility functions for MPaDiffusion boolean handling
def create_mpa_boolean_parser():
    """
    Create a configured boolean parser for MPaDiffusion command-line arguments.
    """
    return MPaBooleanConfig()


def validate_mpa_boolean_parameters(config_dict: dict) -> dict:
    """
    Validate and convert boolean parameters in MPaDiffusion configuration.

    Args:
        config_dict: Dictionary containing configuration parameters.

    Returns:
        dict: Dictionary with converted boolean values.
    """
    mpa_parser = MPaBooleanConfig()
    validated_config = config_dict.copy()

    # Common boolean parameters in MPaDiffusion
    boolean_parameters = [
        'use_mpa', 'property_aware_var', 'physics_consistency',
        'multi_modal_conditioning', 'anisotropy_regularization',
        'classifier_free_guidance', 'use_fp16', 'class_cond',
        'learn_sigma', 'use_kl', 'predict_xstart', 'rescale_timesteps',
        'use_checkpoint', 'use_scale_shift_norm', 'resblock_updown'
    ]

    for param in boolean_parameters:
        if param in validated_config:
            try:
                validated_config[param] = mpa_parser.parse_mpa_boolean(
                    validated_config[param], param
                )
            except argparse.ArgumentTypeError:
                # Keep original value if conversion fails and parameter is optional
                if param not in ['use_mpa', 'property_aware_var']:  # Required params
                    validated_config[param] = False  # Default to False for optional params

    return validated_config


def str2bool_with_default(value, default=False):
    """
    Convert string to boolean with default value fallback.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.

    Returns:
        bool: Converted value or default.
    """
    try:
        return str2bool(value)
    except argparse.ArgumentTypeError:
        return default


# Example usage and testing
def test_boolean_conversion():
    """
    Test function for boolean conversion utilities.
    """
    test_cases = [
        # (input, expected_output, description)
        ("yes", True, "Basic true value"),
        ("no", False, "Basic false value"),
        ("TRUE", True, "Case insensitive true"),
        ("0", False, "Numeric false"),
        ("1", True, "Numeric true"),
        (True, True, "Boolean true"),
        (False, False, "Boolean false"),
    ]

    print("Testing boolean conversion functions:")
    print("-" * 50)

    for input_val, expected, description in test_cases:
        try:
            result = str2bool(input_val)
            status = "PASS" if result == expected else "FAIL"
            print(f"{status}: {description} - '{input_val}' -> {result}")
        except argparse.ArgumentTypeError as e:
            print(f"FAIL: {description} - '{input_val}' -> Error: {e}")

    # Test MPaDiffusion specific parser
    print("\nTesting MPaDiffusion boolean parser:")
    print("-" * 50)

    mpa_parser = MPaBooleanConfig()
    mpa_test_cases = [
        ("physics", True, "MPaDiffusion physics term"),
        ("no_physics", False, "MPaDiffusion negative physics term"),
    ]

    for input_val, expected, description in mpa_test_cases:
        try:
            result = mpa_parser.parse_mpa_boolean(input_val)
            status = "PASS" if result == expected else "FAIL"
            print(f"{status}: {description} - '{input_val}' -> {result}")
        except argparse.ArgumentTypeError as e:
            print(f"FAIL: {description} - '{input_val}' -> Error: {e}")


# Main execution block
if __name__ == "__main__":
    # Run tests if script is executed directly
    test_boolean_conversion()

    # Example usage in MPaDiffusion context
    print("\nMPaDiffusion configuration example:")
    print("-" * 50)

    sample_config = {
        'use_mpa': 'yes',
        'property_aware_var': 'true',
        'physics_consistency': 'physics',
        'multi_modal_conditioning': 'enabled',
        'anisotropy_regularization': 'on',
        'classifier_free_guidance': '1'
    }

    validated = validate_mpa_boolean_parameters(sample_config)
    for key, value in validated.items():
        print(f"{key}: {value} ({type(value).__name__})")


