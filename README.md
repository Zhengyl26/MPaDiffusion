

This repository presents the PyTorch implementation of the research work outlined in the paper [A unified framework of multi-modal property-aware diffusion model for 3D reconstruction and on-demand design]. Unlike conventional diffusion-based approaches that focus solely on image-level tasks, this project introduces a novel unified framework that integrates multi-modal data fusion, property-aware guidance, and flexible 3D structure generation, enabling end-to-end 3D reconstruction and customizable design based on user-specified properties. 

Core Contributions & Technical Highlights
The implementation encapsulates the key innovations of the proposed framework:

Multi-modal Data Integration: Seamlessly processes heterogeneous input modalities (e.g., 2D images, point clouds, and physical property measurements) through a cross-modal attention mechanism, enabling the model to leverage complementary information for robust 3D representation learning.

On-Demand Design Flexibility: Supports conditional generation of 3D models based on user-defined property parameters (e.g., adjusting porosity for porous materials or specifying load-bearing capacities for mechanical parts), bridging the gap between generic 3D reconstruction and task-specific design requirements.

Efficient 3D Structure Modeling: Adopts a hybrid UNet architecture with 3D-aware residual blocks and adaptive resolution scaling, balancing reconstruction accuracy and computational efficiency for large-scale 3D volumes.



## Usage

Configuration Flags
Define core parameters for model architecture, diffusion process, and training setup:
```
# Model configuration: adapted for 3D multi-modal input and property encoding
MODEL_FLAGS="--vol_size 64 --num_channels 64 --num_res_blocks 3 --num_heads 4 --class_cond True --property_cond True --learn_sigma True --use_scale_shift_norm True --attention_resolutions 32,16 --cross_attention_dim 256"

# Diffusion setup: property-guided noise scheduling
DIFFUSION_FLAGS="--diffusion_steps 2000 --noise_schedule cosine --rescale_learned_sigmas True --rescale_timesteps True --property_guidance_scale 1.5"

# Training parameters: multi-modal data loading and property-aware loss
TRAIN_FLAGS="--lr 5e-5 --batch_size 4 --property_loss_weight 0.3 --multi_modal_weight 0.2 --max_train_steps 1000000"
```
To train the multi-modal property-aware diffusion model, run:

```
python3 scripts/train.py \
  

3D Reconstruction & On-Demand Generation
To generate 3D structures from multi-modal inputs (e.g., a 2D image + property constraints) or sample custom designs, run:

```
  
  python3 sample.py \
  
  
```


####
