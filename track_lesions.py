import torch
import numpy as np
import nibabel as nib
from difference_weighting.architectures.longi_unet_difference_weighting import LongiUNetDiffWeighting

def load_model(checkpoint_path, input_channels=1, num_classes=2):
    # Initialize the model structure
    # NOTE: Ensure these parameters match your training configuration
    model = LongiUNetDiffWeighting(
        input_channels=input_channels,
        num_classes=num_classes,
        backbone_class_name="dynamic_network_architectures.architectures.unet.PlainConvUNet",
        kernel_sizes=[[3, 3, 3]] * 6,      # Example for standard 3D U-Net
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        features_per_stage=[32, 64, 128, 256, 320, 320],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        conv_op=torch.nn.Conv3d,
        norm_op=torch.nn.InstanceNorm3d,
        dropout_op=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={'negative_slope': 0.01, 'inplace': True},
        deep_supervision=False
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['network_weights'])
    model.eval()
    return model

def track_lesions(model, current_scan_path, prior_scan_path, diff_threshold=0.5):
    # 1. Load Data
    # Assuming data is preprocessed (registered, skull-stripped, normalized)
    curr_img = nib.load(current_scan_path).get_fdata()
    prior_img = nib.load(prior_scan_path).get_fdata()
    
    # Add batch and channel dimensions (Batch, Channel, D, H, W)
    curr_tensor = torch.from_numpy(curr_img).float().unsqueeze(0).unsqueeze(0)
    prior_tensor = torch.from_numpy(prior_img).float().unsqueeze(0).unsqueeze(0)

    # 2. Forward Pass
    with torch.no_grad():
        # Returns tuple: (logits, [diff_map_0, diff_map_1, ...])
        logits, diff_maps = model(curr_tensor, prior_tensor)
        
        # Get binary segmentation (0 = Background, 1 = Lesion)
        segmentation = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    
    # 3. Analyze Difference Maps for Tracking
    # We use the deepest/lowest resolution map for semantic context, 
    # or the highest resolution map (last in list) for fine detail.
    # Let's use the last one (highest resolution) and upsample it to match input size.
    raw_diff_map = diff_maps[-1] 
    
    # Upsample difference map to match original image size
    upsampled_diff = torch.nn.functional.interpolate(
        raw_diff_map, 
        size=curr_tensor.shape[2:], 
        mode='trilinear', 
        align_corners=False
    )
    
    # Normalize the difference map to 0-1 range for thresholding
    # (Since it comes from InstanceNorm, it might be unbounded)
    diff_norm = torch.sigmoid(upsampled_diff) 
    
    # 4. Classify Lesions
    # High difference activation indicates "Change" -> New/Active
    high_change_mask = (diff_norm > diff_threshold).squeeze()
    
    # Logic:
    # - New/Active: Is a Lesion AND has High Difference
    # - Stable: Is a Lesion AND has Low Difference
    new_lesion_mask = segmentation & high_change_mask.long()
    stable_lesion_mask = segmentation & (~high_change_mask).long()
    
    return new_lesion_mask, stable_lesion_mask

# --- Usage ---
if __name__ == "__main__":
    model_path = "path/to/your/checkpoint.pth"
    scan_t2 = "data/patient_01_time02.nii.gz"
    scan_t1 = "data/patient_01_time01.nii.gz" # Baseline (Prior)
    
    print("Loading model...")
    model = load_model(model_path)
    
    print("Tracking lesions...")
    new_mask, stable_mask = track_lesions(model, scan_t2, scan_t1)
    
    print(f"New Lesions Voxels: {torch.sum(new_mask)}")
    print(f"Stable Lesions Voxels: {torch.sum(stable_mask)}")
    
    # Save results
    # (Using the affine from the original image for proper saving)
    original_nii = nib.load(scan_t2)
    nib.save(nib.Nifti1Image(new_mask.numpy().astype(np.uint8), original_nii.affine), "output_new_lesions.nii.gz")
    nib.save(nib.Nifti1Image(stable_mask.numpy().astype(np.uint8), original_nii.affine), "output_stable_lesions.nii.gz")
    print("Saved outputs.")
