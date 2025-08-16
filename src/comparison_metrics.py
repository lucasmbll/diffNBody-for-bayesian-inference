import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def fourier_downsample_3d(field, target_shape):
    """
    Downsample a 3D field using Fourier low-pass filtering followed by decimation.
    This preserves the large-scale structure better than interpolation.
    
    Parameters:
    -----------
    field : array
        Input field with shape (nx, ny, nz, n_components) or (nx, ny, nz)
    target_shape : tuple
        Target spatial dimensions (target_nx, target_ny, target_nz)
    
    Returns:
    --------
    downsampled_field : array
        Downsampled field with target spatial dimensions
    """
    original_shape = field.shape[:3]
    has_components = len(field.shape) > 3
    n_components = field.shape[-1] if has_components else 1
    
    if not has_components:
        field = field[..., None]  # Add component dimension
    
    # Process each component separately
    downsampled_components = []
    
    for i in range(n_components):
        # Take FFT of the field
        fft_field = jnp.fft.fftn(field[..., i])
        
        # Create frequency coordinates for cropping
        nx, ny, nz = original_shape
        target_nx, target_ny, target_nz = target_shape
        
        # Calculate the cropping indices (keep low frequencies)
        # FFT layout: [0, 1, ..., N/2, -N/2+1, ..., -1]
        def get_crop_indices(n_orig, n_target):
            if n_target >= n_orig:
                return slice(None)  # No cropping needed
            
            # Keep frequencies from 0 to n_target//2 and from -(n_target//2) to -1
            half_target = n_target // 2
            indices = list(range(half_target + 1)) + list(range(n_orig - (n_target - half_target - 1), n_orig))
            return indices
        
        # Get cropping indices for each dimension
        x_indices = get_crop_indices(nx, target_nx)
        y_indices = get_crop_indices(ny, target_ny)
        z_indices = get_crop_indices(nz, target_nz)
        
        # Crop the FFT (keep only low frequencies)
        if isinstance(x_indices, list):
            fft_cropped = fft_field[jnp.ix_(jnp.array(x_indices), jnp.array(y_indices), jnp.array(z_indices))]
        else:
            fft_cropped = fft_field[x_indices, y_indices, z_indices]
        
        # Adjust normalization to conserve total power
        scale_factor = np.prod(target_shape) / np.prod(original_shape)
        fft_cropped = fft_cropped * scale_factor
        
        # Take inverse FFT to get downsampled field
        downsampled = jnp.fft.ifftn(fft_cropped)
        
        # Take real part (should be real for real input fields)
        downsampled = jnp.real(downsampled)
        
        downsampled_components.append(downsampled)
    
    # Stack components back together
    result = jnp.stack(downsampled_components, axis=-1)
    
    if not has_components:
        result = result[..., 0]  # Remove added dimension
    
    return result

def resize_field_3d(field, target_shape, method='fourier'):
    """
    Resize a 3D field to target shape using the specified method.
    
    Parameters:
    -----------
    field : array
        Input field with shape (..., field_components)
    target_shape : tuple
        Target spatial dimensions (nx, ny, nz)
    method : str
        Resizing method ('fourier' for low-pass + decimation, 'trilinear' for interpolation)
    
    Returns:
    --------
    resized_field : array
        Resized field with target shape
    """
    original_shape = field.shape[:3]
    
    if original_shape == target_shape:
        return field
    
    if method == 'fourier':
        # Use Fourier downsampling for better preservation of large-scale structure
        return fourier_downsample_3d(field, target_shape)
    else:
        # Fall back to trilinear interpolation for upsampling or when requested
        field_components = field.shape[-1] if len(field.shape) > 3 else 1
        
        if field_components == 1:
            field = field[..., None]  # Add component dimension
        
        # Create coordinate grids
        x_orig = np.linspace(0, 1, original_shape[0])
        y_orig = np.linspace(0, 1, original_shape[1]) 
        z_orig = np.linspace(0, 1, original_shape[2])
        
        x_new = np.linspace(0, 1, target_shape[0])
        y_new = np.linspace(0, 1, target_shape[1])
        z_new = np.linspace(0, 1, target_shape[2])
        
        # Interpolate each field component
        resized_components = []
        for i in range(field.shape[-1]):
            interpolator = RegularGridInterpolator(
                (x_orig, y_orig, z_orig), 
                np.array(field[..., i]),
                method='linear',
                bounds_error=False,
                fill_value=0
            )
            
            # Create target coordinate grid
            X, Y, Z = np.meshgrid(x_new, y_new, z_new, indexing='ij')
            points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
            
            # Interpolate
            interpolated = interpolator(points).reshape(target_shape)
            resized_components.append(interpolated)
        
        resized_field = np.stack(resized_components, axis=-1)
        
        if field_components == 1:
            resized_field = resized_field[..., 0]  # Remove added dimension
        
        return jnp.array(resized_field)

@jax.jit
def compute_field_statistics(field):
    """Compute basic statistical moments of a field."""
    field_flat = field.flatten()
    mean = jnp.mean(field_flat)
    std = jnp.std(field_flat)
    skewness = jnp.mean(((field_flat - mean) / std) ** 3)
    kurtosis = jnp.mean(((field_flat - mean) / std) ** 4) - 3
    
    return {
        'mean': mean,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'min': jnp.min(field_flat),
        'max': jnp.max(field_flat)
    }

@jax.jit
def mse_metric(field1, field2):
    """Mean Squared Error between two fields."""
    return jnp.mean((field1 - field2) ** 2)

@jax.jit
def mae_metric(field1, field2):
    """Mean Absolute Error between two fields."""
    return jnp.mean(jnp.abs(field1 - field2))

@jax.jit
def normalized_rmse(field1, field2):
    """
    Normalized Root Mean Square Error with different normalization options.
    
    Parameters:
    -----------
    normalization : str
        'range' - normalize by range of field2 (less stable)
        'std' - normalize by standard deviation of field2 (more stable)
    """
    rmse = jnp.sqrt(jnp.mean((field1 - field2) ** 2))
    
    
    norm_factor = jnp.std(field2) + 1e-10
    
    return rmse / norm_factor

@jax.jit
def correlation_coefficient(field1, field2):
    """Pearson correlation coefficient between two fields."""
    field1_flat = field1.flatten()
    field2_flat = field2.flatten()
    
    mean1 = jnp.mean(field1_flat)
    mean2 = jnp.mean(field2_flat)
    
    numerator = jnp.sum((field1_flat - mean1) * (field2_flat - mean2))
    denominator = jnp.sqrt(jnp.sum((field1_flat - mean1) ** 2) * jnp.sum((field2_flat - mean2) ** 2))
    
    return numerator / (denominator + 1e-10)

@jax.jit
def structural_similarity_index(field1, field2, dynamic_range=None):
    """
    Simplified Structural Similarity Index (SSIM) for 3D fields.
    Based on luminance, contrast, and structure comparison.
    """
    if dynamic_range is None:
        dynamic_range = jnp.max(jnp.maximum(field1, field2)) - jnp.min(jnp.minimum(field1, field2))
    
    # Constants for stability
    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2
    
    # Local means (using simple global means for 3D fields)
    mu1 = jnp.mean(field1)
    mu2 = jnp.mean(field2)
    
    # Local variances and covariance
    sigma1_sq = jnp.var(field1)
    sigma2_sq = jnp.var(field2)
    sigma12 = jnp.mean((field1 - mu1) * (field2 - mu2))
    
    # SSIM calculation
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    
    return numerator / (denominator + 1e-10)

def compute_cross_spectrum_and_coherence(fft1, fft2, k_bins):
    """
    Compute cross-spectrum and coherence between two FFT fields.
    
    Parameters:
    -----------
    fft1, fft2 : complex arrays
        FFT of the two fields with shape (nx, ny, nz)
    k_bins : array
        K-bins for spherical averaging
    
    Returns:
    --------
    k_centers : array
        Centers of k-bins
    cross_spectrum : array
        Cross-power spectrum P_cross(k) = <F1(k) * conj(F2(k))>
    coherence : array
        Coherence r(k) = P_cross(k) / sqrt(P1(k) * P2(k))
    power1, power2 : arrays
        Individual power spectra
    """
    # Get k-space coordinates
    nx, ny, nz = fft1.shape
    kx = jnp.fft.fftfreq(nx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(ny) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(nz) * 2 * jnp.pi
    
    # Create 3D k-grid
    kx_grid, ky_grid, kz_grid = jnp.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = jnp.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
    
    # Compute power and cross-power in 3D
    power1_3d = jnp.abs(fft1)**2
    power2_3d = jnp.abs(fft2)**2
    cross_power_3d = jnp.real(fft1 * jnp.conj(fft2))  # Real part of cross-spectrum
    
    # Spherically average
    n_bins = len(k_bins) - 1
    k_centers = jnp.zeros(n_bins)
    power1_binned = jnp.zeros(n_bins)
    power2_binned = jnp.zeros(n_bins)
    cross_power_binned = jnp.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1]) & (k_mag > 0)
        if jnp.sum(mask) > 0:
            k_centers = k_centers.at[i].set(jnp.mean(k_mag[mask]))
            power1_binned = power1_binned.at[i].set(jnp.mean(power1_3d[mask]))
            power2_binned = power2_binned.at[i].set(jnp.mean(power2_3d[mask]))
            cross_power_binned = cross_power_binned.at[i].set(jnp.mean(cross_power_3d[mask]))
        else:
            k_centers = k_centers.at[i].set((k_bins[i] + k_bins[i + 1]) / 2)
    
    # Compute coherence: r(k) = P_cross(k) / sqrt(P1(k) * P2(k))
    coherence = cross_power_binned / (jnp.sqrt(power1_binned * power2_binned) + 1e-10)
    
    return k_centers, cross_power_binned, coherence, power1_binned, power2_binned

def power_spectrum_comparison_from_existing(k_centers1, power_spec1, k_centers2, power_spec2, 
                                          fft1=None, fft2=None, field_names=None):
    """
    Compare power spectra using already computed FFT and power spectrum data.
    
    Parameters:
    -----------
    k_centers1, k_centers2 : arrays
        K-centers for the two datasets with shape (..., n_components)
    power_spec1, power_spec2 : arrays  
        Power spectra for the two datasets with shape (..., n_components)
    fft1, fft2 : arrays, optional
        FFT data for cross-spectrum analysis with shape (..., n_components)
    field_names : list, optional
        Names of field components
    
    Returns:
    --------
    dict : Contains power spectrum metrics including coherence and agreement scale
    """
    if field_names is None:
        field_names = ['density', 'vx', 'vy', 'vz']
    
    # Determine common k-range for interpolation
    k_min = max(jnp.min(k_centers1[k_centers1 > 0]), jnp.min(k_centers2[k_centers2 > 0]))
    k_max = min(jnp.max(k_centers1), jnp.max(k_centers2))
    
    # Create common k-grid (logarithmic spacing)
    k_common = jnp.logspace(jnp.log10(k_min), jnp.log10(k_max), 30)
    k_bins = jnp.logspace(jnp.log10(k_min * 0.9), jnp.log10(k_max * 1.1), 31)
    
    metrics = {}
    
    n_components = min(k_centers1.shape[-1], k_centers2.shape[-1])
    
    for i in range(n_components):
        if i >= len(field_names):
            field_name = f'component_{i}'
        else:
            field_name = field_names[i]
            
        # Extract power spectra and k-centers for this field
        k1_field = k_centers1[..., i]
        k2_field = k_centers2[..., i]
        ps1_field = power_spec1[..., i]
        ps2_field = power_spec2[..., i]
        
        # Interpolate in log-log space for better power-law behavior
        valid1 = (k1_field > 0) & (ps1_field > 0) & jnp.isfinite(ps1_field)
        valid2 = (k2_field > 0) & (ps2_field > 0) & jnp.isfinite(ps2_field)
        
        if jnp.sum(valid1) > 5 and jnp.sum(valid2) > 5:
            # Log-space interpolation
            log_k1 = jnp.log10(k1_field[valid1])
            log_ps1 = jnp.log10(ps1_field[valid1])
            log_k2 = jnp.log10(k2_field[valid2])
            log_ps2 = jnp.log10(ps2_field[valid2])
            
            log_k_common = jnp.log10(k_common)
            log_ps1_interp = jnp.interp(log_k_common, log_k1, log_ps1)
            log_ps2_interp = jnp.interp(log_k_common, log_k2, log_ps2)
            
            # Convert back to linear space
            ps1_interp = 10**log_ps1_interp
            ps2_interp = 10**log_ps2_interp
            
            # Power spectrum ratio analysis
            ratio = ps1_interp / ps2_interp
            metrics[f'{field_name}_ps_ratio_mean'] = jnp.mean(ratio)
            metrics[f'{field_name}_ps_ratio_std'] = jnp.std(ratio)
            metrics[f'{field_name}_ps_correlation'] = correlation_coefficient(
                log_ps1_interp, log_ps2_interp
            )
            
            # Relative difference in power
            rel_diff = jnp.abs(ps1_interp - ps2_interp) / (ps2_interp + 1e-10)
            metrics[f'{field_name}_ps_rel_error_mean'] = jnp.mean(rel_diff)
            metrics[f'{field_name}_ps_rel_error_max'] = jnp.max(rel_diff)
            
            # Compute cross-spectrum and coherence if FFT data is available
            if fft1 is not None and fft2 is not None:
                try:
                    fft1_field = fft1[..., i]
                    fft2_field = fft2[..., i]
                    
                    k_centers, cross_spec, coherence, power1_binned, power2_binned = compute_cross_spectrum_and_coherence(
                        fft1_field, fft2_field, k_bins
                    )
                    
                    # Filter valid coherence values
                    valid_coherence = jnp.isfinite(coherence) & (k_centers > 0)
                    
                    if jnp.sum(valid_coherence) > 0:
                        coherence_valid = coherence[valid_coherence]
                        k_valid = k_centers[valid_coherence]
                        
                        # Mean coherence over large scales (lowest 1/3 of k-range)
                        large_scale_mask = k_valid <= jnp.percentile(k_valid, 33)
                        if jnp.sum(large_scale_mask) > 0:
                            metrics[f'{field_name}_coherence_large_scale'] = jnp.mean(coherence_valid[large_scale_mask])
                        else:
                            metrics[f'{field_name}_coherence_large_scale'] = jnp.nan
                        
                        # Agreement scale: largest k where r(k) >= 0.9
                        high_coherence_mask = coherence_valid >= 0.9
                        if jnp.sum(high_coherence_mask) > 0:
                            agreement_scale = jnp.max(k_valid[high_coherence_mask])
                        else:
                            agreement_scale = jnp.min(k_valid)  # No agreement at any scale
                        
                        metrics[f'{field_name}_agreement_scale'] = agreement_scale
                        metrics[f'{field_name}_coherence_mean'] = jnp.mean(coherence_valid)
                        
                        # Store k and coherence arrays for plotting
                        metrics[f'{field_name}_k_coherence'] = k_valid
                        metrics[f'{field_name}_coherence_values'] = coherence_valid
                    else:
                        # No valid coherence data
                        for metric_suffix in ['coherence_large_scale', 'agreement_scale', 'coherence_mean']:
                            metrics[f'{field_name}_{metric_suffix}'] = jnp.nan
                        
                except Exception as e:
                    print(f"Warning: Coherence computation failed for {field_name}: {e}")
                    for metric_suffix in ['coherence_large_scale', 'agreement_scale', 'coherence_mean']:
                        metrics[f'{field_name}_{metric_suffix}'] = jnp.nan
            else:
                # No FFT data available for coherence analysis
                for metric_suffix in ['coherence_large_scale', 'agreement_scale', 'coherence_mean']:
                    metrics[f'{field_name}_{metric_suffix}'] = jnp.nan
        else:
            # Fill with NaN if not enough valid points
            for metric_suffix in ['ps_ratio_mean', 'ps_ratio_std', 'ps_correlation', 
                                'ps_rel_error_mean', 'ps_rel_error_max',
                                'coherence_large_scale', 'agreement_scale', 'coherence_mean']:
                metrics[f'{field_name}_{metric_suffix}'] = jnp.nan
    
    return metrics

def load_fft_data(fft_file_path):
    """
    Load and reconstruct complex FFT data from saved NPZ file.
    
    Parameters:
    -----------
    fft_file_path : str
        Path to the saved FFT NPZ file
        
    Returns:
    --------
    dict : Dictionary containing reconstructed FFT data
    """
    data = np.load(fft_file_path)
    
    # Reconstruct complex FFT arrays
    fft_initial = data['fft_initial_real'] + 1j * data['fft_initial_imag']
    fft_final = data['fft_final_real'] + 1j * data['fft_final_imag']
    
    return {
        'fft_initial': fft_initial,
        'fft_final': fft_final,
        'field_names': data['field_names'],
        'box_size': data['box_size'],
        'kpc_per_pixel': data['kpc_per_pixel'],
        'grid_shape': data['grid_shape']
    }

    
def compute_all_comparison_metrics_from_files(current_fields_path, current_fft_path, current_ps_path,
                                            reference_fields_path, reference_fft_path, reference_ps_path,
                                            field_names=None, resize_method='fourier'):
    """
    Compute comprehensive comparison metrics using saved data files.
    
    Parameters:
    -----------
    current_fields_path, reference_fields_path : str
        Paths to final_fields.npz files
    current_fft_path, reference_fft_path : str  
        Paths to fft_data.npz files
    current_ps_path, reference_ps_path : str
        Paths to power_spectrum_data.npz files
    field_names : list, optional
        Names of field components
    resize_method : str
        Method for resizing fields ('fourier' or 'trilinear')
    
    Returns:
    --------
    dict : Comprehensive comparison metrics
    """
    if field_names is None:
        field_names = ['density', 'vx', 'vy', 'vz']
    
    print("Loading current simulation data...")
    # Load current data
    current_fields_data = np.load(current_fields_path)
    current_fft_data = load_fft_data(current_fft_path)
    current_ps_data = np.load(current_ps_path)
    
    print("Loading reference simulation data...")
    # Load reference data
    ref_fields_data = np.load(reference_fields_path)
    ref_fft_data = load_fft_data(reference_fft_path)
    ref_ps_data = np.load(reference_ps_path)
    
    # Extract fields
    current_field = current_fields_data['final_field']
    ref_field = ref_fields_data['final_field']
    current_box_size = current_fields_data['box_size']
    ref_box_size = ref_fields_data['box_size']
    
    # Extract FFT data
    current_fft_final = current_fft_data['fft_final']
    ref_fft_final = ref_fft_data['fft_final']
    
    # Extract power spectrum data
    current_k_centers = current_ps_data['k_center_final']
    current_power_spec = current_ps_data['power_spectrum_final']
    ref_k_centers = ref_ps_data['k_center_final']
    ref_power_spec = ref_ps_data['power_spectrum_final']
    
    print(f"Current field shape: {current_field.shape}")
    print(f"Reference field shape: {ref_field.shape}")
    
    print(f"Current field shape: {current_field.shape}")
    print(f"Reference field shape: {ref_field.shape}")
    
    # Resize fields if needed for real-space comparison (always downsample the higher-res to the lower-res)
    if current_field.shape[:3] != ref_field.shape[:3]:
        cur_sz = np.array(current_field.shape[:3])
        ref_sz = np.array(ref_field.shape[:3])
        cur_vox = int(np.prod(cur_sz))
        ref_vox = int(np.prod(ref_sz))
        if ref_vox > cur_vox:
            print(f"Resizing reference field from {tuple(ref_sz)} to {tuple(cur_sz)} using {resize_method} method")
            ref_field_resized = resize_field_3d(ref_field, tuple(cur_sz), method=resize_method)
            current_field_resized = current_field
        else:
            print(f"Resizing current field from {tuple(cur_sz)} to {tuple(ref_sz)} using {resize_method} method")
            current_field_resized = resize_field_3d(current_field, tuple(ref_sz), method=resize_method)
            ref_field_resized = ref_field
    else:
        current_field_resized = current_field
        ref_field_resized = ref_field
    
    metrics = {}
    
    # Overall field comparison with both normalization methods
    metrics['overall'] = {
        'mse': float(mse_metric(current_field_resized, ref_field_resized)),
        'mae': float(mae_metric(current_field_resized, ref_field_resized)),
        'normalized_rmse_std': float(normalized_rmse(current_field_resized, ref_field_resized)),
        'correlation': float(correlation_coefficient(current_field_resized, ref_field_resized)),
        'ssim': float(structural_similarity_index(current_field_resized, ref_field_resized))
    }
    
    # Component-wise comparison
    n_components = min(current_field_resized.shape[-1], ref_field_resized.shape[-1])
    for i in range(n_components):
        name = field_names[i] if (field_names is not None and i < len(field_names)) else f'component_{i}'
        
        current_comp = current_field_resized[..., i]
        ref_comp = ref_field_resized[..., i]
        
        metrics[name] = {
            'mse': float(mse_metric(current_comp, ref_comp)),
            'mae': float(mae_metric(current_comp, ref_comp)), 
            'normalized_rmse_std': float(normalized_rmse(current_comp, ref_comp)),
            'correlation': float(correlation_coefficient(current_comp, ref_comp)),
            'ssim': float(structural_similarity_index(current_comp, ref_comp))
        }
        
        # Statistical comparison
        stats1 = compute_field_statistics(current_comp)
        stats2 = compute_field_statistics(ref_comp)
        
        metrics[name]['statistical_differences'] = {
            'mean_diff': float(stats1['mean'] - stats2['mean']),
            'std_ratio': float(stats1['std'] / (stats2['std'] + 1e-10)),
            'skewness_diff': float(stats1['skewness'] - stats2['skewness']),
            'kurtosis_diff': float(stats1['kurtosis'] - stats2['kurtosis'])
        }
    
    # Power spectrum comparison using existing data (unchanged)
    print("Computing power spectrum comparison metrics...")
    try:
        ps_metrics = power_spectrum_comparison_from_existing(
            current_k_centers, current_power_spec, 
            ref_k_centers, ref_power_spec,
            fft1=current_fft_final, fft2=ref_fft_final,
            field_names=field_names
        )
        metrics['power_spectrum'] = ps_metrics
    except Exception as e:
        print(f"Warning: Power spectrum comparison failed: {e}")
        metrics['power_spectrum'] = {}
    
    return metrics

def compute_all_comparison_metrics(field1, field2, box_size1, box_size2=None, field_names=None, 
                                 resize_method='fourier', k_centers1=None, power_spec1=None, 
                                 k_centers2=None, power_spec2=None, fft1=None, fft2=None):
    """
    Compute comprehensive comparison metrics between two phase fields.
    Enhanced version that can use pre-computed FFT and power spectrum data.
    
    Parameters:
    -----------
    field1, field2 : arrays
        Phase fields to compare with shape (..., n_components)
    box_size1, box_size2 : float
        Physical box sizes
    field_names : list
        Names of field components ['density', 'vx', 'vy', 'vz']
    resize_method : str
        Method for resizing fields ('fourier' or 'trilinear')
    k_centers1, k_centers2 : arrays, optional
        Pre-computed k-centers for power spectra
    power_spec1, power_spec2 : arrays, optional
        Pre-computed power spectra
    fft1, fft2 : arrays, optional
        Pre-computed FFT data
    
    Returns:
    --------
    dict : Comprehensive comparison metrics
    """
    if field_names is None:
        field_names = ['density', 'vx', 'vy', 'vz']
    
    if box_size2 is None:
        box_size2 = box_size1
    
    if field1.shape[:3] != field2.shape[:3]:
        sz1 = np.array(field1.shape[:3]); sz2 = np.array(field2.shape[:3])
        vox1 = int(np.prod(sz1)); vox2 = int(np.prod(sz2))
        if vox1 > vox2:
            print(f"Resizing field1 from {tuple(sz1)} to {tuple(sz2)} using {resize_method} method")
            field1_resized = resize_field_3d(field1, tuple(sz2), method=resize_method)
            field2_resized = field2
        else:
            print(f"Resizing field2 from {tuple(sz2)} to {tuple(sz1)} using {resize_method} method")
            field2_resized = resize_field_3d(field2, tuple(sz1), method=resize_method)
            field1_resized = field1
    else:
        field1_resized = field1
        field2_resized = field2

    metrics = {}
    
    # Overall field comparison with both normalization methods
    metrics['overall'] = {
        'mse': float(mse_metric(field1_resized, field2_resized)),
        'mae': float(mae_metric(field1_resized, field2_resized)),
        'normalized_rmse_std': float(normalized_rmse(field1_resized, field2_resized)),
        'correlation': float(correlation_coefficient(field1_resized, field2_resized)),
        'ssim': float(structural_similarity_index(field1_resized, field2_resized))
    }
    
    # Component-wise comparison
    n_components = min(field1_resized.shape[-1], field2_resized.shape[-1])
    for i in range(n_components):
        name = field_names[i] if (field_names is not None and i < len(field_names)) else f'component_{i}'
        f1_comp = field1_resized[..., i]
        f2_comp = field2_resized[..., i]
        metrics[name] = {
            'mse': float(mse_metric(f1_comp, f2_comp)),
            'mae': float(mae_metric(f1_comp, f2_comp)), 
            'normalized_rmse_std': float(normalized_rmse(f1_comp, f2_comp)),
            'correlation': float(correlation_coefficient(f1_comp, f2_comp)),
            'ssim': float(structural_similarity_index(f1_comp, f2_comp))
        }
        stats1 = compute_field_statistics(f1_comp)
        stats2 = compute_field_statistics(f2_comp)
        metrics[name]['statistical_differences'] = {
            'mean_diff': float(stats1['mean'] - stats2['mean']),
            'std_ratio': float(stats1['std'] / (stats2['std'] + 1e-10)),
            'skewness_diff': float(stats1['skewness'] - stats2['skewness']),
            'kurtosis_diff': float(stats1['kurtosis'] - stats2['kurtosis'])
        }

    # Power spectrum comparison
    if k_centers1 is not None and power_spec1 is not None and k_centers2 is not None and power_spec2 is not None:
        # Use pre-computed power spectrum data
        try:
            ps_metrics = power_spectrum_comparison_from_existing(
                k_centers1, power_spec1, k_centers2, power_spec2,
                fft1=fft1, fft2=fft2, field_names=field_names
            )
            metrics['power_spectrum'] = ps_metrics
        except Exception as e:
            print(f"Warning: Power spectrum comparison failed: {e}")
            metrics['power_spectrum'] = {}
    else:
        # Compute power spectra from fields (fallback)
        try:
            from utils import compute_fft_and_power_spectrum_3d_fields
            fft1_computed, k_centers1_computed, power_spec1_computed = compute_fft_and_power_spectrum_3d_fields(field1_resized, box_size1)
            fft2_computed, k_centers2_computed, power_spec2_computed = compute_fft_and_power_spectrum_3d_fields(field2, box_size2)
            
            ps_metrics = power_spectrum_comparison_from_existing(
                k_centers1_computed, power_spec1_computed, 
                k_centers2_computed, power_spec2_computed,
                fft1=fft1_computed, fft2=fft2_computed, field_names=field_names
            )
            metrics['power_spectrum'] = ps_metrics
        except Exception as e:
            print(f"Warning: Power spectrum comparison failed: {e}")
            metrics['power_spectrum'] = {}
    
    return metrics

def save_comparison_results_csv(metrics, save_path):
    """Save comparison metrics to a CSV file with hierarchical structure flattened."""
    # Flatten the nested dictionary structure
    flattened_metrics = {}
    
    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (jnp.ndarray, np.ndarray)):
                # For arrays, just store the mean for CSV (full arrays go to NPZ)
                if v.size == 1:
                    items.append((new_key, float(v)))
                else:
                    items.append((f"{new_key}_mean", float(jnp.mean(v))))
            else:
                items.append((new_key, v))
        return dict(items)
    
    flattened_metrics = flatten_dict(metrics)
    
    # Convert to DataFrame and save
    df = pd.DataFrame([flattened_metrics])
    df.to_csv(save_path.replace('.npz', '.csv'), index=False)
    
    # Also save full data including arrays to NPZ
    np.savez_compressed(save_path, **metrics)
    
    print(f"Comparison metrics saved to:")
    print(f"  - CSV: {save_path.replace('.npz', '.csv')}")
    print(f"  - NPZ: {save_path}")

def plot_comparison_summary(metrics, save_path):
    """Create comprehensive comparison summary plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Field Comparison Summary', fontsize=16, fontweight='bold')
    
    field_names = ['density', 'vx', 'vy', 'vz']
    colors = ['blue', 'red', 'green', 'orange']
    
    # 1. Correlation coefficients
    ax = axes[0, 0]
    correlations = []
    labels = []
    for name in field_names:
        if name in metrics and 'correlation' in metrics[name]:
            correlations.append(metrics[name]['correlation'])
            labels.append(name.capitalize())
    
    if correlations:
        bars = ax.bar(labels, correlations, color=colors[:len(correlations)])
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title('Field Correlations')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, correlations):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom')
    
    # 2. SSIM values
    ax = axes[0, 1]
    ssim_values = []
    for name in field_names:
        if name in metrics and 'ssim' in metrics[name]:
            ssim_values.append(metrics[name]['ssim'])
    
    if ssim_values:
        bars = ax.bar(labels[:len(ssim_values)], ssim_values, color=colors[:len(ssim_values)])
        ax.set_ylabel('SSIM')
        ax.set_title('Structural Similarity Index')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, ssim_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom')
    
    # 3. Normalized RMSE (std-based)
    ax = axes[0, 2]
    nrmse_values = []
    for name in field_names:
        if name in metrics and 'normalized_rmse_std' in metrics[name]:
            nrmse_values.append(metrics[name]['normalized_rmse_std'])
    
    if nrmse_values:
        bars = ax.bar(labels[:len(nrmse_values)], nrmse_values, color=colors[:len(nrmse_values)])
        ax.set_ylabel('Normalized RMSE (std)')
        ax.set_title('Normalized Root Mean Square Error')
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, nrmse_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom')
    
    # 4. Power spectrum correlations
    ax = axes[1, 0]
    ps_correlations = []
    for name in field_names:
        if 'power_spectrum' in metrics and f'{name}_ps_correlation' in metrics['power_spectrum']:
            val = metrics['power_spectrum'][f'{name}_ps_correlation']
            if not jnp.isnan(val):
                ps_correlations.append(val)
            else:
                ps_correlations.append(0)
    
    if ps_correlations:
        bars = ax.bar(labels[:len(ps_correlations)], ps_correlations, color=colors[:len(ps_correlations)])
        ax.set_ylabel('Power Spectrum Correlation')
        ax.set_title('Power Spectrum Correlations')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, ps_correlations):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom')
    
    # 5. Coherence (large scale)
    ax = axes[1, 1]
    coherence_values = []
    for name in field_names:
        if 'power_spectrum' in metrics and f'{name}_coherence_large_scale' in metrics['power_spectrum']:
            val = metrics['power_spectrum'][f'{name}_coherence_large_scale']
            if not jnp.isnan(val):
                coherence_values.append(val)
            else:
                coherence_values.append(0)
    
    if coherence_values:
        bars = ax.bar(labels[:len(coherence_values)], coherence_values, color=colors[:len(coherence_values)])
        ax.set_ylabel('Large-Scale Coherence')
        ax.set_title('Large-Scale Coherence r(k)')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, coherence_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom')
    
    # 6. Agreement scales
    ax = axes[1, 2]
    agreement_scales = []
    for name in field_names:
        if 'power_spectrum' in metrics and f'{name}_agreement_scale' in metrics['power_spectrum']:
            val = metrics['power_spectrum'][f'{name}_agreement_scale']
            if not jnp.isnan(val):
                agreement_scales.append(val)
            else:
                agreement_scales.append(0)
    
    if agreement_scales:
        bars = ax.bar(labels[:len(agreement_scales)], agreement_scales, color=colors[:len(agreement_scales)])
        ax.set_ylabel('Agreement Scale k [kpc⁻¹]')
        ax.set_title('Agreement Scale (r(k) ≥ 0.9)')
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, agreement_scales):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig

def print_comparison_summary(metrics):
    """Print a comprehensive summary of the most important comparison metrics."""
    print("\n" + "="*60)
    print("           FIELD COMPARISON SUMMARY")
    print("="*60)
    
    # Overall metrics
    print("\nOVERALL METRICS:")
    print(f"  Correlation:           {metrics['overall']['correlation']:.4f}")
    print(f"  SSIM:                  {metrics['overall']['ssim']:.4f}")
    print(f"  Normalized RMSE (std): {metrics['overall']['normalized_rmse_std']:.4f}")
    
    # Component-wise metrics
    field_names = ['density', 'vx', 'vy', 'vz']
    for name in field_names:
        if name in metrics:
            print(f"\n{name.upper()} FIELD:")
            print(f"  Correlation:           {metrics[name]['correlation']:.4f}")
            print(f"  SSIM:                  {metrics[name]['ssim']:.4f}")
            print(f"  Normalized RMSE (std): {metrics[name]['normalized_rmse_std']:.4f}")
            
            # Power spectrum metrics
            if 'power_spectrum' in metrics:
                ps_corr_key = f'{name}_ps_correlation'
                coherence_key = f'{name}_coherence_large_scale'
                agreement_key = f'{name}_agreement_scale'
                
                if ps_corr_key in metrics['power_spectrum']:
                    ps_corr = metrics['power_spectrum'][ps_corr_key]
                    if not jnp.isnan(ps_corr):
                        print(f"  Power Spectrum Corr:   {ps_corr:.4f}")
                
                if coherence_key in metrics['power_spectrum']:
                    coherence = metrics['power_spectrum'][coherence_key]
                    if not jnp.isnan(coherence):
                        print(f"  Large-Scale Coherence: {coherence:.4f}")
                
                if agreement_key in metrics['power_spectrum']:
                    agreement = metrics['power_spectrum'][agreement_key]
                    if not jnp.isnan(agreement):
                        print(f"  Agreement Scale:       {agreement:.3f} kpc⁻¹")
    
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE:")
    print("  Correlation & SSIM:     Higher is better (max = 1.0)")
    print("  Normalized RMSE:        Lower is better (min = 0.0)")
    print("  Power Spectrum Corr:    Measures agreement in Fourier space")
    print("  Large-Scale Coherence:  Agreement at largest scales")
    print("  Agreement Scale:        Highest k where coherence ≥ 0.9")
    print("="*60 + "\n")

def save_comparison_results(metrics, save_path):
    """Backward compatible function - now saves both CSV and NPZ."""
    save_comparison_results_csv(metrics, save_path)

# Maintain backward compatibility
def load_comparison_results(load_path):
    """Load comparison results from NPZ file."""
    return dict(np.load(load_path, allow_pickle=True))