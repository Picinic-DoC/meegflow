"""
Example custom preprocessing step for NICE EEG Preprocessing Pipeline.

This file demonstrates how to create custom preprocessing steps that can be
dynamically loaded by the pipeline. Custom steps are Python functions that
follow a specific signature and pattern.

To use custom steps:
1. Create a Python file (.py) in a dedicated folder
2. Define functions with the signature: func(data: Dict, step_config: Dict) -> Dict
3. Specify the folder path in your config file under 'custom_steps_folder'
4. Reference the custom step by its function name in your pipeline configuration

Example config.yaml:
```yaml
custom_steps_folder: /path/to/your/custom_steps
pipeline:
  - name: my_custom_filter
    cutoff_freq: 30.0
  - name: bandpass_filter  # Built-in steps still work
    l_freq: 0.5
    h_freq: 40.0
```

For Docker usage:
```bash
docker run -v /host/path/to/custom_steps:/custom_steps nice-preprocessing \\
  --config /path/to/config.yaml
```

And in config.yaml:
```yaml
custom_steps_folder: /custom_steps
```
"""

from typing import Dict, Any


def my_custom_filter(data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example custom preprocessing step: Apply a custom low-pass filter.
    
    This step demonstrates the required structure for custom preprocessing steps.
    All custom steps must:
    1. Accept exactly 2 parameters: data (Dict) and step_config (Dict)
    2. Return the updated data dictionary
    3. Validate that required data instances exist (e.g., 'raw', 'epochs')
    4. Append a summary to data['preprocessing_steps']
    5. Document parameters clearly
    
    Parameters (via step_config)
    -----------------------------
    cutoff_freq : float, optional
        Low-pass filter cutoff frequency in Hz (default: 30.0)
    
    Updates
    -------
    data['raw'] : mne.io.Raw
        Applies low-pass filter in-place
    data['preprocessing_steps'] : list
        Appends step information
    
    Returns
    -------
    data : dict
        Updated data dictionary with filtered raw data
    
    Raises
    ------
    ValueError
        If 'raw' is not present in data
    """
    # Validate that required data exists
    if 'raw' not in data:
        raise ValueError("my_custom_filter requires 'raw' to be in data")
    
    # Get parameters from step_config with defaults
    cutoff_freq = step_config.get('cutoff_freq', 30.0)
    
    # Apply the custom processing
    data['raw'].filter(h_freq=cutoff_freq, l_freq=None)
    
    # Record what was done for the report
    data['preprocessing_steps'].append({
        'step': 'my_custom_filter',
        'cutoff_freq': cutoff_freq,
        'description': f'Applied custom low-pass filter at {cutoff_freq} Hz'
    })
    
    return data


def mark_bad_channels_by_name(data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example custom step: Mark specific channels as bad by name.
    
    This demonstrates how to create a custom step that marks channels as bad
    without dropping them, allowing later interpolation.
    
    Parameters (via step_config)
    -----------------------------
    channels : list of str
        Channel names to mark as bad
    instance : str, optional
        Which data instance to work on: 'raw' or 'epochs' (default: 'raw')
    
    Updates
    -------
    data[instance].info['bads'] : list
        Adds specified channels to the bad channels list
    data['preprocessing_steps'] : list
        Appends step information
    
    Returns
    -------
    data : dict
        Updated data dictionary with marked bad channels
    """
    channels = step_config.get('channels', [])
    instance = step_config.get('instance', 'raw')
    
    if instance not in data:
        raise ValueError(f"mark_bad_channels_by_name requires '{instance}' to be in data")
    
    # Mark channels as bad
    existing_bads = set(data[instance].info['bads'])
    new_bads = set(channels)
    data[instance].info['bads'] = list(existing_bads | new_bads)
    
    # Record the action
    data['preprocessing_steps'].append({
        'step': 'mark_bad_channels_by_name',
        'instance': instance,
        'channels_marked': list(new_bads),
        'total_bads': len(data[instance].info['bads'])
    })
    
    return data


def compute_custom_metric(data: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example custom step: Compute a custom data quality metric.
    
    This demonstrates how to compute custom metrics without modifying the data.
    The metric is stored in the data dictionary and will appear in reports.
    
    Parameters (via step_config)
    -----------------------------
    metric_name : str, optional
        Name for the metric (default: 'custom_quality')
    instance : str, optional
        Which data instance to analyze: 'raw' or 'epochs' (default: 'raw')
    
    Updates
    -------
    data[metric_name] : float
        Computed metric value
    data['preprocessing_steps'] : list
        Appends step information
    
    Returns
    -------
    data : dict
        Updated data dictionary with computed metric
    """
    import numpy as np
    
    metric_name = step_config.get('metric_name', 'custom_quality')
    instance = step_config.get('instance', 'raw')
    
    if instance not in data:
        raise ValueError(f"compute_custom_metric requires '{instance}' to be in data")
    
    # Example metric: compute standard deviation across all channels
    raw_data = data[instance].get_data()
    metric_value = float(np.std(raw_data))
    
    # Store the metric
    data[metric_name] = metric_value
    
    # Record the computation
    data['preprocessing_steps'].append({
        'step': 'compute_custom_metric',
        'metric_name': metric_name,
        'metric_value': metric_value,
        'instance': instance
    })
    
    return data


# Note: Functions starting with underscore are NOT loaded as steps
def _helper_function():
    """Helper functions starting with _ are ignored by the loader."""
    pass


# Note: Functions with wrong signatures are NOT loaded as steps
def wrong_signature(only_one_param):
    """Functions without exactly 2 parameters are ignored."""
    pass
