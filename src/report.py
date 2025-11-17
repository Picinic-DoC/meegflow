"""
Report generation utilities for EEG preprocessing pipeline.

This module contains helper functions for generating HTML reports,
including bad channels visualization and preprocessing steps tables.
"""
import json
from typing import Dict, Any, List, Optional
import numpy as np
import mne
from mne.utils import logger
import matplotlib.pyplot as plt


def collect_bad_channels_from_steps(preprocessing_steps: List[Dict[str, Any]]) -> List[str]:
    """
    Collect all bad channels from preprocessing steps.
    
    Parameters
    ----------
    preprocessing_steps : list of dict
        List of preprocessing step dictionaries, each potentially containing
        a 'bad_channels' key.
    
    Returns
    -------
    bad_channels : list of str
        Unique list of all bad channels found in preprocessing steps.
    """
    bad_channels = []
    for step in preprocessing_steps:
        if 'bad_channels' in step and step['bad_channels']:
            # Add bad channels from this step
            step_bad_channels = step['bad_channels']
            if isinstance(step_bad_channels, list):
                bad_channels.extend(step_bad_channels)
            elif isinstance(step_bad_channels, str):
                bad_channels.append(step_bad_channels)
    
    # Return unique channels preserving order
    seen = set()
    unique_bad_channels = []
    for ch in bad_channels:
        if ch not in seen:
            seen.add(ch)
            unique_bad_channels.append(ch)
    
    return unique_bad_channels


def create_bad_channels_topoplot(
    info: mne.Info,
    bad_channels: List[str],
    figsize: tuple = (8, 6)
) -> Optional[plt.Figure]:
    """
    Create a topoplot showing bad channels marked with red crosses.
    
    Uses the montage from the info object to determine the appropriate
    head shape and electrode positions.
    
    Parameters
    ----------
    info : mne.Info
        MNE Info object containing channel information and montage.
    bad_channels : list of str
        List of bad channel names to mark on the topoplot.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (8, 6).
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure containing the topoplot, or None if creation failed.
    """
    if not bad_channels:
        logger.info("No bad channels to plot")
        return None
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get all EEG channel positions
        eeg_picks = mne.pick_types(info, eeg=True, exclude=[])
        
        if len(eeg_picks) == 0:
            logger.warning("No EEG channels found for topoplot")
            plt.close(fig)
            return None
        
        # Get channel names
        ch_names = [info['ch_names'][i] for i in eeg_picks]
        
        # Create data array (all zeros for white background)
        data_to_plot = np.zeros(len(eeg_picks))
        
        # Create mask for bad channels
        mask = np.array([ch in bad_channels for ch in ch_names])
        
        if not np.any(mask):
            logger.warning("None of the bad channels are in the EEG channels")
            plt.close(fig)
            return None
        
        # Plot topomap with white background
        # The montage from info will be used automatically by plot_topomap
        from mne.viz import plot_topomap
        im, cn = plot_topomap(
            data_to_plot, 
            info,
            axes=ax,
            show=False,
            cmap='Greys',
            vlim=(0, 0.1),
            mask=mask,
            mask_params=dict(
                marker='x',
                markerfacecolor='red',
                markeredgecolor='red',
                linewidth=0,
                markersize=15
            ),
            sensors=True,
            contours=0
        )
        
        ax.set_title(f'Bad Channels (n={len(bad_channels)})', fontsize=14, fontweight='bold')
        
        # Add text listing bad channels
        bad_channels_text = ', '.join(bad_channels)
        fig.text(0.5, 0.05, f'Bad channels: {bad_channels_text}', 
                ha='center', fontsize=10, wrap=True)
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        logger.warning(f"Failed to create bad channels topoplot: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def create_preprocessing_steps_table(preprocessing_steps: List[Dict[str, Any]]) -> str:
    """
    Create an HTML table with collapsible rows for preprocessing steps.
    
    Uses MNE Report table styling with bootstrap-table classes.
    Each step's parameters are displayed in a two-column table:
    - First column: parameter keys
    - Second column: parameter values
      - Numbers displayed as numbers
      - Lists displayed as bullet points
      - Dicts displayed as prettified JSON with indent 4
    
    Parameters
    ----------
    preprocessing_steps : list of dict
        List of preprocessing step dictionaries containing step information.
    
    Returns
    -------
    html_content : str
        HTML string containing the styled, collapsible table.
    """
    if not preprocessing_steps:
        return ""
    
    def format_value(value):
        """Format a value based on its type."""
        if isinstance(value, dict):
            # Format dicts as prettified JSON with indent 4
            return f'<pre style="margin: 0; background-color: #f8f9fa; padding: 8px; border-radius: 4px;">{json.dumps(value, indent=4)}</pre>'
        elif isinstance(value, list):
            # Format lists as bullet points
            if not value:
                return '<em>empty list</em>'
            bullet_points = ''.join([f'<li>{item}</li>' for item in value])
            return f'<ul style="margin: 0; padding-left: 20px;">{bullet_points}</ul>'
        elif isinstance(value, (int, float)):
            # Format numbers as numbers (not strings)
            return str(value)
        elif value is None:
            return '<em>None</em>'
        else:
            # Format everything else as string
            return str(value)
    
    # Create HTML with collapsible sections for each step
    html_content = """
    <style>
        .step-container {
            margin-bottom: 20px;
        }
        .step-header {
            cursor: pointer;
            padding: 12px;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            font-weight: bold;
            user-select: none;
            transition: background-color 0.2s;
        }
        .step-header:hover {
            background-color: #e9ecef;
        }
        .step-header .toggle-icon {
            float: right;
            font-weight: bold;
        }
        .step-details {
            display: none;
            margin-top: 10px;
        }
        .step-details.active {
            display: block;
        }
        .params-table {
            width: 100%;
            margin-top: 10px;
        }
        .params-table td {
            padding: 8px;
            border: 1px solid #dee2e6;
            vertical-align: top;
        }
        .params-table td:first-child {
            background-color: #f8f9fa;
            font-weight: 500;
            width: 30%;
        }
    </style>
    <script>
        function toggleStep(stepId) {
            var details = document.getElementById('details-' + stepId);
            var icon = document.getElementById('icon-' + stepId);
            if (details.classList.contains('active')) {
                details.classList.remove('active');
                icon.textContent = '▼';
            } else {
                details.classList.add('active');
                icon.textContent = '▲';
            }
        }
    </script>
    """
    
    for idx, step in enumerate(preprocessing_steps, 1):
        step_name = step.get('step', 'Unknown')
        step_id = f"step-{idx}"
        
        # Create collapsible section for this step
        html_content += f"""
        <div class="step-container">
            <div class="step-header" onclick="toggleStep('{step_id}')">
                <span>Step {idx}: {step_name}</span>
                <span class="toggle-icon" id="icon-{step_id}">▼</span>
            </div>
            <div class="step-details" id="details-{step_id}">
                <table class="params-table table table-hover">
                    <tbody>
        """
        
        # Add each parameter as a row in the table
        for key, value in step.items():
            if key != 'step':
                formatted_value = format_value(value)
                html_content += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{formatted_value}</td>
                        </tr>
                """
        
        html_content += """
                    </tbody>
                </table>
            </div>
        </div>
        """
    
    return html_content
