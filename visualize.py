"""
Visualization Functions for MRI and Segmentation Results
Includes 2D slice viewer and 3D volume rendering
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Color map for tumor classes
TUMOR_COLORS = {
    0: [0, 0, 0, 0],           # Background - transparent
    1: [255, 0, 0, 180],       # Necrotic core - red
    2: [0, 255, 0, 120],       # Edema - green
    3: [255, 255, 0, 200],     # Enhancing tumor - yellow
}

CLASS_NAMES = {
    0: "Background",
    1: "Necrotic Core",
    2: "Edema",
    3: "Enhancing Tumor"
}


def plot_slice(mri_data, slice_idx, axis=2, title="MRI Slice", figsize=(8, 8), cmap='gray'):
    """
    Plot a single 2D slice from 3D MRI data
    
    Args:
        mri_data: 3D numpy array
        slice_idx: index of slice to display
        axis: axis along which to slice (0, 1, or 2)
        title: plot title
        figsize: figure size
        cmap: colormap
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if axis == 0:
        slice_data = mri_data[slice_idx, :, :]
    elif axis == 1:
        slice_data = mri_data[:, slice_idx, :]
    else:
        slice_data = mri_data[:, :, slice_idx]
    
    im = ax.imshow(slice_data.T, cmap=cmap, origin='lower')
    ax.set_title(f"{title} (Slice {slice_idx})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('off')
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    plt.tight_layout()
    return fig


def plot_overlay(mri_data, segmentation, slice_idx, axis=2, alpha=0.4, figsize=(12, 6)):
    """
    Plot MRI with segmentation overlay
    
    Args:
        mri_data: 3D numpy array
        segmentation: 3D segmentation mask
        slice_idx: slice index
        axis: slicing axis
        alpha: overlay transparency
        figsize: figure size
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Extract slices
    if axis == 0:
        mri_slice = mri_data[slice_idx, :, :]
        seg_slice = segmentation[slice_idx, :, :]
    elif axis == 1:
        mri_slice = mri_data[:, slice_idx, :]
        seg_slice = segmentation[:, slice_idx, :]
    else:
        mri_slice = mri_data[:, :, slice_idx]
        seg_slice = segmentation[:, :, slice_idx]
    
    # Original MRI
    axes[0].imshow(mri_slice.T, cmap='gray', origin='lower')
    axes[0].set_title('Original MRI')
    axes[0].axis('off')
    
    # Segmentation only
    seg_colored = np.zeros((*seg_slice.shape, 4))
    for class_id, color in TUMOR_COLORS.items():
        mask = seg_slice == class_id
        seg_colored[mask] = np.array(color) / 255.0
    
    axes[1].imshow(seg_colored.transpose(1, 0, 2), origin='lower')
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(mri_slice.T, cmap='gray', origin='lower')
    axes[2].imshow(seg_colored.transpose(1, 0, 2), alpha=alpha, origin='lower')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array(TUMOR_COLORS[i][:3])/255, label=CLASS_NAMES[i])
        for i in range(1, 4)
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    return fig


def plot_multiplane(mri_data, segmentation, slice_indices=None, alpha=0.4, figsize=(15, 5)):
    """
    Plot three orthogonal planes (axial, sagittal, coronal)
    
    Args:
        mri_data: 3D numpy array
        segmentation: 3D segmentation mask
        slice_indices: tuple of (axial, sagittal, coronal) indices
        alpha: overlay transparency
        figsize: figure size
        
    Returns:
        matplotlib figure
    """
    if slice_indices is None:
        slice_indices = (
            mri_data.shape[0] // 2,
            mri_data.shape[1] // 2,
            mri_data.shape[2] // 2
        )
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    titles = ['Axial', 'Sagittal', 'Coronal']
    
    for idx, (ax, axis, title) in enumerate(zip(axes, range(3), titles)):
        # Extract slice
        if axis == 0:
            mri_slice = mri_data[slice_indices[axis], :, :]
            seg_slice = segmentation[slice_indices[axis], :, :]
        elif axis == 1:
            mri_slice = mri_data[:, slice_indices[axis], :]
            seg_slice = segmentation[:, slice_indices[axis], :]
        else:
            mri_slice = mri_data[:, :, slice_indices[axis]]
            seg_slice = segmentation[:, :, slice_indices[axis]]
        
        # Create colored segmentation
        seg_colored = np.zeros((*seg_slice.shape, 4))
        for class_id, color in TUMOR_COLORS.items():
            mask = seg_slice == class_id
            seg_colored[mask] = np.array(color) / 255.0
        
        # Plot
        ax.imshow(mri_slice.T, cmap='gray', origin='lower')
        ax.imshow(seg_colored.transpose(1, 0, 2), alpha=alpha, origin='lower')
        ax.set_title(f'{title} View (Slice {slice_indices[axis]})')
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_3d_volume(segmentation, downsample=2):
    """
    Create 3D volume visualization using Plotly
    
    Args:
        segmentation: 3D segmentation mask
        downsample: downsampling factor to reduce data size
        
    Returns:
        plotly figure
    """
    # Downsample for performance
    seg_downsampled = segmentation[::downsample, ::downsample, ::downsample]
    
    # Create figure
    fig = go.Figure()
    
    # Add each tumor class as separate surface
    for class_id in [1, 2, 3]:  # Skip background
        # Extract voxels for this class
        mask = seg_downsampled == class_id
        
        if np.sum(mask) == 0:
            continue
        
        # Get coordinates
        x, y, z = np.where(mask)
        
        # Get color
        color = TUMOR_COLORS[class_id]
        color_str = f'rgba({color[0]}, {color[1]}, {color[2]}, {color[3]/255})'
        
        # Add scatter plot
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=color_str,
                symbol='square'
            ),
            name=CLASS_NAMES[class_id]
        ))
    
    # Update layout
    fig.update_layout(
        title="3D Tumor Segmentation",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        height=700,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def create_interactive_slice_viewer(mri_data, segmentation=None):
    """
    Create interactive slice viewer with Plotly
    
    Args:
        mri_data: 3D numpy array
        segmentation: optional 3D segmentation mask
        
    Returns:
        plotly figure
    """
    num_slices = mri_data.shape[2]
    
    # Create subplots
    if segmentation is not None:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('MRI', 'Segmentation', 'Overlay'),
            horizontal_spacing=0.05
        )
    else:
        fig = go.Figure()
    
    # Add frames for each slice
    frames = []
    
    for i in range(num_slices):
        frame_data = []
        
        # MRI slice
        mri_slice = mri_data[:, :, i].T
        frame_data.append(
            go.Heatmap(
                z=mri_slice,
                colorscale='Gray',
                showscale=False,
                hovertemplate='x: %{x}<br>y: %{y}<br>intensity: %{z}<extra></extra>'
            )
        )
        
        if segmentation is not None:
            # Segmentation slice
            seg_slice = segmentation[:, :, i].T
            frame_data.append(
                go.Heatmap(
                    z=seg_slice,
                    colorscale=[
                        [0, 'rgba(0,0,0,0)'],
                        [0.33, 'rgba(255,0,0,0.7)'],
                        [0.66, 'rgba(0,255,0,0.5)'],
                        [1, 'rgba(255,255,0,0.8)']
                    ],
                    showscale=False,
                    hovertemplate='x: %{x}<br>y: %{y}<br>class: %{z}<extra></extra>'
                )
            )
            
            # Overlay
            frame_data.append(
                go.Heatmap(
                    z=mri_slice,
                    colorscale='Gray',
                    showscale=False,
                    hovertemplate='x: %{x}<br>y: %{y}<br>intensity: %{z}<extra></extra>'
                )
            )
        
        frames.append(go.Frame(data=frame_data, name=str(i)))
    
    # Add initial data
    if segmentation is not None:
        mid_slice = num_slices // 2
        fig.add_trace(
            go.Heatmap(
                z=mri_data[:, :, mid_slice].T,
                colorscale='Gray',
                showscale=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(
                z=segmentation[:, :, mid_slice].T,
                colorscale=[
                    [0, 'rgba(0,0,0,0)'],
                    [0.33, 'rgba(255,0,0,0.7)'],
                    [0.66, 'rgba(0,255,0,0.5)'],
                    [1, 'rgba(255,255,0,0.8)']
                ],
                showscale=False
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Heatmap(
                z=mri_data[:, :, mid_slice].T,
                colorscale='Gray',
                showscale=False
            ),
            row=1, col=3
        )
    
    # Add frames
    fig.frames = frames
    
    # Add slider
    sliders = [dict(
        active=num_slices // 2,
        yanchor="top",
        y=0,
        xanchor="left",
        x=0.1,
        currentvalue=dict(
            prefix="Slice: ",
            visible=True,
            xanchor="right"
        ),
        pad=dict(b=10, t=50),
        len=0.9,
        steps=[
            dict(
                args=[[f.name], dict(
                    frame=dict(duration=0, redraw=True),
                    mode="immediate"
                )],
                label=str(k),
                method="animate"
            )
            for k, f in enumerate(fig.frames)
        ]
    )]
    
    # Update layout
    fig.update_layout(
        sliders=sliders,
        height=500,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=100)
    )
    
    # Hide axes
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    
    return fig


def plot_probability_maps(probabilities, slice_idx, axis=2, figsize=(15, 4)):
    """
    Plot probability maps for each class
    
    Args:
        probabilities: 4D array (classes, H, W, D)
        slice_idx: slice index
        axis: slicing axis
        figsize: figure size
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    for i, ax in enumerate(axes):
        if axis == 0:
            prob_slice = probabilities[i, slice_idx, :, :]
        elif axis == 1:
            prob_slice = probabilities[i, :, slice_idx, :]
        else:
            prob_slice = probabilities[i, :, :, slice_idx]
        
        im = ax.imshow(prob_slice.T, cmap='hot', vmin=0, vmax=1, origin='lower')
        ax.set_title(CLASS_NAMES[i])
        ax.axis('off')
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    plt.tight_layout()
    return fig