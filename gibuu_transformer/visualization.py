"""
Visualization and evaluation utilities for GiBUU Transformer.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import confusion_matrix
from .constants import EOS_STEP_TOKEN, EOS_TOKEN, PAD_TOKEN, FEATS_MEAN, FEATS_SIGMA, GIBUU_CHARGE_TO_PDG, PDG_COLOR_MAP
from .data_processing import decode_id


def load_and_plot_loss_curves(log_dir):
    """
    Load and plot loss curves from Lightning's CSV logger.
    Plots both loss vs. step and loss vs. epoch, ignoring NaNs.
    """
    # Find the CSV file created by Lightning
    csv_files = [f for f in Path(log_dir).iterdir() if f.suffix == '.csv']
    
    if not csv_files:
        print(f"No CSV files found in: {log_dir}")
        return None

    csv_path = csv_files[0]  # Usually only one CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded loss curves from: {csv_path}")
    print(f"Columns: {list(df.columns)}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    def plot_loss(ax, train_col, val_col, title, logy=False, custom_epoch_ticks=None):
        # Plot train loss (all steps, ignore NaN)
        if train_col in df.columns:
            train_mask = ~df[train_col].isna()
            ax.plot(df['step'][train_mask][::1], df[train_col][train_mask][::1], label='Train', alpha=0.7)
        # Plot val loss (only at epoch end, ignore NaN)
        if val_col in df.columns:
            val_mask = ~df[val_col].isna()
            ax.plot(df['step'][val_mask], df[val_col][val_mask], 'o-', label='Validation', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        if logy:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Add secondary x-axis for epoch
        if 'epoch' in df.columns:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            if custom_epoch_ticks is not None:
                # Map each custom epoch tick to the first step where that epoch appears
                step_ticks = []
                for e in custom_epoch_ticks:
                    steps = df.loc[df['epoch'] == e, 'step']
                    if not steps.empty:
                        step_ticks.append(steps.iloc[0])
                    else:
                        # If epoch not found, interpolate
                        step_ticks.append(np.interp(e, df['epoch'], df['step']))
                ax2.set_xticks(step_ticks)
                ax2.set_xticklabels(custom_epoch_ticks)
            elif val_col in df.columns and val_mask.any():
                epoch_ticks = df['epoch'][val_mask].astype(int)+1
                step_ticks = df['step'][val_mask]
                ax2.set_xticks(step_ticks)
                ax2.set_xticklabels(epoch_ticks)
            else:
                ax2.set_xticks(ax.get_xticks())
                ax2.set_xticklabels(['']*len(ax.get_xticks()))
            ax2.set_xlabel('Epoch')

    logy = True
    custom_epoch_ticks = np.arange(0, 290+1, 20)
    plot_loss(axes[0, 0], 'train_loss', 'val_loss', 'Total Loss', logy, custom_epoch_ticks)
    plot_loss(axes[0, 1], 'train_type_loss', 'val_type_loss', 'Type Loss', logy, custom_epoch_ticks)
    plot_loss(axes[1, 0], 'train_feat_loss', 'val_feat_loss', 'Feature Loss', logy, custom_epoch_ticks)

    # Hide unused subplot if only 3 plots
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    return df


def count_position_matches(list1, list2, ignore_values=None, consider_values=None):
    """
    Counts how many elements match in the same position in two lists,
    with options to ignore and/or only consider certain values.

    - If ignore_values is provided, positions where either value is in ignore_values are skipped.
    - If consider_values is provided, only positions where both values are in consider_values are considered (after ignoring).
    - Returns a tuple: (number of matches, total positions considered)
    """
    ignore_set = set(ignore_values) if ignore_values is not None else set()
    consider_set = set(consider_values) if consider_values is not None else None

    filtered = []
    for a, b in zip(list1, list2):
        if a in ignore_set or b in ignore_set:
            continue
        if consider_set is not None and (a not in consider_set or b not in consider_set):
            continue
        filtered.append((a, b))

    matches = sum(1 for a, b in filtered if a == b)
    total_considered = len(filtered)
    return matches, total_considered


def plot_2d_feature_distribution(
    all_feats_pred, all_feats_target, all_preds, all_targets,
    feature_index, matched=False, particle_id=None, bins=None, vmin=None, vmax=None, 
    xlabel="Predicted", ylabel="True", title="2D Feature Distribution"
):
    """
    Draws a 2D histogram of a feature: predicted vs. true.
    
    Parameters:
    -----------
    all_feats_pred: tensor/array, shape [N, seq_len, n_feat]
    all_feats_target: tensor/array, shape [N, seq_len, n_feat]
    all_preds: tensor/array, shape [N, seq_len]
    all_targets: tensor/array, shape [N, seq_len]
    feature_index: int, which feature to plot ([x,y,z,E,Px,Py,Pz] e.g., 3 for E)
    matched: bool, if True, only plot where predicted ID == true ID
    particle_id: int, list or None, if set, only plot for the provided IDs
    bins: list, bin edges for the histogram
    vmin, vmax: for color scale
    xlabel, ylabel, title: plot labels
    """
    # Flatten everything
    pred_ids = all_preds.flatten()
    true_ids = all_targets.flatten()
    pred_feat = all_feats_pred[..., feature_index].flatten()
    true_feat = all_feats_target[..., feature_index].flatten()
    
    # Mask for non-particle tokens
    mask = (true_ids != PAD_TOKEN) & (true_ids != EOS_STEP_TOKEN) & (true_ids != EOS_TOKEN)
    if matched:
        mask = mask & (pred_ids == true_ids)
    if particle_id is not None:
        if type(particle_id) is int:
            particle_id = [particle_id]
        mask = mask & np.isin(true_ids, particle_id)
    
    # Apply mask
    mask = mask.bool()
    x = pred_feat[mask]
    y = true_feat[mask]

    # Set the same range for both axes
    if bins is None:
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        bins = np.linspace(min_val, max_val, 101)
    xmin = bins[0]
    xmax = bins[-1]

    # Set up colormap with white for empty bins
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('white')
    
    plt.figure(figsize=(7, 6))
    plt.hist2d(x, y, bins=bins, cmap=cmap, cmin=1, vmin=vmin, vmax=vmax)
    plt.plot([xmin, xmax], [xmin, xmax], "r:") # reference line
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlim(xmin, xmax)
    plt.ylim(xmin, xmax)
    plt.title(title + (f" (ID={particle_id})" if particle_id is not None else " (all particle IDs)"))
    plt.colorbar(label="Counts")
    plt.tight_layout()
    plt.show()


def extract_visualization_lists_from_output_sequence(output_sequence, gibuu_charge_to_pdg):
    """
    Given output_sequence and a lookup table, return positions_list, momenta_list, pdg_codes_list, timestep_list.
    
    Parameters:
    -----------
    output_sequence: list of [time_step, gibuuID, charge, x, y, z, E, Px, Py, Pz]
    gibuu_charge_to_pdg: dict mapping (gibuuID, charge) -> PDG code
    
    Returns:
    --------
    positions_list: list of np.ndarray, each shape (n_particles, 3)
    momenta_list: list of np.ndarray, each shape (n_particles, 3)
    pdg_codes_list: list of np.ndarray, each shape (n_particles,)
    timestep_list: list of int
    """
    positions_list = []
    momenta_list = []
    pdg_codes_list = []
    timestep_list = []

    current_timestep = None
    cur_positions = []
    cur_momenta = []
    cur_pdgs = []

    for row in output_sequence:
        time_step, gibuuID, charge, x, y, z, E, Px, Py, Pz = row
        if current_timestep is None:
            current_timestep = time_step

        if time_step != current_timestep:
            # Save the previous time step's data
            if cur_positions:
                positions_list.append(np.array(cur_positions))
                momenta_list.append(np.array(cur_momenta))
                pdg_codes_list.append(np.array(cur_pdgs))
                timestep_list.append(current_timestep)
            # Start new time step
            current_timestep = time_step
            cur_positions = []
            cur_momenta = []
            cur_pdgs = []

        # Lookup PDG code
        pdg = gibuu_charge_to_pdg.get((gibuuID, charge), 0)  # 0 if not found
        cur_positions.append([x, y, z])
        cur_momenta.append([Px, Py, Pz])
        cur_pdgs.append(pdg)

    # Don't forget the last time step
    if cur_positions:
        positions_list.append(np.array(cur_positions))
        momenta_list.append(np.array(cur_momenta))
        pdg_codes_list.append(np.array(cur_pdgs))
        timestep_list.append(current_timestep)

    return positions_list, momenta_list, pdg_codes_list, timestep_list


def visualize_particles_with_slider(
    positions_list, momenta_list, pdg_codes_list, 
    timestep_list=None,
    xlim=None, ylim=None, zlim=None
):
    """
    Visualize particles at each time step with an interactive slider.
    """
    try:
        from ipywidgets import widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets not available. Please install with: pip install ipywidgets")
        return
    
    n_steps = len(positions_list)

    # Create figure and axis once
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    def plot_frame(step):
        ax.clear()
        if False: # Draw a sphere of radius centered at the origin
            radius = 5
            u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
            sphere_x = radius * np.cos(u) * np.sin(v)
            sphere_y = radius * np.sin(u) * np.sin(v)
            sphere_z = radius * np.cos(v)
            ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color='gray', alpha=0.2, linewidth=0.5)
            ax.plot_surface(sphere_x, sphere_y, sphere_z, color='gray', alpha=0.05, linewidth=0, shade=True)
        
        positions = np.asarray(positions_list[step])
        momenta = np.asarray(momenta_list[step])
        pdg_codes = np.asarray(pdg_codes_list[step])

        for i in range(len(positions)):
            x, y, z = positions[i]
            px, py, pz = momenta[i]
            color = PDG_COLOR_MAP.get(pdg_codes[i], 'black')

            # Draw momentum vector (arrow)
            ax.quiver(x, y, z, px, py, pz, length=1, normalize=True, 
                      arrow_length_ratio=0.1, color=color)
            # Draw particle position (marker)
            ax.scatter(x, y, z, color=color, marker='o', s=50)

        ax.set_xlabel('X [fm]')
        ax.set_ylabel('Y [fm]')
        ax.set_zlabel('Z [fm]')
        if timestep_list:
            ax.set_title(f'Particle Visualization - Time Step {timestep_list[step]} ({len(pdg_codes)} particles)')
        else:
            ax.set_title(f'Particle Visualization - Time Step {step} ({len(pdg_codes)} particles)')

        # Apply fixed limits if provided
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        if zlim:
            ax.set_zlim(*zlim)

        fig.canvas.draw_idle()  # Efficient redraw

    # Create and display the slider
    slider = widgets.IntSlider(
        min=0, max=n_steps - 1, step=1, value=0, description='Time Step'
    )

    def on_slider_change(change):
        if change['name'] == 'value':
            plot_frame(change['new'])

    slider.observe(on_slider_change)
    display(slider)

    # Initial plot
    plot_frame(0)


def save_particles_gif(
    positions_list, momenta_list, pdg_codes_list,
    timestep_list=None,
    xlim=None, ylim=None, zlim=None,
    step_range=None,  # e.g. [1, 152]
    filename="particle_evolution.gif",
    fps=5
):
    """
    Save particle visualization as an animated GIF.
    """
    if step_range is None:
        step_range = (0, len(positions_list))

    start_step, end_step = step_range
    if start_step < 1:
        start_step = 1
    if end_step > len(positions_list):
        end_step = len(positions_list)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    def plot_frame(step):
        ax.clear()
        if True: # Draw a sphere of radius centered at the origin
            radius = 5
            u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
            sphere_x = radius * np.cos(u) * np.sin(v)
            sphere_y = radius * np.sin(u) * np.sin(v)
            sphere_z = radius * np.cos(v)
            ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color='gray', alpha=0.2, linewidth=0.5)
        
        positions = np.asarray(positions_list[step])
        momenta = np.asarray(momenta_list[step])
        pdg_codes = np.asarray(pdg_codes_list[step])

        for i in range(len(positions)):
            x, y, z = positions[i]
            px, py, pz = momenta[i]
            color = PDG_COLOR_MAP.get(pdg_codes[i], 'black')
            ax.quiver(x, y, z, px, py, pz, length=1, normalize=True,
                      arrow_length_ratio=0.1, color=color)
            ax.scatter(x, y, z, color=color, marker='o', s=50)

        ax.set_xlabel('X [fm]')
        ax.set_ylabel('Y [fm]')
        ax.set_zlabel('Z [fm]')
        title = f"Particle Visualization - Time Step {timestep_list[step]}" if timestep_list else f"Step {step}"
        ax.set_title(f"{title} ({len(pdg_codes)} particles)")

        if xlim: ax.set_xlim(*xlim)
        if ylim: ax.set_ylim(*ylim)
        if zlim: ax.set_zlim(*zlim)

    def update(frame_idx):
        plot_frame(start_step-1 + frame_idx)
        return ax,

    ani = animation.FuncAnimation(
        fig, update, frames=(end_step+1 - start_step), blit=False
    )

    ani.save(filename, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"[Saved] {filename} - time step [{start_step}, {end_step}]")


def evaluate_model(model, test_loader, device="cuda"):
    """
    Evaluate model performance on test data.
    
    Returns:
    --------
    dict: Evaluation metrics including accuracy and feature MSE
    """
    model.to(device)
    model.eval()

    all_type_losses = []
    all_feat_losses = []
    all_total_losses = []
    all_preds = []
    all_targets = []
    all_feats_pred = []
    all_feats_target = []

    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch['encoded_ids'], batch['particle_feats'], batch['padding_mask'])

            # Loss calculation (matching training)
            particle_types = output['particle_types']  # (batch, seq_len, num_classes)
            particle_feats = output['particle_feats']  # (batch, seq_len, 7)
            targets_types = batch['encoded_ids']       # (batch, seq_len)
            targets_feats = batch['particle_feats']    # (batch, seq_len, 7)
            padding_mask = batch['padding_mask']       # (batch, seq_len)

            # Shift for next-token prediction
            particle_types_shifted = particle_types[:, :-1]
            particle_feats_shifted = particle_feats[:, :-1]
            targets_types_shifted = targets_types[:, 1:]
            targets_feats_shifted = targets_feats[:, 1:]
            padding_mask_shifted = padding_mask[:, 1:]

            # Flatten for loss
            pt_shape = particle_types_shifted.shape
            type_loss = F.cross_entropy(
                particle_types_shifted.reshape(-1, pt_shape[-1]),
                targets_types_shifted.reshape(-1),
                reduction='none'
            )

            feat_loss = F.mse_loss(
                particle_feats_shifted.reshape(-1, 7),
                targets_feats_shifted.reshape(-1, 7),
                reduction='none'
            ).mean(dim=1)

            # Mask out padding
            padding_mask_flat = padding_mask_shifted.reshape(-1)
            valid_mask = ~padding_mask_flat
            type_loss = type_loss * valid_mask.float()
            feat_loss = feat_loss * valid_mask.float()
            num_valid = valid_mask.sum()
            type_loss = type_loss.sum() / max(num_valid, 1)
            feat_loss = feat_loss.sum() / max(num_valid, 1)
            total_loss = type_loss + feat_loss

            all_type_losses.append(type_loss.item())
            all_feat_losses.append(feat_loss.item())
            all_total_losses.append(total_loss.item())

            # For accuracy and inspection
            preds = torch.argmax(particle_types_shifted, dim=-1)  # [batch, seq_len-1]
            all_preds.append(preds.cpu())
            all_targets.append(targets_types_shifted.cpu())
            all_feats_pred.append(particle_feats_shifted.cpu())
            all_feats_target.append(targets_feats_shifted.cpu())

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_feats_pred = torch.cat(all_feats_pred, dim=0)
    all_feats_target = torch.cat(all_feats_target, dim=0)
    
    # Denormalize feature variables
    feats_sigma = torch.tensor(FEATS_SIGMA, device=all_feats_pred.device)
    feats_mean = torch.tensor(FEATS_MEAN, device=all_feats_pred.device)
    all_feats_pred = (all_feats_pred * feats_sigma + feats_mean).float()
    all_feats_target = (all_feats_target * feats_sigma + feats_mean).float()

    # Compute accuracy, ignoring PAD tokens
    mask = all_targets != PAD_TOKEN
    accuracy = (all_preds[mask] == all_targets[mask]).float().mean().item()

    # Compute feature MSE
    mean_feat_loss = sum(all_feat_losses) / len(all_feat_losses)

    return {
        'accuracy': accuracy,
        'feature_mse': mean_feat_loss,
        'type_loss': sum(all_type_losses) / len(all_type_losses),
        'total_loss': sum(all_total_losses) / len(all_total_losses),
        'predictions': all_preds,
        'targets': all_targets,
        'feats_pred': all_feats_pred,
        'feats_target': all_feats_target
    }
