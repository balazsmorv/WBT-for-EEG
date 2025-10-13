import os

import numpy as np
import matplotlib.pyplot as plt
import plotly
from sklearn.decomposition import PCA
import plotly.graph_objects as go  # Import plotly for interactive 3D plotting
import torch
import ot.plot
import mlflow

def visualize_barycenter_diracs(barycenter, num_images, random_seed=42):
    """
    Plots a specified number of images from the barycenter array after reshaping them into square images,
    displayed vertically with larger sizes.

    Args:
        barycenter (np.ndarray): The array containing images as rows, with shape (num_images, num_pixels).
        num_images (int): The number of images to plot from the barycenter array.
        random_seed (int, optional): Random seed for selecting images to plot. Default is 42.
    """
    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # Determine the total number of available images
    total_images = barycenter.shape[0]

    # Ensure num_images does not exceed the total number of images available
    num_images = min(num_images, total_images)

    # Randomly select num_images indices from the barycenter array
    selected_indices = np.random.choice(total_images, num_images, replace=False)

    # Create subplots for the selected images, arranged vertically
    fig, axarr = plt.subplots(num_images, 1, figsize=(8, 2 * num_images))  # Larger figure size

    # Plot each selected image
    ims = []
    for i, idx in enumerate(selected_indices):
        # Reshape the image vector into its correct shape
        image = barycenter[idx].reshape((22, 1251))
        ims.append(image)

        # Handle single or multiple axes correctly
        if num_images == 1:
            axarr.imshow(image, aspect="auto")
            axarr.axis("off")
        else:
            axarr[i].imshow(image, aspect="auto")
            axarr[i].axis("off")

    plt.tight_layout()

    return fig


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import torch

def plot_pca_for_arrays(arrays, n_components=2, labels=None, save=False):
    if n_components not in [2, 3]:
        raise ValueError("Only 2D and 3D PCA visualizations are supported (n_components must be 2 or 3).")

    transformed_arrays = []

    for arr in arrays:
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()  # Convert to NumPy if it's a torch tensor

        arr = np.array(arr)  # Ensure it's a NumPy array

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)  # Reshape 1D array into a column vector

        # Ensure we have enough dimensions for PCA
        if arr.shape[1] < n_components:
            raise ValueError(f"Array must have at least {n_components} features for PCA, but got {arr.shape[1]}.")

        pca = PCA(n_components=n_components)
        transformed_arrays.append(pca.fit_transform(arr))

    if n_components == 2:
        plt.figure(figsize=(8, 6))
        for i, transformed in enumerate(transformed_arrays):
            label = labels[i] if labels else f"Array {i + 1}"
            plt.scatter(transformed[:, 0], transformed[:, 1], label=label, alpha=0.6)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.title("PCA Projection (2D)")
        plt.show()

    elif n_components == 3:
        fig = go.Figure()
        for i, transformed in enumerate(transformed_arrays):
            label = labels[i] if labels is not None else f"Array {i + 1}"
            fig.add_trace(go.Scatter3d(
                x=transformed[:, 0], y=transformed[:, 1], z=transformed[:, 2],
                mode='markers', marker=dict(size=5, opacity=0.6),
                name=label
            ))
        fig.update_layout(
            title="PCA Projection (3D)",
            scene=dict(
                xaxis_title="PC 1",
                yaxis_title="PC 2",
                zaxis_title="PC 3"
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.show()

    if save:
        plotly.io.write_image(fig, 'barycenter.jpg', scale=2, width=1063, height=1063)


from sklearn.decomposition import PCA
import torch

def plot_pca_for_arrays2(arrays, n_components=2, labels=None, save=False):
    if n_components not in [2, 3]:
        raise ValueError("Only 2D and 3D PCA visualizations are supported.")

    transformed_arrays = []
    flattened_labels = []

    for i, arr in enumerate(arrays):
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()  # Convert PyTorch tensor to NumPy

        arr = np.array(arr)  # Ensure it's a NumPy array

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)  # Reshape 1D arrays into column vectors

        if arr.shape[1] < n_components:
            raise ValueError(f"Array must have at least {n_components} features, but got {arr.shape[1]}.")

        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(arr)
        transformed_arrays.append(transformed)

        # Store per-point labels
        if labels:
            if len(labels[i]) != arr.shape[0]:
                raise ValueError(f"labels[{i}] should have {arr.shape[0]} elements, but got {len(labels[i])}.")
            flattened_labels.append(labels[i])

    # 2D Plot
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        for i, transformed in enumerate(transformed_arrays):
            plt.scatter(transformed[:, 0], transformed[:, 1], alpha=0.6, label=f"Set {i}")
            if labels:
                for j in range(len(transformed)):
                    plt.annotate(flattened_labels[i][j], (transformed[j, 0], transformed[j, 1]), fontsize=8, alpha=0.7)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.title("PCA Projection (2D)")
        plt.show()

    # 3D Plot
    elif n_components == 3:
        fig = go.Figure()
        for i, transformed in enumerate(transformed_arrays):
            fig.add_trace(go.Scatter3d(
                x=transformed[:, 0], y=transformed[:, 1], z=transformed[:, 2],
                mode='markers',
                marker=dict(size=5, opacity=0.6),
                text=flattened_labels[i] if labels else None,  # Use per-point labels
                hoverinfo="text" if labels else "none",
                name=f"Set {i}"  # Set name properly
            ))

        fig.update_layout(
            title="PCA Projection (3D)",
            scene=dict(
                xaxis_title="PC 1",
                yaxis_title="PC 2",
                zaxis_title="PC 3"
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.show()

    if save:
        plotly.io.write_image(fig, 'barycenter.jpg', scale=2, width=1063, height=1063)

def plot_pca_for_arrays3(arrays, n_components=2, labels=None):
    if n_components not in [2, 3]:
        raise ValueError("Only 2D and 3D PCA visualizations are supported.")

    transformed_arrays = []
    flattened_labels = []

    for i, arr in enumerate(arrays):
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()  # Convert PyTorch tensor to NumPy

        arr = np.array(arr)  # Ensure it's a NumPy array

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)  # Reshape 1D arrays into column vectors

        if arr.shape[1] < n_components:
            raise ValueError(f"Array must have at least {n_components} features, but got {arr.shape[1]}.")

        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(arr)
        transformed_arrays.append(transformed)

        # Store per-point labels
        if labels:
            if len(labels[i]) != arr.shape[0]:
                raise ValueError(f"labels[{i}] should have {arr.shape[0]} elements, but got {len(labels[i])}.")
            flattened_labels.append(labels[i])

    fig = None

    # 2D Plot
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, transformed in enumerate(transformed_arrays):
            ax.scatter(transformed[:, 0], transformed[:, 1], alpha=0.6, label=f"Set {i}")
            if labels:
                for j in range(len(transformed)):
                    ax.annotate(flattened_labels[i][j], (transformed[j, 0], transformed[j, 1]), fontsize=8, alpha=0.7)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend()
        ax.set_title("PCA Projection (2D)")

    # 3D Plot
    elif n_components == 3:
        fig = go.Figure()
        for i, transformed in enumerate(transformed_arrays):
            fig.add_trace(go.Scatter3d(
                x=transformed[:, 0], y=transformed[:, 1], z=transformed[:, 2],
                mode='markers',
                marker=dict(size=5, opacity=0.6),
                text=flattened_labels[i] if labels else None,  # Use per-point labels
                hoverinfo="text" if labels else "none",
                name=f"Set {i}"
            ))

        fig.update_layout(
            title="PCA Projection (3D)",
            scene=dict(
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2",
                zaxis_title="Principal Component 3"
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

    return fig  # Return the figure instead of showing it


def plot_pca_comparisons(X1, y1, X2, y2, X3, y3, X4, y4, subject, use_mlflow=False):

    pca = PCA(n_components=2)
    X1_pca = pca.fit_transform(X1)
    X2_pca = pca.fit_transform(X2)
    X3_pca = pca.fit_transform(X3)
    X4_pca = pca.fit_transform(X4)

    def make_plot(base_X, base_y, overlay_X, overlay_y, filename):
        plt.figure(figsize=(6, 5))
        plt.xticks([])
        plt.yticks([])

        # Build legend manually using class labels
        from matplotlib.lines import Line2D
        classes = np.unique(np.concatenate([base_y, overlay_y]))
        colors = plt.cm.tab20b(np.linspace(0, 1, len(classes)))
        label_map = {0: "Left hand", 1: "Right hand", 2: "Feet", 3: "Tongue"}  # example mapping
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=label_map.get(c, f"Class {c}"),
                   markerfacecolor=colors[i], markersize=8, markeredgecolor='k')
            for i, c in enumerate(classes)
        ]
        plt.legend(handles=legend_elements, title="Classes", loc='upper right', fontsize=8, title_fontsize=9)

        plt.scatter(base_X[:, 0], base_X[:, 1], c=base_y, cmap='tab20b', alpha=0.10, edgecolor='k', marker="o")
        plt.scatter(overlay_X[:, 0], overlay_X[:, 1], c=overlay_y, cmap='tab20b', alpha=1.0, edgecolor='k', marker="+")
        import re
        match = re.search(r'\d+', str(subject))
        subj_num = int(match.group()) if match else subject
        plt.title(f"Subject {subj_num}")
        plt.tight_layout()
        if use_mlflow:
            plt.savefig(filename, dpi=500)
            mlflow.log_artifact(filename)
            os.remove(filename)
        else:
            plt.savefig(filename, dpi=300)
        plt.close()

    make_plot(X1_pca, y1, X2_pca, y2, f'{subject}-original+bary.jpg')
    make_plot(X2_pca, y2, X3_pca, y3, f'{subject}-bary+transported.jpg')
    make_plot(X3_pca, y3, X4_pca, y4, f'{subject}-transported+target.jpg')