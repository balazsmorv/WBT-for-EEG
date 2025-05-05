import os
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import torch

def plot_pca_for_arrays(arrays, n_components=2, labels=None):
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

    return fig

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
        label_map = {0: "Non-target", 1: "Target", 2: "Feet", 3: "Tongue"}  # example mapping
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=label_map.get(c, f"Class {c}"),
                   markerfacecolor=colors[i], markersize=8, markeredgecolor='k')
            for i, c in enumerate(classes)
        ]
        plt.legend(handles=legend_elements, title="Classes", loc='upper right', fontsize=8, title_fontsize=9)

        plt.scatter(base_X[:, 0], base_X[:, 1], c=base_y, cmap='tab20b', alpha=0.10, edgecolor='k', marker="o")
        plt.scatter(overlay_X[:, 0], overlay_X[:, 1], c=overlay_y, cmap='tab20b', alpha=1.0, edgecolor='k', marker="+")
        plt.title(f"Subject {subject}") # get only number
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