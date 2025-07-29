import os
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.keras.utils import plot_model


log_level = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=getattr(
        logging, log_level, logging.INFO
    ),  # Fallback to INFO if the level is invalid
    format="%(asctime)s - %(name)-20s - %(levelname) -8s - %(message)s",
)

logger = logging.getLogger(__name__)


def histogram_dataset(
    dfall, target, weights, columns=None, nbin=25, save_path="histogram.png"
):
    """
    Plots histograms of the dataset features.

    Args:
        * columns (list): The list of column names to consider (default: None, which includes all columns).
        * nbin (int): The number of bins for the histogram (default: 25).

    .. Image:: images/histogram_datasets.png
    """

    if columns is None:
        columns = columns
    else:
        for col in columns:
            if col not in columns:
                logger.warning(f"Column {col} not found in dataset. Skipping.")
                columns.remove(col)
    if len(columns) == 0:
        raise ValueError("No valid columns provided for histogram plotting.")

    sns.set_theme(style="whitegrid")

    df = pd.DataFrame(dfall, columns=columns)

    # Number of rows and columns in the subplot grid
    n_cols = 2  # Number of columns in the subplot grid
    n_rows = int(np.ceil(len(columns) / n_cols))  # Calculate the number of rows needed

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 6 * n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy indexing

    for i, column in enumerate(columns):
        # Determine the combined range for the current column

        print(f"[*] --- {column} histogram")

        lower_percentile = 0
        upper_percentile = 97.5

        lower_bound = np.percentile(df[column], lower_percentile)
        upper_bound = np.percentile(df[column], upper_percentile)

        df_clipped = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        weights_clipped = weights[
            (df[column] >= lower_bound) & (df[column] <= upper_bound)
        ]
        target_clipped = target[
            (df[column] >= lower_bound) & (df[column] <= upper_bound)
        ]

        min_value = df_clipped[column].min()
        max_value = df_clipped[column].max()

        # Define the bin edges
        bin_edges = np.linspace(min_value, max_value, nbin + 1)

        signal_field = df_clipped[target_clipped == 1][column]
        background_field = df_clipped[target_clipped == 0][column]
        signal_weights = weights_clipped[target_clipped == 1]
        background_weights = weights_clipped[target_clipped == 0]

        # Plot the histogram for label == 1 (Signal)
        axes[i].hist(
            signal_field,
            bins=bin_edges,
            alpha=0.4,
            color="blue",
            label="Signal",
            weights=signal_weights,
            density=True,
        )

        axes[i].hist(
            background_field,
            bins=bin_edges,
            alpha=0.4,
            color="red",
            label="Background",
            weights=background_weights,
            density=True,
        )

        # Set titles and labels
        axes[i].set_title(f"{column}", fontsize=16)
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Density")

        # Add a legend to each subplot
        axes[i].legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    fig.savefig(save_path)
    plt.show()
    plt.close(fig)
    return save_path


def roc_curve_wrapper(
    score,
    labels,
    weights,
    plot_label="model",
    color="b",
    lw=2,
    save_path="roc_curve.png",
):
    """
    Plots the ROC curve.

    Args:
        * score (ndarray): The score.
        * labels (ndarray): The labels.
        * weights (ndarray): The weights.
        * plot_label (str, optional): The plot label. Defaults to "model".
        * color (str, optional): The color. Defaults to "b".
        * lw (int, optional): The line width. Defaults to 2.

    .. Image:: images/roc_curve.png
    """

    auc = roc_auc_score(y_true=labels, y_score=score, sample_weight=weights)

    fig = plt.figure(figsize=(8, 7))

    fpr, tpr, _ = roc_curve(y_true=labels, y_score=score, sample_weight=weights)
    plt.plot(fpr, tpr, color=color, lw=lw, label=plot_label + " AUC :" + f"{auc:.3f}")

    plt.plot([0, 1], [0, 1], color="k", lw=lw, linestyle="--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.show()
    plt.close(fig)
    return save_path


def visualize_model_architecture(model, filename="nn_architecture.png"):
    """
    Saves a visual diagram of a Keras model architecture to a file.

    Args:
        model (tf.keras.Model): The compiled Keras model.
        filename (str): Path to save the image (e.g., 'model.png').
    """
    try:
        plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)
        print(f"[✔] Saved model visualization to: {filename}")
    except ImportError as e:
        print(
            "[✘] Error: Missing 'pydot' or 'graphviz'. Please install them to visualize the model."
        )
        print(str(e))


def stacked_histogram(
    dfall,
    target,
    weights,
    detailed_label,
    field_name,
    mu_hat=1.0,
    nbins=30,
    y_scale="linear",
    save_path="stacked_histogram.png",
):
    """
    Plots a stacked histogram of a specific field in the dataset.

    Args:
        * dfall : Pandas Dataframe
        * target : numpy array with labels
        * weights : numpy array with event weights
        * weights : numpy array with detailed labels of the events
        * detailed_label : The name of the field to plot.
        * mu_hat : The value of mu (default: 1.0).
        * bins (int): The number of bins for the histogram (default: 30).

    .. Image:: images/stacked_histogram.png
    """
    field = dfall[field_name]

    weight_keys = {}
    keys = np.unique(detailed_label)

    for key in keys:
        weight_keys[key] = weights[detailed_label == key]

    print("keys", keys)
    print("keys 2", weight_keys.keys())

    sns.set_theme(rc={"figure.figsize": (8, 7)}, style="whitegrid")

    lower_percentile = 0
    upper_percentile = 97.5

    lower_bound = np.percentile(field, lower_percentile)
    upper_bound = np.percentile(field, upper_percentile)

    field_clipped = field[(field >= lower_bound) & (field <= upper_bound)]
    weights_clipped = weights[(field >= lower_bound) & (field <= upper_bound)]
    target_clipped = target[(field >= lower_bound) & (field <= upper_bound)]
    detailed_labels_clipped = detailed_label[
        (field >= lower_bound) & (field <= upper_bound)
    ]

    min_value = field_clipped.min()
    max_value = field_clipped.max()

    # Define the bin edges
    bins = np.linspace(min_value, max_value, nbins + 1)

    hist_s, bins = np.histogram(
        field_clipped[target_clipped == 1],
        bins=bins,
        weights=weights_clipped[target_clipped == 1],
    )

    hist_b, bins = np.histogram(
        field_clipped[target_clipped == 0],
        bins=bins,
        weights=weights_clipped[target_clipped == 0],
    )

    hist_bkg = hist_b.copy()

    higgs = "htautau"

    for key in keys:
        if key != higgs:
            hist, bins = np.histogram(
                field_clipped[detailed_labels_clipped == key],
                bins=bins,
                weights=weights_clipped[detailed_labels_clipped == key],
            )
            plt.stairs(hist_b, bins, fill=True, label=f"{key} bkg")
            hist_b -= hist
        else:
            print(key, hist_s.shape)

    plt.stairs(
        hist_s * mu_hat + hist_bkg,
        bins,
        fill=False,
        color="orange",
        label=f"$H \\rightarrow \\tau \\tau (\\mu = {mu_hat:.3f})$",
    )

    plt.stairs(
        hist_s + hist_bkg,
        bins,
        fill=False,
        color="red",
        label=f"$H \\rightarrow \\tau \\tau (\\mu = {1.0:.3f})$",
    )

    plt.legend()
    plt.title(f"Stacked histogram of {field_name}")
    plt.xlabel(f"{field_name}")
    plt.ylabel("Weighted count")
    plt.yscale(y_scale)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()
    return save_path


import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep  # Assuming you're using ATLAS/MPLHEP style
import os


def abline(slope, intercept, **kwargs):
    """Plot a line y = slope * x + intercept across the current plot."""
    ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, **kwargs)


def fill_histograms_wError(data, weights, edges, histrange, epsilon=1.0e-20):
    """
    Compute weighted histogram and its statistical error.

    Args:
        data (np.ndarray): Data points to histogram.
        weights (np.ndarray): Weights for each data point.
        edges (np.ndarray): Bin edges.
        histrange (tuple): Range of histogram (min, max).
        epsilon (float): Small number to avoid division by zero.

    Returns:
        hist (np.ndarray): Weighted histogram counts.
        hist_err (np.ndarray): Statistical errors per bin.
    """

    # Weighted histogram
    hist = np.histogram(data, bins=edges, range=histrange, weights=weights)[0]

    # For the error, assuming weights are event weights, the variance per bin is sum of squared weights
    # Compute sum of squared weights in each bin:
    squared_weights = weights**2
    hist_err = np.sqrt(
        np.histogram(data, bins=edges, range=histrange, weights=squared_weights)[0]
    )

    # Avoid zero errors by adding epsilon
    hist_err = np.maximum(hist_err, epsilon)

    return hist, hist_err


def plot_calibration_curve(
    data_den,
    weight_den,
    data_num,
    weight_num,
    data_denH,
    weight_denH,
    data_numH,
    weight_numH,
    path_to_figures="",
    nbins=100,
    epsilon=1.0e-20,
    label="Calibration Curve",
    score_range="standard",
    save="",
):
    # Prepare save path
    save_path = save

    data = np.concatenate([data_num, data_den, data_denH, data_numH]).flatten()
    xmin, xmax = np.amin(data), np.amax(data)
    edges = np.linspace(xmin, xmax, nbins + 1)
    histrange = (xmin, xmax)

    # Compute histograms
    h_S, h_S_err = fill_histograms_wError(
        data_den, weight_den, edges, histrange, epsilon
    )
    h_X, h_X_err = fill_histograms_wError(
        data_num, weight_num, edges, histrange, epsilon
    )
    h_sum = h_S + h_X
    h_ratio = np.divide(h_X, h_sum + epsilon)  # avoids divide by zero
    err = np.sqrt((h_X_err / (h_X + epsilon)) ** 2 + (h_S_err / (h_S + epsilon)) ** 2)

    h_SH, h_SH_err = fill_histograms_wError(
        data_denH, weight_denH, edges, histrange, epsilon
    )
    h_XH, h_XH_err = fill_histograms_wError(
        data_numH, weight_numH, edges, histrange, epsilon
    )
    h_sumH = h_SH + h_XH
    h_ratioH = np.divide(h_XH, h_sumH + epsilon)
    errH = np.sqrt(
        (h_XH_err / (h_XH + epsilon)) ** 2 + (h_SH_err / (h_SH + epsilon)) ** 2
    )

    # Plotting
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    ax1 = fig.add_subplot(2, 1, 1)

    bin_centers = (edges[:-1] + edges[1:]) / 2
    chi_2 = np.sum(((h_ratio - bin_centers) ** 2) / (err**2 + epsilon)) / (nbins - 1)

    chi_H = np.sum(((h_ratioH - bin_centers) ** 2) / (errH**2 + epsilon)) / (nbins - 1)
    hep.histplot(
        [h_ratio, h_ratioH], bins=edges, yerr=[err, errH], label=["Training", "Holdout"]
    )
    abline(1.0, 0.0)
    plt.title(label, fontsize=12)
    plt.text(x=0.2, y=1.0, s=r"${\chi}^2_{\rm Train}/n_{\rm dof}$ = " + f"{chi_2:.3f}")
    plt.text(
        x=0.2, y=0.9, s=r"${\chi}^2_{\rm Holdout}/n_{\rm dof}$ = " + f"{chi_H:.3f}"
    )
    plt.axis(xmin=xmin - 0.1, xmax=xmax + 0.1, ymax=1.1, ymin=-0.1)
    plt.ylabel("Probability ratio", size=12)
    plt.legend(loc="lower right")

    # Residuals
    ax2 = fig.add_subplot(2, 1, 2)
    slopeOne = (edges[:-1] + edges[1:]) / 2
    residue = np.divide(h_ratio - slopeOne, err + epsilon)
    residueH = np.divide(h_ratioH - slopeOne, errH + epsilon)

    plt.errorbar(slopeOne, residue, yerr=1.0, fmt="o", label="Train Residuals")
    plt.errorbar(slopeOne, residueH, yerr=1.0, fmt="o", label="Holdout Residuals")
    plt.xlabel("Predicted Score", size=12)
    plt.ylabel("Residual", size=12)
    plt.axis(xmin=xmin - 0.1, xmax=xmax + 0.1, ymin=-4.0, ymax=4.0)
    abline(0.0, 0.0)
    plt.legend()

    # Save figure
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return save_path


def plot_score_distributions(
    training_set, holdout_set, preselection, columns, models, plots_dir, NP
):
    """
    Plots the score distributions for the nominal, plus, and minus models.

    Args:
        training_set (dict): The training set (should include 'data', 'labels', 'weights')
        holdout_set (dict): The holdout set (should include 'data', 'labels', 'weights')

    Saves:
        A matplotlib figure comparing the score distributions.
    """

    # Apply preselection and extract the required data
    training_set = preselection.apply_pre_selection(training_set, threshold=0.8)
    holdout_set = preselection.apply_pre_selection(holdout_set, threshold=0.8)

    X_train = training_set["data"][columns]
    X_holdout = holdout_set["data"][columns]

    # Predict scores for each systematic model
    scores = {}
    for key in models:
        scores[key] = {
            "train": models[key].predict(X_train),
            "holdout": models[key].predict(X_holdout),
        }

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    colors = {"plus": "red", "minus": "blue", "nominal": "black"}

    for key, color in colors.items():
        if key == "nominal":
            # Estimate nominal as midpoint if not explicitly trained
            scores["nominal"] = {
                "train": 0.5 * (scores["plus"]["train"] + scores["minus"]["train"]),
                "holdout": 0.5
                * (scores["plus"]["holdout"] + scores["minus"]["holdout"]),
            }

        axs[0].hist(
            scores[key]["train"],
            bins=50,
            alpha=0.6,
            label=key,
            color=color,
            density=True,
        )

        axs[1].hist(
            scores[key]["holdout"],
            bins=50,
            alpha=0.6,
            label=key,
            color=color,
            density=True,
        )

    axs[0].set_title("Score Distribution - Training Set")
    axs[1].set_title("Score Distribution - Holdout Set")

    for ax in axs:
        ax.set_xlabel("Model Score")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()

    save_path = os.path.join(plots_dir, f"{NP}_score_comparison.png")
    fig.savefig(save_path)
    plt.show()
    plt.close(fig)
    logger.info(f"Score distribution plot saved at {save_path}")
    return save_path
