"""
Visualization Module using Matplotlib.
Converts all plots to high-resolution base64 PNG data URLs for seamless React rendering.
Aesthetics tailored with sleek modern styling (dark backgrounds, glowing palette, clean typography).
"""

import io
import base64
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Modern theme styling parameters
plt.style.use("dark_background")
PALETTE = ["#38bdf8", "#818cf8", "#c084fc", "#34d399", "#f472b6", "#fbbf24", "#f87171"]
BG_COLOR = "#0f172a"
CARD_BG = "#1e293b"
GRID_COLOR = "#334155"
TEXT_COLOR = "#f8fafc"


def _fig_to_base64(fig: plt.Figure) -> str:
    """Encodes a Matplotlib figure into a base64 string."""
    buf = io.BytesIO()
    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64_data = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64_data}"


def plot_target_distribution(df: pd.DataFrame, target_column: str, problem_type: str) -> Optional[str]:
    """Generates distribution plot for the target variable."""
    if target_column not in df.columns:
        return None
    
    series = df[target_column].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor(CARD_BG)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)

    if problem_type == "regression" or (pd.api.types.is_numeric_dtype(series) and series.nunique() > 20):
        # Histogram with KDE
        n, bins, patches = ax.hist(series, bins=30, color="#38bdf8", edgecolor="#0284c7", alpha=0.85)
        ax.set_title(f"Target Distribution: {target_column} (Continuous)", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(target_column, color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel("Frequency", color=TEXT_COLOR, fontsize=10)
    else:
        # Bar chart for classes
        counts = series.value_counts()
        bars = ax.bar([str(k) for k in counts.index], counts.values, color=PALETTE[:len(counts)], edgecolor="#1e1e2f", width=0.55)
        ax.set_title(f"Class Distribution: {target_column}", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Class", color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel("Count", color=TEXT_COLOR, fontsize=10)
        # Value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        color=TEXT_COLOR, fontsize=9, fontweight="bold")

    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        
    return _fig_to_base64(fig)


def plot_missing_values(df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> Optional[str]:
    """Generates a bar plot showing missing value counts and percentages."""
    if feature_columns:
        cols = [c for c in feature_columns if c in df.columns]
        sub_df = df[cols]
    else:
        sub_df = df
        
    missing_counts = sub_df.isna().sum()
    missing_cols = missing_counts[missing_counts > 0].sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.set_facecolor(CARD_BG)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)

    if missing_cols.empty:
        ax.text(0.5, 0.5, "100% Complete Data\n(No Missing Values Detected)", ha="center", va="center",
                color="#34d399", fontsize=13, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Missing Values Audit", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=12)
    else:
        missing_pct = (missing_cols / len(df)) * 100
        bars = ax.barh(missing_cols.index[:10], missing_pct.values[:10], color="#f87171", edgecolor="#dc2626", height=0.55)
        ax.set_title("Top Missing Feature Ratios (%)", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Missing Percentage (%)", color=TEXT_COLOR, fontsize=10)
        ax.set_xlim(0, max(100, float(missing_pct.max() * 1.15)))
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f"{width:.1f}%", xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                        color=TEXT_COLOR, fontsize=9, fontweight="bold")

    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        
    return _fig_to_base64(fig)


def plot_correlation_matrix(df: pd.DataFrame, numeric_features: List[str]) -> Optional[str]:
    """Generates an annotated correlation heatmap for numeric features."""
    cols = [c for c in numeric_features if c in df.columns]
    if len(cols) < 2:
        return None
    
    # Cap at top 10 features for readability
    cols = cols[:10]
    corr = df[cols].corr().fillna(0.0).values
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.set_facecolor(CARD_BG)
    
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", color=TEXT_COLOR, fontsize=8)
    ax.set_yticklabels(cols, color=TEXT_COLOR, fontsize=8)
    ax.set_title("Feature Correlation Heatmap", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=12)
    
    # Text annotations
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr[i, j]
            text_color = "black" if abs(val) < 0.6 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8, fontweight="bold")
            
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        
    return _fig_to_base64(fig)


def plot_model_comparison(evaluation_results: List[Dict[str, Any]], primary_metric: str) -> Optional[str]:
    """Generates bar chart comparing validation metric across candidate models."""
    if not evaluation_results:
        return None
        
    models = [r["model_name"] for r in evaluation_results]
    scores = [r["metrics"].get(primary_metric, 0.0) for r in evaluation_results]
    train_times = [r.get("train_time_sec", 0.0) for r in evaluation_results]
    
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.set_facecolor(CARD_BG)
    ax1.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)
    
    x = np.arange(len(models))
    width = 0.45
    
    bars = ax1.bar(x, scores, width, color="#38bdf8", edgecolor="#0284c7", label=f"Validation {primary_metric.upper()}")
    ax1.set_ylabel(f"Validation Score ({primary_metric})", color="#38bdf8", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=25, ha="right", color=TEXT_COLOR, fontsize=9)
    ax1.tick_params(colors=TEXT_COLOR)
    
    # Value annotations on bars
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     color=TEXT_COLOR, fontsize=9, fontweight="bold")
                     
    ax1.set_title(f"Candidate Model Validation Comparison ({primary_metric.upper()})", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=12)
    for spine in ax1.spines.values():
        spine.set_color(GRID_COLOR)
        
    return _fig_to_base64(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> str:
    """Generates an annotated Confusion Matrix heatmap for classification."""
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.set_facecolor(CARD_BG)
    
    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, color=TEXT_COLOR, fontsize=9)
    ax.set_yticklabels(class_names, color=TEXT_COLOR, fontsize=9)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(j, i, format(val, "d"), ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=11, fontweight="bold")
                    
    ax.set_ylabel("True Label", color=TEXT_COLOR, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label", color=TEXT_COLOR, fontsize=10, fontweight="bold")
    ax.set_title("Test Confusion Matrix", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        
    return _fig_to_base64(fig)


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> str:
    """Generates ROC Curve with AUC annotation."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.set_facecolor(CARD_BG)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)
    
    ax.plot(fpr, tpr, color="#34d399", lw=2.5, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Chance")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("True Positive Rate", color=TEXT_COLOR, fontsize=10)
    ax.set_title("Receiver Operating Characteristic (ROC)", color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        
    return _fig_to_base64(fig)


def plot_regression_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Generates Predicted vs Actual and Residual plots for regression."""
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))
    
    # Plot 1: Predicted vs Actual
    ax1.set_facecolor(CARD_BG)
    ax1.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.scatter(y_true, y_pred, color="#38bdf8", alpha=0.7, edgecolors="none", s=30)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], color="#f472b6", lw=2, linestyle="--", label="Identity (y=x)")
    ax1.set_xlabel("Actual Values", color=TEXT_COLOR, fontsize=9)
    ax1.set_ylabel("Predicted Values", color=TEXT_COLOR, fontsize=9)
    ax1.set_title("Predicted vs Actual", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax1.tick_params(colors=TEXT_COLOR)
    ax1.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    for spine in ax1.spines.values():
        spine.set_color(GRID_COLOR)

    # Plot 2: Residuals vs Predicted
    ax2.set_facecolor(CARD_BG)
    ax2.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.scatter(y_pred, residuals, color="#c084fc", alpha=0.7, edgecolors="none", s=30)
    ax2.axhline(0, color="#f472b6", lw=1.5, linestyle="--")
    ax2.set_xlabel("Predicted Values", color=TEXT_COLOR, fontsize=9)
    ax2.set_ylabel("Residuals (Actual - Pred)", color=TEXT_COLOR, fontsize=9)
    ax2.set_title("Residual Distribution", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax2.tick_params(colors=TEXT_COLOR)
    for spine in ax2.spines.values():
        spine.set_color(GRID_COLOR)
        
    return _fig_to_base64(fig)
