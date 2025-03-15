import os
import pandas as pd
import numpy as np
import holoviews as hv
import hvplot.pandas
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from bokeh.palettes import Category10  # Color palette for automatic assignment

hv.extension('bokeh')
import panel as pn

# Color cycle for automatic color assignment
COLOR_CYCLE = Category10[10]  # Up to 10 different colors
COLOR_IDX = 0  # Global index to track assigned colors
PROJ_DIR = "/home/shefi/Projects/samueli_ds"


# Get Files absolute Path
def get_ckpt_path(model_ver):
    return f"{PROJ_DIR}/data/outputs/{model_ver['version']}/{model_ver['ckpt']}"

def get_results_df_path(model_ver):
    dirname = f"{PROJ_DIR}/data/outputs/{model_ver['version']}/results"
    fname = f"preds_{model_ver.get('dataset')}_df_{model_ver['ckpt'][:-5]}.pickle"
    return f"{dirname}/{fname}"


# ========================================
# Panel util - hopefully import will work
# ========================================
def panel_dashboard(dashboard):
    dashboard.show()  # Do I need return?


# ============
# Eval Methods
# ============

def construct_results_df(preds, results_path=None):
    results = []
    for batch_idx, pred_batch in enumerate(preds):
        for sample_pred, sample_data in zip(*pred_batch):
            y_hat = sample_pred.cpu().detach().numpy()[0]
            results.append(
                dict(slide_num=sample_data["slide_num"], tile_num=sample_data["tile_num"],
                    tissue_percent=sample_data["tissue_percent"], tile_score=sample_data["score"],
                    y=sample_data["label"], y_hat=y_hat, err=sample_data["label"]-y_hat)
                )

    results_df = pd.DataFrame.from_records(results)
    # save to file
    if results_path is not None:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results_df.to_pickle(results_path)
    else:
        print("No results_path - skipping save df")
    return results_df


def plot_metrics(results_df, label, color=None):
    global COLOR_IDX  # Use global color index to track colors
    if color is None:
        color = COLOR_CYCLE[COLOR_IDX % len(COLOR_CYCLE)]  # Assign color from palette
        COLOR_IDX += 1  # Move to next color for next model
    
    y_true = results_df['y'].values
    y_pred = results_df['y_hat'].values

    # 1️⃣ ROC Curve
    fpr, tpr, scores = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    line_kws = dict(width=400, height=300, line_width=2)
    hv_kws = dict(grid=True, color=color, hover_cols='all')
    
    roc_df = pd.DataFrame(dict(FPR=fpr, TPR=tpr, thr=scores, label=label))
    roc_curve_plot = roc_df.hvplot.line(
        x='FPR', y='TPR', title='ROC Curve', label=f"{label} (AUC={roc_auc:.2f})", **line_kws, **hv_kws,
    )

    # 2️⃣ Precision-Recall Curve
    precision, recall, scores = precision_recall_curve(y_true, y_pred)
    pr_df = pd.DataFrame(dict(Recall=recall, Precision=precision, thr=np.append(scores,[1.0]), label=label))

    pr_curve_plot = pr_df.hvplot.line(
        x='Recall', y='Precision',title='Precision-Recall Curve', **line_kws,  **hv_kws,
    )

    # 3️⃣ Confusion Matrix
    y_pred_labels = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels
    cm = confusion_matrix(y_true, y_pred_labels)  # Consider using normalize
    
    # cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])

    # cm_text = cm_df.applymap(lambda x: f"{x:.1f}%")  # Format as percentages
    
    confusion_matrix_plot = cm_df.hvplot.heatmap(
        cmap='Blues', colorbar=True, title=f"Confusion Matrix ({label})", width=400, height=300
    ).opts(xlabel="Predicted", ylabel="Actual", clim=(0, cm.max()))

    return roc_curve_plot, pr_curve_plot, confusion_matrix_plot

def plot_slide_metrics(results_df, thr=0.5):
    results_df = results_df.assign(y_pred=lambda x: (x.y_hat > thr).astype(int))
    slide_restuls_df = (
        results_df
        .groupby("slide_num")
        .agg(slide_gt=("y", "first"), total_tiles=("tile_num", "count"),
             tiles_votes=("y_pred", "sum"), y_hat_mean=("y_hat", "mean"))
        .assign(mean_votes=lambda x: x.tiles_votes/x.total_tiles)
    )

    # analyze tiles score wrt to tissue_count
    pnts_kws = dict(width=400, height=300,size=3)
    pred2tissue = results_df.hvplot.points(
        x="tissue_percent", y="y_hat", grid=True, **pnts_kws, title="Prediction to tissue percentage"
    )
    pred2tl_sc = results_df.hvplot.points(
        x="tile_score", y="y_hat", grid=True, **pnts_kws, title="Prediction to tile color score"
    )

    vw = pn.Column(
        pn.pane.DataFrame(slide_restuls_df),
        pn.Row(pred2tissue, pred2tl_sc)
    )
    return vw



