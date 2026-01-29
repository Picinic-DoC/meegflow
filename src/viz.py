import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

def _epoch_event_labels(epochs: mne.Epochs) -> np.ndarray:
    """Return a string label per epoch (event type)."""
    # epochs.events[:, 2] are the event codes
    inv = {v: k for k, v in epochs.event_id.items()}
    codes = epochs.events[:, 2]
    labels = np.array([inv.get(int(c), f"code_{int(c)}") for c in codes], dtype=object)
    return labels

def droplog_dataframe(epochs: mne.Epochs) -> pd.DataFrame:
    """
    One row per (epoch that was dropped, reason).
    Columns: epoch_ix, event_type, reason
    """
    event_type = _epoch_event_labels(epochs)

    rows = []
    for ei, reasons in enumerate(epochs.drop_log):
        # epochs.drop_log[ei] is a tuple/list of ’reasons’; empty means kept
        if not reasons:
            continue
        for r in reasons:
            rows.append({"epoch_ix": ei, "event_type": event_type[ei], "reason": r})
    return pd.DataFrame(rows)

def plot_drops_by_reason_and_type(epochs: mne.Epochs, title="Dropped epochs by reason and event type"):
    df = droplog_dataframe(epochs)

    fig, ax = plt.subplots(figsize=(8, 4))
    if df.empty:
        ax.text(0.5, 0.5, "No dropped epochs.", ha="center", va="center")
        ax.axis("off")
        return fig

    # counts(reason, event_type)
    tab = (
        df.groupby(["reason", "event_type"])
          .size()
          .unstack("event_type", fill_value=0)
          .sort_index()
    )

    tab.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Drop reason")
    ax.set_ylabel("# epochs dropped")
    ax.legend(title="Event type", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig
