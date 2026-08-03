import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory
import plotly.graph_objs as go

_CLASS_LABELS = {False: "No", True: "Yes"}
_COLORS = {"No": "blue", "Yes": "red"}


def descriptive_stats(scored: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation, standard error, and group size of scores for each group."""
    scored = scored.dropna(subset=["score", "terminated"]).assign(
        terminated=lambda _df: _df.terminated.map(_CLASS_LABELS)
    )
    stats = scored.groupby("terminated")["score"].agg(mean="mean", standard_deviation="std", group_size="size")
    stats["standard_error"] = stats["standard_deviation"] / np.sqrt(stats["group_size"])
    return stats


def termination_probability_histogram(
    scored: pd.DataFrame, box_plot: bool = False, histnorm: str | None = None
) -> go.Figure:
    """Histogram of risk scores for employees who left vs. stayed.

    Set box_plot=True to add a marginal box plot above the histogram.
    """
    scored = scored.dropna(subset=["score", "terminated"])
    figure = px.histogram(
        scored.assign(terminated=lambda _df: _df.terminated.map(_CLASS_LABELS)),
        x="score",
        color="terminated",
        histnorm=histnorm,
        barmode="overlay",
        opacity=0.8,
        marginal="box" if box_plot else None,
        color_discrete_map=_COLORS,
        title="Estimated Probability of Termination<br>between Terminated and Retained Employees",
        labels={"score": "Probability of Termination", "terminated": "Left the company?"},
    )
    figure.update_layout(
        yaxis_title="Employees",
        legend_title_text="Terminated",
        font_size=16,
    )

    if histnorm == "probability density":
        _groups = [scored.loc[scored.terminated == _class_value, "score"] for _class_value in _CLASS_LABELS]
        _kde = plotly.figure_factory.create_distplot(
            _groups,
            group_labels=list(_CLASS_LABELS.values()),
            colors=[_COLORS[_label] for _label in _CLASS_LABELS.values()],
            show_hist=False,
            show_rug=False,
        )
        for _curve in _kde.data:
            _curve.showlegend = False
            figure.add_trace(_curve)
    return figure
