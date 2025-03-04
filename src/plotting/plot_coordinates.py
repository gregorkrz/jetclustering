import plotly.express as px
import pandas as pd
import numpy as np
import os

def plot_coordinates(
    coords,
    pt, # the size of the points
    tidx, # truth idx
    outdir=None,
    filename=None
):
    data = {
        "X": coords[:, 0].view(-1, 1).detach().cpu().numpy(),
        "Y": coords[:, 1].view(-1, 1).detach().cpu().numpy(),
        "Z": coords[:, 2].view(-1, 1).detach().cpu().numpy(),
        "tIdx": tidx.view(-1, 1).detach().cpu().numpy(),
        "pt": pt.view(-1, 1).detach().cpu().numpy(),
    }
    print([(k, data[k].shape) for k in data])
    df = pd.DataFrame(
        np.concatenate([data[k] for k in sorted(data.keys())], axis=1),
        columns=[k for k in sorted(data.keys())],
    )
    df["orig_tIdx"] = df["tIdx"]
    fig = px.scatter_3d(
        df,
        x="X",
        y="Y",
        z="Z",
        color="tIdx",
        size="pt",
        # hover_data=hover_data,
        template="plotly_dark",
        color_continuous_scale=px.colors.sequential.Rainbow,
        # make it opaque a bit
        opacity=0.5,
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    if filename is None or outdir is None:
        return fig
    fig.write_html(
        os.path.join(outdir, filename)
    )

