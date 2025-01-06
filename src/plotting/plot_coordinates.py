import plotly.express as px
import pandas as pd
import numpy as np
import os

def plot_coordinates(
    coords,
    pt, # the size of the points
    tidx, # truth idx
    outdir,
    filename
):
    print("coords", coords.shape, "pt", pt.shape, "tidx", tidx.shape)
    data = {
        "X": coords[:, 0].view(-1, 1).detach().cpu().numpy(),
        "Y": coords[:, 1].view(-1, 1).detach().cpu().numpy(),
        "Z": coords[:, 2].view(-1, 1).detach().cpu().numpy(),
        "tIdx": tidx.view(-1, 1).detach().cpu().numpy(),
        "pt": pt.view(-1, 1).detach().cpu().numpy(),
    }
    df = pd.DataFrame(
        np.concatenate([data[k] for k in data], axis=1),
        columns=[k for k in data],
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
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.write_html(
        os.path.join(outdir, filename)
    )

