import warnings

import folium
import h3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from branca.colormap import linear
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

from sparkmobility.utils.session import create_spark_session

warnings.filterwarnings("ignore")

sns.set(style="whitegrid", font_scale=1.5)
custom_colors = [
    "#c45161",
    "#e094a0",
    "#f2b6c0",
    "#f2dde1",
    "#cbc7d8",
    "#8db7d2",
    "#5e62a9",
    "#434279",
]
cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors)
matplotlib.rcParams.update({"legend.fontsize": 14, "legend.handlelength": 2})


def plot_mobility_distributions(myDataset):
    output_path = myDataset.output_path

    pdelta_t = (
        create_spark_session()
        .read.parquet(f"{output_path}/Metrics/StayDurationDistribution")
        .toPandas()
    )
    pt = (
        create_spark_session()
        .read.parquet(f"{output_path}/Metrics/DepartureTimeDistribution")
        .toPandas()
    )
    pn = (
        create_spark_session()
        .read.parquet(f"{output_path}/Metrics/DailyVisitedLocations")
        .toPandas()
    )

    fig, ax = plt.subplots(1, 3, figsize=(12, 3), dpi=300)
    ax[0].plot(
        pn["locations"],
        pn["probability"],
        label="Sparse Input",
        color=custom_colors[-1],
        marker="o",
        markersize=5,
    )
    ax[0].set_yscale("log")

    ax[1].plot(
        pt["hour_of_day"],
        pt["probability"],
        label="Sparse Input",
        color=custom_colors[-1],
        marker="o",
        markersize=5,
    )

    ax[2].plot(
        pdelta_t["range"],
        pdelta_t["probability"],
        color=custom_colors[-1],
        marker="o",
        markersize=5,
    )
    ax[2].set_yscale("log")

    ax[0].set(
        xlabel="$N$",
        xticks=np.arange(0, 21, 4),
        xticklabels=np.arange(0, 24, 4),
        xlim=(0, 24),
    )
    ax[1].set(
        xlabel="$t$",
        xticks=np.arange(0, 25, 8),
        xticklabels=["12am", "8am", "4pm", "12am"],
        xlim=(0, 24),
    )
    ax[2].set(
        xlabel=r"$\Delta t$",
        xticks=[0, 6, 12, 18, 24],
        xticklabels=["0", "6", "12", "18", "24"],
        xlim=(0, 24),
    )

    fig.text(
        0.03,
        0.5,
        "Probability",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=32,
    )
    plt.subplots_adjust(wspace=0.35)
    # plt.legend(bbox_to_anchor=(-2.3,-0.35), loc='upper left', ncol=2, fontsize=24);

    minorLocator = MultipleLocator(2)
    for i in range(3):
        ax[i].xaxis.set_minor_locator(minorLocator)
        ax[i].grid(True, alpha=0.4, linestyle="--", which="both")

    ax[0].text(
        -0.06,
        1.2,
        "A)",
        transform=ax[0].transAxes,
        fontsize=24,
        verticalalignment="top",
    )
    ax[1].text(
        -0.06,
        1.2,
        "B)",
        transform=ax[1].transAxes,
        fontsize=24,
        verticalalignment="top",
    )
    ax[2].text(
        -0.06,
        1.2,
        "C)",
        transform=ax[2].transAxes,
        fontsize=24,
        verticalalignment="top",
    )

    return fig, ax


def plot_flow(
    df,
    min_thickness=0.1,
    max_thickness=3,
    factor=5,
    cutoff_count=0.02,
    max_circle_size=30,
):
    if (
        "origin_lat" not in df.columns
        or "origin_lng" not in df.columns
        or "destination_lat" not in df.columns
        or "destination_lng" not in df.columns
    ):
        df["origin_lat"] = df["origin"].apply(lambda x: h3.cell_to_latlng(x)[0])
        df["origin_lng"] = df["origin"].apply(lambda x: h3.cell_to_latlng(x)[1])
        df["destination_lat"] = df["destination"].apply(
            lambda x: h3.cell_to_latlng(x)[0]
        )
        df["destination_lng"] = df["destination"].apply(
            lambda x: h3.cell_to_latlng(x)[1]
        )

    max_count = df[(df["origin"] != df["destination"])]["flow"].max()
    unique_destinations = df["destination"].unique()
    unique_origins = df["origin"].unique()

    combined_unique_values = pd.unique(
        pd.concat([pd.Series(unique_destinations), pd.Series(unique_origins)])
    )
    flow_counter = pd.Series(0, index=combined_unique_values, name="flow").astype(float)

    map_center = [df["origin_lat"].mean(), df["origin_lng"].mean()]
    m = folium.Map(
        location=map_center,
        zoom_start=10,
        width=900,
        height=630,
        tiles="CartoDB positron",
    )

    for idx, row in df.iterrows():
        start = (row["origin_lat"], row["origin_lng"])
        end = (row["destination_lat"], row["destination_lng"])
        thickness = max(
            min(row["flow"] / max_count * factor, max_thickness), min_thickness
        )
        if (row["flow"] > cutoff_count) & (start != end):
            flow_counter[row["origin"]] += thickness
            flow_counter[row["destination"]] += thickness
            folium.PolyLine(
                [start, end],
                color="blue",
                weight=thickness,  # Adjust divisor to
                opacity=0.3,
            ).add_to(m)

    division = flow_counter.max() / max_circle_size

    for index, item in flow_counter.items():
        if item > 0.1:
            folium.CircleMarker(
                location=h3.cell_to_latlng(index),
                radius=item / division,
                weight=0,
                color="red",  # Adjust the color as needed
                fill=True,
                fill_color="red",
            ).add_to(m)
    return m


def plot_count(df, h3_index_col_name, hex_resolution):
    df["h3_res"] = df[h3_index_col_name].apply(
        lambda x: h3.cell_to_parent(x, hex_resolution)
    )
    count_df = df.groupby("h3_res").sum("count").reset_index()
    count_df["log_count"] = np.log10(count_df["count"])

    count_df["latitude"] = count_df["h3_res"].apply(lambda x: h3.cell_to_latlng(x)[0])
    count_df["longitude"] = count_df["h3_res"].apply(lambda x: h3.cell_to_latlng(x)[1])
    map_center = [count_df["latitude"].mean(), count_df["longitude"].mean()]

    m = folium.Map(
        location=map_center,
        zoom_start=10,
        width=900,
        height=630,
        tiles="CartoDB positron",
    )
    colormap = linear.Blues_09.scale(
        count_df["log_count"].min(), count_df["log_count"].max()
    )
    colormap.caption = "Log Count"
    for i, row in count_df.iterrows():
        hex_boundary = h3.cell_to_boundary(row["h3_res"])
        color = colormap(row["log_count"])
        folium.Polygon(
            locations=hex_boundary,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
        ).add_to(m)

    colormap.add_to(m)

    return m
