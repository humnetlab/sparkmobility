import folium
import h3
import matplotlib
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

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


def plot_trajectories(df, latitude_col="latitude", longitude_col="longitude"):
    try:
        map_center = [df[latitude_col].mean(), df[longitude_col].mean()]
    except Exception:
        df["latitude"] = df["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[0])
        df["longitude"] = df["h3_index"].apply(lambda x: h3.cell_to_latlng(x)[1])
        map_center = [df["latitude"].mean(), df["longitude"].mean()]

    m = folium.Map(location=map_center, zoom_start=12, tiles="CartoDB positron")
    coordinates = df[["latitude", "longitude"]].values.tolist()
    folium.PolyLine(
        locations=coordinates, color=custom_colors[-1], weight=2.5, opacity=0.8
    ).add_to(m)

    for coord in coordinates:
        folium.CircleMarker(
            location=coord, radius=2, color=custom_colors[0], fill=True
        ).add_to(m)
    return m
