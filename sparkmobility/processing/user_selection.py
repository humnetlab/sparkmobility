import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

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


class UserSelection:
    def __init__(
        self,
        MobilityDataset,
    ):
        self.dataset = MobilityDataset

    def filter_users(
        self,
        num_stay_points_range,
        time_span_days_range,
    ):
        df = self.dataset.load_stays()

        df_grouped = (
            df.groupBy("caid")
            .agg(
                F.count("*").alias("num_stays"),
                F.min("stay_start_timestamp").alias("first_stay"),
                F.max("stay_end_timestamp").alias("last_stay"),
            )
            .withColumn(
                "duration_days",
                F.round(
                    (F.unix_timestamp("last_stay") - F.unix_timestamp("first_stay"))
                    / 86400
                ).cast("int"),
            )
        ).cache()

        active_level = (
            df_grouped.groupBy("duration_days", "num_stays")
            .count()
            .orderBy("duration_days", "num_stays")
            .toPandas()
        )

        filter_condition = (
            (F.col("num_stays") >= num_stay_points_range[0])
            & (F.col("num_stays") <= num_stay_points_range[1])
            & (F.col("duration_days") >= time_span_days_range[0])
            & (F.col("duration_days") <= time_span_days_range[1])
        )

        selected_users = df_grouped.filter(filter_condition).select("caid")

        filtered_df = df.join(selected_users, on="caid", how="inner")

        # df_grouped has one row per caid, so its row count equals the distinct
        # user count. Computing both totals in a single agg avoids the extra
        # distinct+count and filter+count shuffles.
        counts = df_grouped.agg(
            F.count("*").alias("total"),
            F.sum(F.when(filter_condition, 1).otherwise(0)).alias("filtered"),
        ).first()
        self.dataset.num_total_users = counts["total"]
        self.dataset.num_filtered_users = counts["filtered"]

        filtered_df.write.mode("overwrite").parquet(
            self.dataset.output_path + "/FilteredUserStayPoints"
        )

        fig, ax = self._visualize(
            active_level, num_stay_points_range, time_span_days_range
        )

        out_path = self.dataset.output_path + "/FilteredUserStayPoints"
        logger.info(
            "filtered %d/%d users; saved to %s",
            self.dataset.num_filtered_users,
            self.dataset.num_total_users,
            out_path,
        )
        df_grouped.unpersist()
        return fig, ax

    @staticmethod
    def _visualize(active_level, num_stay_points_range, time_span_days_range):
        pivot = active_level.pivot(
            index="duration_days", columns="num_stays", values="count"
        ).fillna(0)
        log_data = np.log10(pivot + 1)
        mask = pivot.values == 0
        masked_data = np.ma.masked_where(mask, log_data.values)

        cmap_masked = LinearSegmentedColormap.from_list(
            "custom_cmap_masked", custom_colors
        )
        cmap_masked.set_bad("white")

        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        ax.grid(False)
        ax.minorticks_off()
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        im = ax.imshow(masked_data, aspect="auto", cmap=cmap_masked)

        rect = plt.Rectangle(
            (num_stay_points_range[0], time_span_days_range[0]),
            (num_stay_points_range[1] - num_stay_points_range[0]),
            (time_span_days_range[1] - time_span_days_range[0]),
            linewidth=2,
            edgecolor=custom_colors[-1],
            facecolor="none",
        )
        ax.add_patch(rect)

        num_cols = log_data.shape[1]
        if num_cols > 10:
            xticks = np.linspace(0, num_cols - 1, 10, dtype=int)
        else:
            xticks = np.arange(num_cols)
        ax.set_xticks(xticks)
        ax.set_xticklabels(log_data.columns[xticks], rotation=90)

        num_rows = log_data.shape[0]
        if num_rows > 10:
            yticks = np.linspace(0, num_rows - 1, 10, dtype=int)
        else:
            yticks = np.arange(num_rows)

        ax.set_yticks(yticks)
        ax.set_yticklabels(log_data.index[yticks])
        ax.invert_yaxis()

        ax.set_xlabel("Number of Stay Points")
        ax.set_ylabel("Time Span (Days)")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Log10 Number of Users")
        plt.tight_layout()

        return fig, ax
