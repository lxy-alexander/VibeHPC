# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

##########################
#    Support
##########################


def get_option(options, field, default):
    value = options.get(field, default)
    if value is None:
        value = default
    return default


##########################
#    Back-end methods
##########################


def create_subplot(fig_list):
    this_figure = sp.make_subplots(rows=1, cols=len(fig_list))

    for idx, fig in enumerate(fig_list):
        for trace in fig["data"]:
            this_figure.append_trace(trace, row=1, col=idx + 1)

    return this_figure


##########################
#    Fig catalogue
##########################


def new_bar_breakdown(curr_df, time_base, title, bar_tag):
    fig = px.bar(
        curr_df,
        x="Experiment",
        y=time_base,
        title=title,
        hover_data=["Layer", "Kernel", "At", "ResMma", "ResDram"],
        color=bar_tag,
    )

    return fig


def new_sunburst_breakdown(
    curr_df, time_base, color_base, title, breakdown, alpha_range
):
    if color_base == "At" and alpha_range is not None:
        fig = px.sunburst(
            curr_df,
            path=breakdown,
            values=time_base,
            color=color_base,
            hover_data=["Layer", "Kernel", "At", "ResMma", "ResDram"],
            title=title,
            range_color=[0, float(alpha_range)],
        )

    else:
        fig = px.sunburst(
            curr_df,
            path=breakdown,
            values=time_base,
            color=color_base,
            hover_data=["Layer", "Kernel", "At", "ResMma", "ResDram"],
            title=title,
        )
    fig.update_traces(textinfo="label+percent entry")

    return fig


def new_scatter_map(
    curr_df,
    metric_base_list,
    size_base,
    color_base,
    title,
    range_x=[0, 100],
    range_y=[0, 100],
):
    fig = px.scatter(
        curr_df,
        x=metric_base_list[0],
        y=metric_base_list[1],
        hover_data=["Layer", "Kernel", time_base, "At", "ResMma", "ResDram"],
        color=color_base,
        size=size_base,
        range_x=range_x,
        range_y=range_y,
        title=title,
    )
    return fig


def new_table(df, columns):
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["id"] + list(columns), fill_color="turquoise", align="left"
                ),
                # cells=dict(values=[df.Rank, df.State, df.Postal, df.Population],
                cells=dict(
                    values=[df.index] + [df[col] for col in columns],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )

    return fig


###########################
#   Tables/globals
###########################

METRICS_DEFAULT = ["power.draw [W]", "clocks.current.graphics [MHz]", "temperature.gpu"]
METRICS_DEFAULT_EXT = [
    "power.draw [W]",
    "clocks.current.graphics [MHz]",
    "temperature.gpu",
    "utilization.memory [%]",
    "memory.used [MiB]",
]

####################################
#   Matplotlib backend
####################################


def butter_lowpass_filter(data):
    from scipy.signal import butter, filtfilt

    # Filter requirements.
    T = 5.0  # Sample Period
    fs = 30.0  # sample rate, Hz
    # cutoff = 2      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    cutoff = 0.50  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2  # sin wave can be approx represented as quadratic
    n = int(T * fs)  # total number of samples

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y


def get_plot_data(raw_data, filt=True):
    # convert to FP
    data_vec = [float(item) for item in raw_data]

    # apply low-pass filter
    if filt:
        data_vec_filt = butter_lowpass_filter(data_vec)
        return data_vec_filt
    else:
        return data_vec


def get_canvas(x_dim, y_dim):
    canvas = plt.figure(figsize=(3 * 7.0, y_dim * 3.4))
    rect = canvas.patch
    rect.set_facecolor("white")
    return canvas


def p_conf_splt(splt, tag, color="red"):
    splt.tick_params(axis="x", colors=color)
    splt.tick_params(axis="y", colors=color)
    splt.spines["bottom"].set_color(color)
    splt.spines["top"].set_color(color)
    splt.spines["left"].set_color(color)
    splt.spines["right"].set_color(color)


def p_plot(splt, tag, time, data, color="red", graph="plot"):
    p_conf_splt(splt, tag, color)
    if graph == "plot":
        y_data = time
        splt.plot(y_data, data, ":", linewidth=0.6, color=color)
    if graph == "hist":
        splt.hist(data, density=True, color=color)


def get_min(vec):
    return round(min(vec), 3)


def get_max(vec):
    return round(max(vec), 3)


def get_avg(vec):
    return round(sum(vec) / float(len(vec)), 3)


def p_2d_plot(df_in, time, main_tag, metrics):
    # static variables
    x_colors = ["red", "blue", "green", "darkorange", "purple", "darkgreen"]
    graph = "plot"

    # decode main parameters
    df = df_in.copy(deep=True)
    x_tags = metrics
    y_tags = df[main_tag].unique()
    x_dim = len(x_tags)
    y_dim = len(y_tags)
    y_lim = dict()

    # create canvas and add sub-plots
    canvas = get_canvas(x_dim, y_dim)

    # get ranges and apply low pass filter
    for mtr in metrics:
        y_lim[mtr] = [df[mtr].min(), df[mtr].max()]

        # apply filter per gpu
        for tag in y_tags:
            df_index = df[main_tag] == tag
            df.loc[df_index, mtr] = get_plot_data(
                list(df.loc[df_index, mtr]), filt=True
            )

        # re-arrange y_lim max for aesthetics
        y_lim[mtr][1] = y_lim[mtr][1] * 1.05
        if y_lim[mtr][1] == 0:
            y_lim[mtr][1] = 1

    # generate subplots
    for row_idx, y_tag in enumerate(y_tags):
        for col_idx, x_tag in enumerate(x_tags):
            name = x_tag
            row_name = re.sub(r"00000000:", "", y_tag)
            tag = name
            canvas_idx = row_idx * x_dim + col_idx + 1
            splt = canvas.add_subplot(y_dim, x_dim, canvas_idx)
            splt.set_ylim(y_lim[x_tag])
            if row_idx == 0:
                splt.set_title(tag, color=x_colors[col_idx])
            if row_idx == y_dim - 1:
                splt.set_xlabel(time, color=x_colors[col_idx])
            if col_idx == 0:
                splt.set_ylabel(row_name, color=x_colors[0])
            df_data = df[df[main_tag] == y_tag]
            p_plot(
                splt,
                name,
                list(df_data[time]),
                list(df_data[x_tag]),
                color=x_colors[col_idx],
                graph=graph,
            )


def create_report_matplotlib(df, output_file, options):
    time = "Time(s)"
    main_tag = "pci.bus_id"
    metric_list = get_option(options, "metrics", METRICS_DEFAULT_EXT)

    # create plots
    p_2d_plot(df, time, main_tag, metric_list)

    # summary
    summary_df = get_summary_df(df, metric_list)
    mean_df = get_mean_df(
        df,
        output_file,
        [
            "clocks.current.graphics [MHz]",
            "power.draw [W]",
            "utilization.memory [%]",
            "memory.used [MiB]",
        ],
    )

    # get GPU to GPU variance
    divergence_df = get_divergence_df(df, metric_list)

    # render
    if output_file == "gui":
        plt.show()
        print(summary_df.to_string())
    else:
        if re.search(r"\.html\s*$", output_file):
            fig_file = re.sub(r"\.html\s*$", ".svg", output_file)
            plt.savefig(fig_file)
            with open(output_file, "w") as fh:
                fh.write("<h2>{}</h2>".format(options.get("input")))
                fh.write(summary_df.to_html())
                fh.write("<h2>divergence</h2>")
                fh.write(divergence_df.to_html())
                fh.write(
                    '<img src="{}" alt="{}">'.format(fig_file, options.get("input"))
                )
        else:
            fig_file = output_file
            plt.savefig(fig_file)
            print(summary_df.to_string())


###########################
#   Plotly backend
###########################


def figures_to_html(figs, filename="dashboard.html"):
    dashboard = open(filename, "w")
    dashboard.write("<html><head></head><body>" + "\n")
    for fig in figs:
        inner_html = fig.to_html().split("<body>")[1].split("</body>")[0]
        dashboard.write(inner_html)
    dashboard.write("</body></html>" + "\n")


def create_col_subplot(fig_list):
    this_figure = sp.make_subplots(rows=1, cols=len(fig_list))
    for idx, fig in enumerate(fig_list):
        for trace in fig["data"]:
            this_figure.append_trace(trace, row=1, col=idx + 1)
    return this_figure


def create_row_subplot(fig_list):
    this_figure = sp.make_subplots(rows=len(fig_list), cols=1)
    for idx, fig in enumerate(fig_list):
        for trace in fig["data"]:
            this_figure.append_trace(trace, row=idx + 1, col=1)
    return this_figure


def create_time_plot(df, metric, time, color, secondary_color, options):
    multi_column = options.get("multi_column", False)

    kwargs = {"height": 400}
    if multi_column:
        kwargs.update({"facet_col": color, "color": secondary_color})
    else:
        kwargs.update({"color": color, "marginal_y": "violin"})

    fig = px.scatter(df, x=time, y=metric, **kwargs)
    return fig


def create_melt_time_plot(df, metric, time, color, secondary_color, options):
    multi_column = options.get("multi_column", False)
    if not isinstance(metric, list):
        raise (Exception("metric is not a list"))

    height = 1300
    kwargs = {"height": height}
    if multi_column:
        kwargs.update({"facet_col": color, "color": secondary_color})
    else:
        kwargs.update({"color": color, "marginal_y": "violin"})

    id_vars = [time, color]
    if multi_column:
        id_vars.append(secondary_color)

    df_melt = pd.melt(df, id_vars=id_vars, value_vars=metric)  # long form
    # print(df_melt)
    kwargs["facet_row"] = "variable"

    fig = px.scatter(df_melt, x=time, y="value", **kwargs)

    # change y-range per row to be 'floating'
    fig.update_yaxes(matches=None)

    return fig


def create_correlation_plot(df, metric, base):
    fig = px.scatter(df, x=base, y=metric, marginal_x="violin", marginal_y="violin")
    return fig


def create_report_plotly(df, output_html, options):
    fig_list = list()
    # tags = df.columns

    time = "Time(s)"
    main_tag = "pci.bus_id"
    secondary_tag = get_option(options, "multi_column_color", "temperature.gpu")
    metric_list = get_option(options, "metrics", METRICS_DEFAULT)

    fig_list.append(
        create_melt_time_plot(df, metric_list, time, main_tag, secondary_tag, options)
    )
    figures_to_html(fig_list, output_html)

    summary_df = get_summary_df(df, metric_list)
    divergence_df = get_divergence_df(df, metric_list)
    print(summary_df.to_string())
    print(divergence_df.to_string())


###########################
#   Pandas backend
###########################


def get_summary_df(df, metrics=None):
    # pre-process args
    if metrics is None:
        metrics = METRICS_DEFAULT

    # remove some non-aggregatable columns
    filter_list = ["timestamp", "pstate"]
    agg_fields = [col for col in df.columns if col not in filter_list]

    # Define the aggregation functions and apply the aggregation functions to each column
    agg_funcs = ["min", "max", "mean"]
    summary_df = df[agg_fields].groupby("pci.bus_id").agg(agg_funcs)
    return summary_df.T


def get_mean_df(df, output_name, metrics=None):
    # pre-process args
    if metrics is None:
        metrics = METRICS_DEFAULT
    mean_df = df[metrics + ["pci.bus_id"]].groupby("pci.bus_id").agg("mean")
    mean_summary_df = mean_df.mean().to_frame().T
    mean_summary_df = mean_summary_df.round(1)
    mean_summary_df["file"] = output_name
    mean_summary_df.set_index("file", inplace=True)
    print(mean_summary_df.to_string())
    return mean_df


def get_divergence_df(df, metrics=None):
    # pre-process args
    if metrics is None:
        metrics = METRICS_DEFAULT

    # remove some non-aggregatable columns
    filter_list = ["timestamp", "pstate"]
    agg_fields = [col for col in df.columns if col not in filter_list]
    mean_df = df[agg_fields].groupby("pci.bus_id").agg("mean")
    deviations_df = mean_df.apply(lambda x: 100 * abs(x - x.mean()) / x.mean())
    max_deviations_df = deviations_df.max().to_frame()
    return max_deviations_df.T


###########################
#   Frontend processing
###########################


def read_df_from_csv(csv_file_name, active=False, df_query=None):
    # Read csv and obtain relative time in s
    df = pd.read_csv(csv_file_name)
    df["Time(s)"] = pd.to_datetime(df["timestamp"]).astype(np.int64) / int(1e9)
    df["Time(s)"] -= df["Time(s)"].iloc[0]

    # Filter out inactive rows
    if active:
        threshold = 0.80 * df["utilization.gpu [%]"].max()
        df = df[df["utilization.gpu [%]"] > threshold]

    # User level queries
    if df_query is not None:
        df = df.query(df_query)

    return df


###########################
#   Main method
###########################


def create_report(df, output, options):
    if options.get("dynamic", False):
        if not re.search(r"\.html\s*$", output):
            raise (Exception("Invalid file format (must be html)"))
        create_report_plotly(df, output, options)
    else:
        create_report_matplotlib(df, output, options)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, help="Input nvidia-smi csv file"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output file (html/svg/png/gui)"
    )
    parser.add_argument(
        "--dynamic",
        dest="dynamic",
        action="store_true",
        help="Dynamic vs static html (ploty vs matplotlib)",
    )
    parser.add_argument(
        "--active",
        dest="active",
        action="store_true",
        help="Show only active GPU portions",
    )
    parser.add_argument(
        "--metrics", required=False, nargs="+", help="Specify metric(s) to visualize"
    )
    parser.add_argument(
        "--multi_column",
        dest="multi_column",
        action="store_true",
        help="Show GPU plots in different columns",
    )
    parser.add_argument(
        "--multi_column_color",
        required=False,
        help="Secondary color/temp metric (multi-column only)",
    )
    parser.add_argument(
        "--query",
        required=False,
        default=None,
        help="Filter table through pandas query",
    )
    args = parser.parse_args()

    file_name = args.input
    options = vars(args)

    df = read_df_from_csv(file_name, args.active, args.query)
    create_report(df, args.output, options)


if __name__ == "__main__":
    main()
