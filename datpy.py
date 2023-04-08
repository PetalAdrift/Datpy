import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import math
from decimal import Decimal as D
import numpy as np
import sympy as sp
from scipy.optimize import curve_fit
import scipy.stats as ss
from typing import List, Callable
from inspect import signature


HEADING_FONT_SIZE = 12
LABEL_FONT_SIZE = 12
BODY_FONT_SIZE = 10
PLOT_SIZE = 3
CAP_SIZE = 2
EXTRAPOLATION_LOWER_RANGE = 0.2
EXTRAPOLATION_UPPER_RANGE = 0.2
PLOT_STEP = 0.0025
FIGURE_MARGIN = 0.05
COLOR_PALETTE = ["#EE5515", "#65CB4D", "#446688", "#663377"]
COLOR_NUM = 0


def get_params_count(func: callable) -> int:
    """
    Returns number of parameters used by func.
    """
    sig = signature(func)
    params = sig.parameters
    return len(params)


def set_ex_range(lower_range: float, upper_range: float) -> None:
    """
    Changes global variables EXTRAPOLATION_LOWER_RANGE and
    EXTRAPOLATION_UPPER_RANGE.
    """
    global EXTRAPOLATION_LOWER_RANGE
    global EXTRAPOLATION_UPPER_RANGE
    EXTRAPOLATION_LOWER_RANGE = lower_range
    EXTRAPOLATION_UPPER_RANGE = upper_range


def set_plot_step(proportion: float) -> None:
    """
    Changes global variable PLOT_STEP.
    If proportion is greater than 1, treats it as number of steps.
    """
    global PLOT_STEP
    if proportion > 1:
        proportion = 1 / int(proportion)
    PLOT_STEP = proportion


def set_margin(margin: float) -> None:
    """
    Changes global variable FIGURE_MARGIN
    """
    global FIGURE_MARGIN
    FIGURE_MARGIN = margin


def get_color() -> str:
    """
    Returns a color from COLOR_PALETTE in order.
    """
    global COLOR_NUM
    color = COLOR_PALETTE[COLOR_NUM % len(COLOR_PALETTE)]
    COLOR_NUM += 1
    return color


def remove_legend(fig: "matplotlib.figure.Figure", ax: int) -> None:
    """
    Removes legend from axis with index ax.
    """
    ax = fig.get_axes()[ax]
    current_leg = ax.get_legend()
    if current_leg is not None:
        current_leg.remove()


def get_gauss(x: float, amp: float, mu: float, sigma: float, base: float) -> float:
    """
    Returns value of Gaussian distribution given by
    f(x) = amp exp(-(x - mu)^2 / (2 * sigma^2)) + base.
    Width of Gaussian distribution is 2 * sqrt(2 * ln(2)) * sigma.
    """
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + base


def get_round(num: D, digit: int) -> D:
    """
    Returns the rounded value of num with precision of digit.
    """
    aug_num = round(num * D("10") ** (-D(str(digit))))
    aug_num = aug_num * D("10") ** D(str(digit))
    return aug_num


def get_high_digit(num: D) -> int:
    """
    Returns the highest significant digit of num.
    """
    return int(np.floor(np.log10(float(num))))


def keep_digits(num: D, digit: int) -> D:
    """
    Returns num with only specified significant digits kept.
    """
    return get_round(num, get_high_digit(num) - digit + 1)


def load_data(
    file_name: str, headers=(1, 0), orientation="col", delimiter=",", skiprows=0
) -> list[list]:
    """
    The file with file_name is read as a .csv-like file with headers.
    Returns the data as a list containing separate columns or rows depending on
    parameter orientation.
    """
    data = np.loadtxt(
        file_name,
        dtype="str",
        comments=None,
        delimiter=delimiter,
        skiprows=skiprows,
        encoding="utf-8",
    )
    data = data[headers[0] :, headers[1] :]
    data = np.array(data, dtype="float")
    data_list = []
    if orientation == "col":
        for col in range(data.shape[1]):
            data_list.append(data[:, col])
    elif orientation == "row":
        for row in range(data.shape[0]):
            data_list.append(data[row, :])
    return data_list


def get_chi_squared(
    val_data: list[float],
    val_predicted: list[float],
    val_err: list[float],
    para_amount: int,
) -> [float, float, list[float]]:
    """
    Returns chi-squared, reduced chi-squared, chi-squared probability,
    and uncertainty that normalizes reduced chi-squared.
    """
    val_data = np.array(val_data)
    val_predicted = np.array(val_predicted)
    val_err = np.array(val_err)

    degree_of_freedom = len(val_data) - para_amount
    chi_squared = np.sum(((val_data - val_predicted) / val_err) ** 2)
    chi_squared_red = chi_squared / degree_of_freedom
    chi_squared_prob = 1 - ss.chi2.cdf(chi_squared, degree_of_freedom)
    unc_acur = chi_squared_red**0.5 * val_err
    return (chi_squared, chi_squared_red, chi_squared_prob, unc_acur)


def get_best_fit(
    y_vals: list[float],
    y_errs: list[float],
    x_vals: list[float],
    expr: str,
    para_amount: int,
    para_names="x a b c d m n p q",
    guess=None,
    sp_expr="",
    fit_func=None,
    max_fev=800,
) -> (list[float], list[float], list[float], list[list[float]]):
    """
    Returns optimized parameters, their standard deviations, predicted
    values calculated with optimized parameters,
    and predicted values for plotting.
    Parameters are in the order of para_names, the amount used is indicated by
    para_amount.
    An auxiliary sp_expr is only used when expr fails for sp.parsing.
    An auxiliary func: Callable is only used when parsing is not satisfactory.
    """
    var_names = sp.symbols(para_names)

    if fit_func is None:
        func = sp.lambdify(var_names[:para_amount], expr)
    elif fit_func is not None:
        func = fit_func

    popt, pcov = curve_fit(
        func,
        x_vals,
        y_vals,
        p0=guess,
        sigma=y_errs,
        absolute_sigma=True,
        maxfev=max_fev,
    )
    pstd = np.sqrt(np.diag(pcov))

    subs_dict = {}
    for i in range(1, para_amount):
        subs_dict[var_names[i]] = popt[i - 1]

    if fit_func is None:
        if sp_expr == "":
            expr = sp.parsing.sympy_parser.parse_expr(expr)
        else:
            expr = sp.parsing.sympy_parser.parse_expr(sp_expr)

        y_predicted = []
        for x_val in x_vals:
            subs_dict[var_names[0]] = x_val
            y_predicted.append(expr.subs(subs_dict))
    elif fit_func is not None:
        y_predicted = []
        for x_val in x_vals:
            params = [x_val]
            params.extend(popt)
            y_predicted.append(fit_func(*params))

    plot_predicted = []
    x_range = max(x_vals) - min(x_vals)
    if fit_func is None:
        for x_val in np.arange(
            min(x_vals) - EXTRAPOLATION_LOWER_RANGE * x_range,
            max(x_vals) + EXTRAPOLATION_UPPER_RANGE * x_range + PLOT_STEP * x_range,
            PLOT_STEP * x_range,
        ):
            subs_dict[var_names[0]] = x_val
            plot_predicted.append([x_val, expr.subs(subs_dict)])
    elif fit_func is not None:
        for x_val in np.arange(
            min(x_vals) - EXTRAPOLATION_LOWER_RANGE * x_range,
            max(x_vals) + EXTRAPOLATION_UPPER_RANGE * x_range + PLOT_STEP * x_range,
            PLOT_STEP * x_range,
        ):
            params = [x_val]
            params.extend(popt)
            plot_predicted.append([x_val, fit_func(*params)])

    y_predicted = np.array(y_predicted, dtype=float)
    plot_predicted = np.array(plot_predicted, dtype=float)

    return (popt, pstd, y_predicted, plot_predicted)


def propagate_uncertainty(
    func: str,
    var_vl: list[list[float]],
    var_unc: list[list[float]],
    var_names="x y z a b c d",
) -> list[float]:
    """
    Returns uncertainties of variable defined by func with respect to var_vl and
    var_unc. Variables are passed in the order of var_names.
    """
    func = sp.parsing.sympy_parser.parse_expr(func)
    uncs = []
    for i in range(len(var_vl[0])):
        subs_dict = {}
        for j in range(len(var_vl)):
            subs_dict[sp.symbols(var_names[2 * j : 2 * j + 1])] = D(var_vl[j][i])
        value = func.subs(subs_dict)
        unc_squared = D(0)
        for j in range(len(var_vl)):
            unc_squared += (
                D(
                    str(
                        sp.diff(func, sp.symbols(var_names[2 * j : 2 * j + 1]))
                        .subs(subs_dict)
                        .evalf()
                    )
                )
                * D(var_unc[j][i])
            ) ** D("2")
        unc = unc_squared ** D("0.5")
        uncs.append(float(unc))
    return uncs


def add_simple_plot(
    fig: "matplotlib.figure.Figure",
    x_vals: list[float],
    y_vals: list[float],
    x_errs: list[float],
    y_errs: list[float],
    leg: str,
    plot_data=False,
    ax=-1,
) -> None:
    """
    Addes a subplot to fig.
    """
    remove_legend(fig, ax)
    ax = fig.get_axes()[ax]
    color = get_color()
    if not plot_data:
        ax.scatter(x_vals, y_vals, s=PLOT_SIZE, c=color, label=leg)
    elif plot_data:
        ax.plot(x_vals, y_vals, linewidth=PLOT_SIZE / 3, c=color, label=leg)
    if not len(x_errs) == 0:
        ax.errorbar(
            x_vals,
            y_vals,
            xerr=x_errs,
            yerr=None,
            fmt="none",
            lw=1,
            capsize=CAP_SIZE,
            c=color,
        )
    if not len(y_errs) == 0:
        ax.errorbar(
            x_vals,
            y_vals,
            xerr=None,
            yerr=y_errs,
            fmt="none",
            lw=1,
            capsize=CAP_SIZE,
            c=color,
        )
    ax.legend(fontsize=BODY_FONT_SIZE)


def create_axis_with_residual(
    fig: "matplotlib.figure.Figure",
    pos: int,
    x_vals: list[float],
    y_vals: list[float],
    y_predicted: list[float],
    plot_predicted: list[list[float]],
    x_errs: list[float],
    y_errs: list[float],
    x_label: str,
    y_label: str,
    title: str,
    scale="linear",
    reversed=False,
    condensed=False,
    data_leg="data",
    pred_leg="prediction",
    remove_errs=(0, 0, 0),
    plot_data=False,
) -> float:
    """
    Creates a subplot of fig with a residual plot.
    The position is given by three digit integer pos.
    Returns proportion of residuals which fall within one sigma.
    Returns -1 if proportion does not exist.
    """
    x_vals = np.array(x_vals, dtype=float)
    y_vals = np.array(y_vals, dtype=float)
    if not len(x_errs) == 0:
        x_errs = np.array(x_errs, dtype=float)
    else:
        x_errs = None
    if not len(y_errs) == 0:
        y_errs = np.array(y_errs, dtype=float)
    else:
        y_errs = None
    plot_predicted = np.array(plot_predicted, dtype=float)

    ax = fig.add_subplot(pos)

    if not plot_data:
        ax.errorbar(
            x_vals,
            y_vals,
            marker=".",
            markersize=PLOT_SIZE,
            ls="none",
            c="#6680FF",
            label=data_leg,
        )
    elif plot_data:
        ax.plot(x_vals, y_vals, linewidth=PLOT_SIZE / 3, c="#6680FF", label=data_leg)
    if (x_errs is not None) and (remove_errs[0] != 1):
        ax.errorbar(
            x_vals, y_vals, xerr=x_errs, fmt="none", lw=1, capsize=CAP_SIZE, c="#66CCFF"
        )
    if (y_errs is not None) and (remove_errs[1] != 1):
        ax.errorbar(
            x_vals, y_vals, yerr=y_errs, fmt="none", lw=1, capsize=CAP_SIZE, c="#66CCFF"
        )
    if scale == "linear":
        ax.set_yscale("linear")
    if scale == "lg":
        ax.set_yscale("log")
    if scale == "ln":
        ax.set_yscale(matplotlib.scale.LogScale(ax, base=np.e))
        y_ticks = np.array(ax.get_yticks())
        y_ticks = y_ticks[(y_ticks >= ax.get_ylim()[0]) & (y_ticks <= ax.get_ylim()[1])]
        e_ticks = np.round(np.log(y_ticks)).astype(int)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(
            [f"$e^{{{tick}}}$" for tick in e_ticks], fontsize=BODY_FONT_SIZE
        )

    ax.set_xlim(
        [
            np.min(x_vals) - FIGURE_MARGIN * (np.max(x_vals) - np.min(x_vals)),
            np.max(x_vals) + FIGURE_MARGIN * (np.max(x_vals) - np.min(x_vals)),
        ]
    )

    if not len(plot_predicted) == 0:
        ax.plot(
            plot_predicted[:, 0],
            plot_predicted[:, 1],
            lw=1,
            c="#FF9966",
            label=pred_leg,
            zorder=99,
        )
        left_lim, right_lim = ax.get_xlim()
        ax.set_xlim(
            [
                np.min(
                    [
                        left_lim,
                        np.min(plot_predicted[:, 0])
                        - FIGURE_MARGIN
                        * (np.max(plot_predicted[:, 0]) - np.min(plot_predicted[:, 0])),
                    ]
                ),
                np.max(
                    [
                        right_lim,
                        np.max(plot_predicted[:, 0])
                        + FIGURE_MARGIN
                        * (np.max(plot_predicted[:, 0]) - np.min(plot_predicted[:, 0])),
                    ]
                ),
            ]
        )

    if reversed or len(y_predicted) == 0:
        ax.set_xlabel(x_label, fontsize=LABEL_FONT_SIZE)
        tick_axis = "both"
        res_tick_axis = "both"
        if condensed:
            res_tick_axis = "y"
    else:
        tick_axis = "both"
        res_tick_axis = "both"
        if condensed:
            tick_axis = "y"
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE)
    if title != "":
        ax.set_title(title, fontsize=HEADING_FONT_SIZE)
    ax.tick_params(
        axis=tick_axis, which="both", direction="in", labelsize=BODY_FONT_SIZE
    )
    if tick_axis == "y":
        ax.set_xticks([])
    ax.legend(fontsize=BODY_FONT_SIZE)

    if not len(y_predicted) == 0:
        ax_divider = make_axes_locatable(ax)
        if not reversed:
            position = "bottom"
        else:
            position = "top"
        if not condensed:
            pad = "10%"
        else:
            pad = "2%"
        ax_res = ax_divider.append_axes(
            position, size="20%", pad=pad
        )  # adds a residual plot

        residual_vals = y_vals - y_predicted

        ax_res.scatter(x_vals, residual_vals, s=PLOT_SIZE, c="blue", label="residual")
        if (y_errs is not None) and (remove_errs[2] != 1):
            ax_res.errorbar(
                x_vals,
                residual_vals,
                yerr=y_errs,
                fmt="none",
                lw=1,
                capsize=CAP_SIZE,
                c="blue",
            )
        ax_res.plot(
            [np.min(plot_predicted[:, 0]), np.max(plot_predicted[:, 0])],
            [0, 0],
            lw=1,
            c="black",
        )
        ax_res.set_xlim(ax.get_xlim())
        if not reversed:
            ax_res.set_xlabel(x_label, fontsize=LABEL_FONT_SIZE)
        y_unit_label = y_label[y_label.rfind("(") :]
        ax_res.set_ylabel("residual " + y_unit_label, fontsize=LABEL_FONT_SIZE)
        ax_res.tick_params(
            axis=res_tick_axis, which="both", direction="in", labelsize=BODY_FONT_SIZE
        )
        if res_tick_axis == "y":
            ax_res.set_xticks([])
        ax_res.legend(fontsize=BODY_FONT_SIZE)

        if y_errs is not None:
            res_proportion = np.sum(np.abs(residual_vals) <= y_errs, axis=0) / len(
                y_vals
            )
            return res_proportion
        else:
            return -1
    return -1


def LEGACY_create_axis_histogram(
    fig: "matplotlib.figure.Figure",
    pos: int,
    vals: list[float],
    bins_count: int,
    x_label: str,
    y_label: str,
    title: str,
    distributions=(1, 1, 1),
) -> None:
    """
    Addes a histogram subplot to fig.
    The position is given by three digit integer pos.
    bins_count could take in a list as defined edges.
    """
    vals_avg = np.mean(vals)
    vals_unc = np.sqrt(vals_avg)
    his_vals, his_bins = np.histogram(vals, bins=bins_count, density=True)

    width = his_bins[1] - his_bins[0]
    bins_centers = his_bins[:-1] + width / 2

    x_stamps = np.linspace(his_bins[0] - width, his_bins[-1] + width, 1000)
    x_marks = np.arange(np.floor(his_bins[0] - width), his_bins[-1] + width, 1)

    ax = fig.add_subplot(pos)
    ax.bar(bins_centers, his_vals, width=width * 0.8, color="#66CCFF", label="data")
    ax.set_xlabel(x_label, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=HEADING_FONT_SIZE)
    ax.tick_params(axis="both", which="both", direction="in", labelsize=BODY_FONT_SIZE)
    if distributions[0] == 1:
        ax.plot(
            x_stamps,
            ss.norm.pdf(x_stamps, vals_avg, vals_unc),
            label="Gaussian distribution",
        )
    if distributions[1] == 1:
        ax.plot(x_stamps, ss.gamma.pdf(x_stamps, vals_avg), label="gamma distribution")
    if distributions[2] == 1:
        ax.plot(
            x_marks, ss.poisson.pmf(x_marks, vals_avg), label="Poisson distribution"
        )

    ax.vlines(
        vals_avg,
        0,
        np.max(his_vals * 1.1),
        color="#FF0000",
        linestyles="dashed",
        label="mean",
    )

    ax.legend(fontsize=BODY_FONT_SIZE)


def create_axis_histogram(
    fig: "matplotlib.figure.Figure",
    pos: int,
    vals: List[float],
    bins_num: int,
    x_label: str,
    y_label: str,
    title: str,
    bins_width=0.0,
    bins_range=(0, -1),
    uncertainties=False,
    distribution=(None, ""),
    guess=None,
    forced=False,
) -> (List[int], List[float], List[float]):
    """
    Adds a histogram to fig. Priority is given in the order of bins_range,
    bins_width, and bins_num. When bins_width is used, bins_num is redefined.
    If uncertainties is True, plot error bars.
    If distibution[0] is a function, tries to fit the function with parameters
    to the data. Name of distribution is given by distribution[1].
    Additionally a guess for the distribution function could be provided.
    Returns values in bins, uncertainties of values, and centers of bins.
    If distribution is given, additionally returns popt, pstd, and chi^2.
    """
    ax = fig.add_subplot(pos)
    if bins_range[0] < bins_range[1]:
        ran = bins_range
    elif bins_width != 0.0:
        bins_num = math.ceil((np.max(vals) - np.min(vals)) / bins_width)
        ran = (
            (np.max(vals) + np.min(vals) - bins_width * bins_num) * 0.5,
            (np.max(vals) + np.min(vals) + bins_width * bins_num) * 0.5,
        )
    else:
        ran = None
    num, bin_edges, _ = ax.hist(
        vals, bins=bins_num, range=ran, color="#66CCFF", histtype="bar", rwidth=0.9
    )
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    unc_num = np.sqrt(num)
    unc_num = np.where(unc_num == 0, 1, unc_num)
    if uncertainties:
        ax.errorbar(bin_centers, num, yerr=unc_num, fmt="none", capsize=CAP_SIZE, c="k")
    ax.set_xlabel(x_label, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=HEADING_FONT_SIZE)
    if distribution[0] is None:
        return (num, unc_num, bin_centers)
    if not forced:
        popt, pstd, pred, plot = get_best_fit(
            num, unc_num, bin_centers, "", 5, guess=guess, fit_func=distribution[0]
        )
    if forced:
        popt = guess
        pstd = []
        pred = []
        plot = np.array([[0, 0]])
        for bin_center in bin_centers:
            params = [bin_center]
            params.extend(popt)
            pred.append(distribution[0](*params))
    chi = get_chi_squared(num, pred, unc_num, get_params_count(distribution[0]) - 1)
    x_unit_label = x_label[x_label.rfind("(") :]
    if distribution[0] == get_gauss:
        full_width = 2 * np.sqrt(2 * np.log(2)) * popt[2]
        text = (
            f"$\mu$ = {str(keep_digits(D(popt[1]), 3))} "
            + f"{x_unit_label}\n"
            + f"FWHM = {str(keep_digits(D(full_width), 3))} "
            + f"{x_unit_label}\n"
            + f"$\chi^2$/DOF = {chi[1]:.2f}\n"
            + f"$\chi^2$ probability = {chi[2]:.2f}\n"
        )
    else:
        text = (
            f"$\mu$ = {str(keep_digits(D(popt[1]), 3))} "
            + f"{x_unit_label}\n"
            + f"{x_unit_label}\n"
            + f"$\chi^2$/DOF = {chi[1]:.2f}\n"
            + f"$\chi^2$ probability = {chi[2]:.2f}\n"
        )
    add_simple_plot(
        fig,
        plot[:, 0],
        plot[:, 1],
        [],
        [],
        f"best fit of {distribution[1]}",
        plot_data=True,
    )
    ##    ax.text(
    ##        0.95 * ax.get_xlim()[0] + 0.05 * ax.get_xlim()[1],
    ##        0.5 * ax.get_ylim()[0] + 0.5 * ax.get_ylim()[1],
    ##        text,
    ##        fontsize=LABEL_FONT_SIZE,
    ##    )
    remove_legend(fig, -1)
    ax.legend(fontsize=BODY_FONT_SIZE, title=text, title_fontsize=BODY_FONT_SIZE)
    return (num, unc_num, bin_centers, popt, pstd, chi)


def format_table() -> None:
    """
    Opens a .csv file and converts into LaTeX-formatted table.
    """
    file_name = input("Enter file name: ")
    max_line = int(input("Enter maximum number of lines: "))
    hline_style = input(
        "Enter horizontal line style (default: header only; "
        "n: none; a: all; b: enclosed): "
    )
    if hline_style == "n":
        top_hline = " \\\\\n"
    else:
        top_hline = " \\\\\n\t\t\\hline\n"
    if hline_style == "a" or hline_style == "b":
        mid_hline = "\t\t\\hline\n\t\t"
    else:
        mid_hline = "\t\t"
    with open(file_name) as f:
        rows = f.readlines()
        col_num = rows[0].count(",") + 1
        headers = []
        if rows[0][0] == "#":
            raw_headers = rows[0].split(",")
            for header in raw_headers:
                headers.append(header[1:].strip())
            rows = rows[1:]
        if len(rows) > max_line:
            col_num = col_num * math.ceil((len(rows) / max_line))
        for i in range(len(rows)):
            rows[i] = rows[i].replace(",", "\t& ")
            if i >= max_line:
                rows[i % max_line] = rows[i % max_line][:-1] + "\t& " + rows[i]
    rows = rows[:max_line]
    table_rows = (row[:-1] + " \\\\\n" for row in rows)
    table = "\t\t" + mid_hline.join(table_rows)
    table_header = ("c|" * col_num)[:-1]
    if hline_style == "b":
        table_header = "|" + table_header + "|"
    if len(headers) == 0:
        table_header_names = " & " * (col_num - 1)
    elif len(headers) != 0:
        table_header_names = ""
        for header in headers:
            table_header_names += header + " & "
        table_header_names = table_header_names[:-3]
    if hline_style == "b":
        table_header_names = "\\hline\n\t\t" + table_header_names
        table = table + "\t\t\\hline\n"
    table = (
        "\\begin{table*}[h]\n\t\\centering\n\t\\caption{\\label{tab:X} }"
        "\n\t\\begin{tabular}{"
        + table_header
        + "}\n\t\t"
        + table_header_names
        + top_hline
        + table
        + "\t\\end{tabular}\n\\end{table*}"
    )
    print(table)
    f.close()


if __name__ == "__main__":
    print("This module is created by Hoyii.")
