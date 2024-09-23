import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import math
from decimal import Decimal as D_DEC
import numpy as np
import sympy as sp
from scipy.optimize import curve_fit
from scipy import odr
import scipy.stats as ss
from inspect import signature


HEADING_FONT_SIZE = 12
LABEL_FONT_SIZE = 12
BODY_FONT_SIZE = 10
PLOT_SIZE = 3
CAP_SIZE = 2
EXTRAPOLATION_LOWER_RANGE = 0.2
EXTRAPOLATION_UPPER_RANGE = 0.2
PLOT_STEP_SCALE = 0.0025
FIGURE_MARGIN = 0.05
COLOR_PALETTE = ["#EE5515", "#65CB4D", "#446688", "#663377"]
COLOR_NUM = 0


def D(num) -> D_DEC:
    """
    Returns a given number in Decimal type.

    :param num: The number to be converted.
    :type num: Any (number-like)
    :return: The given number in Decimal type.
    :rtype: Decimal
    """
    return D_DEC(float(num))


def get_params_count(func: callable) -> int:
    """
    Returns the number of parameters accepted by a function.

    :param func: The function to be considered.
    :type func: callable
    :return: The number of parameters accepted by the function.
    :rtype: int
    """
    sig = signature(func)
    params = sig.parameters
    return len(params)


def set_ex_range(lower_range: float, upper_range: float) -> None:
    """
    Modifies global variables for extrapolation range.

    :param lower_range: The new scale for extrapolation below.
    :param upper_range: The new scale for extrapolation above.
    :type lower_range: float
    :type upper_range: float
    """
    global EXTRAPOLATION_LOWER_RANGE  # access the global variable
    global EXTRAPOLATION_UPPER_RANGE
    EXTRAPOLATION_LOWER_RANGE = lower_range
    EXTRAPOLATION_UPPER_RANGE = upper_range


def set_plot_step(proportion: float) -> None:
    """
    Modifies the global variable for the scale used for plot steps.

    :param proportion: The proportion of range that each step takes up.
        If proportion is greater than 1, treat it as the number of steps
        to be used rather than the proportion.
    :type proportion: float
    """
    global PLOT_STEP_SCALE
    if proportion > 1:
        proportion = 1 / int(proportion)
    PLOT_STEP_SCALE = proportion


def set_margin(margin: float) -> None:
    """
    Modifies the global variable for the margin scale used in figures.

    :param margin: The margin scale used for limits of x-axis in figures.
    :type margin: float
    """
    global FIGURE_MARGIN
    FIGURE_MARGIN = margin


def get_color() -> str:
    """
    Returns the next color from the global variable COLOR_PALETTE.

    :return: The next color in the color palette.
    :rtype: str
    """
    global COLOR_NUM
    color = COLOR_PALETTE[COLOR_NUM % len(COLOR_PALETTE)]
    COLOR_NUM += 1
    return color


def remove_legend(fig: "matplotlib.figure.Figure", ax: int) -> None:
    """
    Removes the legend from an axis of a figure.

    :param fig: The figure from which the axis is chosen.
    :param a
    :type fig: matplotlib.figure.Figure
    :type ax: int
    """
    ax = fig.get_axes()[ax]
    current_leg = ax.get_legend()
    if current_leg is not None:
        current_leg.remove()


def get_gauss(x: float, amp: float, mu: float, sigma: float, base: float) -> float:
    """
    Returns the value of a specified Gaussian distribution.

    :param x: The position at which the Gaussian is evaluated.
    :param amp: The amplitude of the Gaussian.
    :param mu: The mean of the Gaussian.
    :param sigma: The standard deviation of the Gaussian.
    :param base: The baseline height of the Gaussian.
    :type x: float
    :type amp: float
    :type mu: float
    :type sigma: float
    :type base: float
    :return: The value of the Gaussian at x.
    :rtype: float
    """
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + base


def get_round(num: D_DEC, digit: int) -> D_DEC:
    """
    Returns the number rounded to a specified precision.

    :param num: The number to be rounded.
    :param digit: The number of places after the decimal point to be kept.
        Rounded to 10^(-digit) when non-positive.
    :type num: D_DEC
    :type digit: int
    :return: The rounded result.
    :rtype: D_DEC
    """
    aug_num = round(num * D("10") ** (-D(str(digit))))
    aug_num = aug_num * D("10") ** D(str(digit))
    return aug_num


def get_high_digit(num: D_DEC) -> int:
    """
    Returns the highest significant digit of the number.

    :param num: The nubmer to be considered.
    :type num: D_DEC
    :return: The highest significant digit of the number in power of 10.
    :rtype: int
    """
    return int(np.floor(np.log10(float(num))))


def keep_digits(num: D_DEC, digit: int) -> D_DEC:
    """
    Returns the number rounded to a specified number of significant digits.

    :param num: The number to be rounded.
    :param digit: The number of significant digits kept.
    :type num: D_DEC
    :type digit: int
    :return: The rounded number.
    :rtype: D_DEC
    """
    return get_round(num, get_high_digit(num) - digit + 1)


def load_data(
    file_name: str,
    body=(slice(1, None), slice(None)),
    orientation="col",
    delimiter=",",
    conv_type="float",
) -> list[list]:
    """
    Reads data stored in a .csv-like file.

    :param file_name: The path of the file being read.
    :param body: The selection of data being read.
        Defaults to (slice(1, None), slice(None)).
    :param orientation: The orientation in which a class of data is stored.
        Defaults to "col".
    :param delimiter: The delimiter used to separate values in the file.
        Defaults to ",".
    :param conv_type: The type of the data being read.
        Defaults to "float".
    :type file_name: str
    :type body: (slice, slice)
    :type orientation: str
    :type delimiter: str
    :type conv_type: str
    :return: The read values stored as a nested list with inner lists containing
        data of a same class.
    :rtype: [list[Any]]
    """
    data = np.loadtxt(
        file_name,
        dtype="str",
        comments=None,
        delimiter=delimiter,
        skiprows=0,
        encoding="utf-8",
    )
    data = data[body[0], body[1]]
    data = np.array(data, dtype=conv_type)
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
) -> [float, float, float, list[float]]:
    """
    Returns chi-squared statistics calculated from given data.

    :param val_data: The values of data.
    :param val_predicted: The values of data obtained from fits to functions.
    :param val_err: The uncertainties in the data.
    :param para_amount: The number of free parameters in functions.
    :type val_data: list[float]
    :type val_predicted: list[float]
    :type val_err: list[float]
    :type para_amount: int
    :return:
        - chi-squared value
        - reduced chi-squared value
        - chi-squared probability of data folloiwng the function
        - scaled uncertainties which normalize the reduced chi-squared value
    :rtype:
        - float
        - float
        - float
        - list[float]
    """
    val_data = np.array(val_data)
    val_predicted = np.array(val_predicted)
    val_err = np.array(val_err)

    degree_of_freedom = len(val_data) - para_amount
    chi_squared = np.sum(((val_data - val_predicted) / val_err) ** 2)
    red_chi_squared = chi_squared / degree_of_freedom
    chi_squared_prob = 1 - ss.chi2.cdf(chi_squared, degree_of_freedom)
    unc_norm = red_chi_squared**0.5 * val_err
    return (chi_squared, red_chi_squared, chi_squared_prob, unc_norm)


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
) -> (list[float], list[float], list[float], list[[float, float]]):
    """
    Fits given data to a function and returns optimized parameters from NLS fit.

    :param y_vals: The values of the dependent variable.
    :param y_errs: The uncertainties in the values of the dependent variable.
    :param x_vals: The values of the indepedent variable.
    :param expr: The string formulation of the function used for fitting.
    :param para_amount: The number of parameters used in the function,
        including the independent variable.
    :param para_names: The names of parameters used in the function.
        Defaults to "x a b c d m n p q".
    :param guess: The initial guess provided to NLS.
        Defaults to None.
    :param sp_expr: A fallback expression used when the function formulation
        fails for sympy.parsing in evaluation of the function.
        Defaults to "".
    :param fit_func: A fallback function used when sympy.lambdify and
        sympy.parsing fail.
        Defaults to None.
    :param max_fev: The maximum number of calls to the function.
        Defaults to 800.
    :type y_vals: list[float]
    :type y_errs: list[float]
    :type x_vals: list[float]
    :type expr: str
    :type para_amount: int
    :type para_names: str
    :type guess: list[float] or None
    :type sp_expr: str
    :type fit_func: callable or None
    :type max_fev: int
    :return:
        - optimized parameters for the fit function
        - uncertainty in the optimized parameters
        - evaluation of the dependent variable based on fit results
        - input-output pairs evaluated from function over a range for plotting
    :rtype:
        - list[float]
        - list[float]
        - list[float]
        - list[[float, float]]
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
            max(x_vals)
            + EXTRAPOLATION_UPPER_RANGE * x_range
            + PLOT_STEP_SCALE * x_range,
            PLOT_STEP_SCALE * x_range,
        ):
            subs_dict[var_names[0]] = x_val
            plot_predicted.append([x_val, expr.subs(subs_dict)])
    elif fit_func is not None:
        for x_val in np.arange(
            min(x_vals) - EXTRAPOLATION_LOWER_RANGE * x_range,
            max(x_vals)
            + EXTRAPOLATION_UPPER_RANGE * x_range
            + PLOT_STEP_SCALE * x_range,
            PLOT_STEP_SCALE * x_range,
        ):
            params = [x_val]
            params.extend(popt)
            plot_predicted.append([x_val, fit_func(*params)])

    y_predicted = np.array(y_predicted, dtype=float)
    plot_predicted = np.array(plot_predicted, dtype=float)

    return (popt, pstd, y_predicted, plot_predicted)


def get_best_fit_odr(
    y_vals: list[float],
    y_errs: list[float],
    x_vals: list[float],
    x_errs: list[float],
    expr: str,
    para_amount: int,
    para_names="x a b c d m n p q",
    guess=None,
    sp_expr="",
    fit_func=None,
    max_iter=50,
) -> (
    list[float],
    list[float],
    list[list[float]],
    float,
    float,
    float,
    list[float],
    list[[float, float]],
):
    """
    Fits given data to a function and returns optimized parameters from ODR fit.

    :param y_vals: The values of the dependent variable.
    :param y_errs: The uncertainties in the values of the dependent variable.
    :param x_vals: The values of the indepedent variable.
    :param x_errs: The uncertainties in the values of the independent variable.
    :param expr: The string formulation of the function used for fitting.
    :param para_amount: The number of parameters used in the function,
        including the independent variable.
    :param para_names: The names of parameters used in the function.
        Defaults to "x a b c d m n p q".
    :param guess: The initial guess provided to NLS.
        Defaults to None.
    :param sp_expr: A fallback expression used when the function formulation
        fails for sympy.parsing in evaluation of the function.
        Defaults to "".
    :param fit_func: A fallback function used when sympy.lambdify and
        sympy.parsing fail. This should be of the form f([param], x).
        Defaults to None.
    :param max_iters: The maximum number of iterations.
        Defaults to 50.
    :type y_vals: list[float]
    :type y_errs: list[float]
    :type x_vals: list[float]
    :type x_errs: list[float]
    :type expr: str
    :type para_amount: int
    :type para_names: str
    :type guess: list[float] or None
    :type sp_expr: str
    :type fit_func: callable or None
    :type max_iter: int
    :return:
        - optimized parameters for the fit function
        - uncertainty in the optimized parameters
        - covariance matrix of the optimized parameters
        - quasi-chi-squared value from the fit
        - reduced chi-squared value from the fit
        - chi-squared probability of the fit
        - evaluation of the dependent variable based on fit results
        - input-output pairs evaluated from function over a range for plotting
    :rtype:
        - list[float]
        - list[float]
        - list[list[float]]
        - float
        - float
        - float
        - list[float]
        - list[[float, float]]
    """
    var_names = sp.symbols(para_names)

    if fit_func is None:
        func = sp.lambdify([var_names[1:para_amount], var_names[0]], expr)
    elif fit_func is not None:
        func = fit_func

    data = odr.RealData(x_vals, y_vals, sx=x_errs, sy=y_errs)
    model = odr.Model(func)

    if guess is None:
        # assume all 0's for initial guess when not provided
        guess = [0] * (para_amount - 1)

    odr_instance = odr.ODR(data, model, beta0=guess, maxit=max_iter)
    output = odr_instance.run()
    popt = output.beta  # optimal parameters
    pstd = output.sd_beta  # std dev of popt
    pcov = output.cov_beta  # covariance matrix of popt
    # "quasi chi^2" converges to the regular chi^2 for small x uncertainties
    quasi_chi_squared = output.res_var

    degree_of_freedom = len(x_vals) - para_amount + 1  # +1 to account for x
    red_chi_squared = quasi_chi_squared / degree_of_freedom
    chi_squared_prob = 1 - ss.chi2.cdf(quasi_chi_squared, degree_of_freedom)

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
            params = [popt, x_val]
            y_predicted.append(fit_func(*params))

    plot_predicted = []
    x_range = max(x_vals) - min(x_vals)
    if fit_func is None:
        for x_val in np.arange(
            min(x_vals) - EXTRAPOLATION_LOWER_RANGE * x_range,
            max(x_vals)
            + EXTRAPOLATION_UPPER_RANGE * x_range
            + PLOT_STEP_SCALE * x_range,
            PLOT_STEP_SCALE * x_range,
        ):
            subs_dict[var_names[0]] = x_val
            plot_predicted.append([x_val, expr.subs(subs_dict)])
    elif fit_func is not None:
        for x_val in np.arange(
            min(x_vals) - EXTRAPOLATION_LOWER_RANGE * x_range,
            max(x_vals)
            + EXTRAPOLATION_UPPER_RANGE * x_range
            + PLOT_STEP_SCALE * x_range,
            PLOT_STEP_SCALE * x_range,
        ):
            params = [popt, x_val]
            plot_predicted.append([x_val, fit_func(*params)])

    y_predicted = np.array(y_predicted, dtype=float)
    plot_predicted = np.array(plot_predicted, dtype=float)

    return (
        popt,
        pstd,
        pcov,
        quasi_chi_squared,
        red_chi_squared,
        chi_squared_prob,
        y_predicted,
        plot_predicted,
    )


def propagate_uncertainty(
    func: str,
    var_vl: list[list[float]],
    var_unc: list[list[float]],
    var_names="x y z a b c d",
) -> list[float]:
    """
    Returns uncertainties propagated through a function.

    :param func: The string formulation of the function used.
    :param var_vl: A nested list of variable values with inner lists containing
        data of a same class.
    :param var_unc: A nested list of uncertainties with inner lists containing
        uncertainties in data of a same class.
    :param var_names: Variables names used in the function. The order of
        variables used must correspond to the order of names used here.
        Defaults to "x y z a b c d".
    :type func: str
    :type var_vl: list[list[float]]
    :type var_unc: list[list[float]]
    :type var_names: str
    :return: Uncertainties in the function values.
    :rtype: list[float]
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
    Adds a simple plot to an existing axis of a figure.

    :param fig: The figure to which the plot is added.
    :param x_vals: The x values.
    :param y_vals: The y values.
    :param x_errs: The x uncertainties.
    :param y_errs: The y uncertainties.
    :param leg: The legend label.
    :param plot_data: Whether or not to join data points with lines.
        Defaults to False.
    :param ax: The id of the axis in the figure.
        Defaults to -1.
    :type fig: matplotlib.figure.Figure
    :type x_vals: list[float]
    :type y_vals: list[float]
    :type x_errs: list[float]
    :type y_errs: list[float]
    :type leg: str
    :type plot_data: bool
    :type ax: int
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
    plot_predicted: list[[float, float]],
    x_errs: list[float],
    y_errs: list[float],
    x_label: str,
    y_label: str,
    title: str,
    scale="linear",
    layout_reversed=False,
    condensed=False,
    data_leg="data",
    pred_leg="prediction",
    remove_errs=(False, False, False),
    plot_data=False,
) -> float:
    """
    Creates a new axis with plots and residue plots in a figure.

    :param fig: The figure in which the axis is created.
    :param pos: The position of the axis.
    :param x_vals: The x values.
    :param y_vals: The y values.
    :param y_predicted: The predicted y values from a function of x values.
    :param plot_predicted: The input-output pairs of a prediction function.
    :param x_errs: The x uncertainties.
    :param y_errs: The y uncertainties.
    :param x_label: The x-axis label of the plot.
    :param y_label: The y-axis label of the plot.
    :param title: The title of the plot.
    :param scale: The scale used on the y-axis.
        Defaults to "linear".
    :param layout_reversed: Whether or not to put the residue plot at the top.
        Defaults to False.
    :param condensed: Whether or not to condense the space between the regular
        plot and the residue plot.
        Defaults to False.
    :param data_leg: The legend label for the data points.
        Defaults to "data".
    :param pred_leg: The legend label for the prediction function.
        Defaults to "prediction".
    :param remove_errs: Whether or not to remove the error bars for the
        x-axis, y-axis in regular plot, and y-axis in residue plot.
        Defaults to (False, False, False).
    :param plot_data: Whether or not to plot data points connected with lines
        instead of using points with error bars.
        Defaults to False.
    :type fig: matplotlib.figure.Figure
    :type pos: int
    :type x_vals: list[float]
    :type y_vals: list[float]
    :type y_predicted: list[float]
    :type plot_predicted: list[[float, float]]
    :type x_errs: list[float]
    :type y_errs: list[float]
    :type x_label: str
    :type y_label: str
    :type title: str
    :type scale: str
    :type layout_reversed: bool
    :type condensed: bool
    :type data_leg: str
    :type pred_leg: str
    :type remove_errs: (bool, bool, bool)
    :type plot_data: bool
    :return: The percentage (in decimals) of points where the y-residual falls
        within the y uncertainty. This value is set to -1 if the percentage
        cannot be calculated due to a lack of prediction or y uncertainties.
    :rtype: float
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
    if (x_errs is not None) and (not remove_errs[0]):
        ax.errorbar(
            x_vals, y_vals, xerr=x_errs, fmt="none", lw=1, capsize=CAP_SIZE, c="#66CCFF"
        )
    if (y_errs is not None) and (not remove_errs[1]):
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

    if layout_reversed or len(y_predicted) == 0:
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
        if not layout_reversed:
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
        if (y_errs is not None) and (not remove_errs[2]):
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
        if not layout_reversed:
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
    Legacy code.
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
    vals: list[float],
    bins_num: int,
    x_label: str,
    y_label: str,
    title: str,
    bins_width=0.0,
    bins_range=(0.0, -1.0),
    uncertainties=False,
    distribution=(None, ""),
    guess=None,
    forced=False,
) -> (list[int], list[float], list[float]):
    """
    Creates a new axis with histograms in a figure.

    :param fig: The figure in which the axis is created.
    :param pos: The position of the axis.
    :param vals: The values to be binned.
    :param bins_num: The number of bins used. This is overriden by bins_windth
        when it is nonzero.
    :param x_label: The x-axis label of the histogram.
    :param y_label: The y_axis label of the histogram.
    :param title: The title of the histogram.
    :param bins_width: The widths of the bins. When this is nonzero and
        bins_range is an empty interval, this determines the number of bins
        to use and the range of bins to plot by considering the range of values.
        Defaults to 0.0.
    :param bins_range: The range of the bins. When this is a nonempty interval,
        this determines the bins along with bins_num.
        Defaults to (0, -1).
    :param uncertainties: Whether or not to add error bars to each bin.
        Defaults to False.
    :param distribution: The tuple containing the function and name of the
        distribution used to fit the binned data.
        Defaults to (None, "").
    :param guess: The initial guesses for the distribution function.
        Defaults to None.
    :param forced: Whether or not to force the guess as the optimized parameters
        without actually conducting a NLS fit.
        Defaults to False.
    :type fig: matplotlib.figure.Figure
    :type pos: int
    :type vals: list[float]
    :type bins_num: int
    :type x_label: str
    :type y_label: str
    :type title: str
    :type bins_width: float
    :type bins_range: (float, float)
    :type uncertainties: bool
    :type distribution: (callable, str) or (None, str)
    :type guess: list[float] or None
    :type forced: bool
    :return:
        - value of bins
        - uncertainty of value of bins
        - center of bins
        - optimized paramters of the distribution (if provided distribution)
        - uncertainty in the optimized parameters (if provided distribution)
        - chi-squared statistics (if provided distribution)
    :rtype:
        - list[int]
        - list[float]
        - list[float]
        - list[float] (if provided distribution)
        - list[float] (if provided distribution)
        - [float, float, float, list[float]] (if provided distribution)
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
    vals_bin, bin_edges, _ = ax.hist(
        vals, bins=bins_num, range=ran, color="#66CCFF", histtype="bar", rwidth=0.9
    )
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    unc_vals_bin = np.sqrt(vals_bin)
    unc_vals_bin = np.where(unc_vals_bin == 0, 1, unc_vals_bin)
    if uncertainties:
        ax.errorbar(
            bin_centers,
            vals_bin,
            yerr=unc_vals_bin,
            fmt="none",
            capsize=CAP_SIZE,
            c="k",
        )
    ax.set_xlabel(x_label, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=HEADING_FONT_SIZE)
    if distribution[0] is None:
        return (vals_bin, unc_vals_bin, bin_centers)
    if not forced:
        popt, pstd, pred, plot = get_best_fit(
            vals_bin,
            unc_vals_bin,
            bin_centers,
            "",
            5,
            guess=guess,
            fit_func=distribution[0],
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
    chi = get_chi_squared(
        vals_bin, pred, unc_vals_bin, get_params_count(distribution[0]) - 1
    )
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
    return (vals_bin, unc_vals_bin, bin_centers, popt, pstd, chi)


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


if __name__ == "__main__":
    print("This module is created by Hoyii.")
