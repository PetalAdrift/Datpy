import inspect
from os import remove
from turtle import color
from typing import List
from decimal import Decimal as D
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import doctest


def retrieve_name(var) -> str:
    """
    Returns the name of var from the out most frame inner-wards.
    Source: https://stackoverflow.com/questions/18425225/
    getting-the-name-of-a-variable-as-a-string

    >>> a = 10
    >>> retrieve_name(a)
    'a'
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name,
                 var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


def remove_exponent(d: D) -> D:
    """
    Returns d with exponents and trailing zeros removed.
    Source: https://docs.python.org/3/library/decimal.html#decimal-faq
    """
    return d.quantize(D(1)) if d == d.to_integral() else d.normalize()


def get_ceil(num: D, digit: int) -> D:
    """
    Returns the ceil of num with precision of digit.

    >>> a = D('1314')
    >>> get_ceil(a, 2)
    Decimal('1400')
    >>> b = D('0.0100')
    >>> get_ceil(b, -1)
    Decimal('0.1')
    """

    aug_num = np.ceil(num * D('10') ** D(-digit))
    return D(str(aug_num * D('10') ** D(str(digit))))


def get_floor(num: D, digit: int) -> D:
    """
    Returns the floor of num with precision of digit.

    >>> a = D('1314')
    >>> get_floor(a, 2)
    Decimal('1300')
    >>> b = D('0.0100')
    >>> get_floor(b, -1)
    Decimal('0.0')
    """

    aug_num = np.floor(num * D('10') ** D(-digit))
    return D(str(aug_num * D('10') ** D(str(digit))))


def get_round(num: D, digit: int) -> D:
    """
    Returns the rounded value of num with precision of digit.

    >>> a = D('1354')
    >>> get_round(a, 2)
    Decimal('1400')
    >>> b = D('0.0500')
    >>> get_floor(b, -1)
    Decimal('0.0')
    """

    aug_num = round(num * D('10') ** D(-digit))
    return D(str(aug_num * D('10') ** D(str(digit))))


def get_high_digit(num: D) -> int:
    """
    Returns the highest significant digit of num.

    >>> a = D('1314')
    >>> get_high_digit(a)
    3
    >>> b = D('0.0100')
    >>> get_high_digit(b)
    -2
    """

    return int(np.floor(np.log10(float(num))))


def get_low_digit(num: D, sig_fig: int) -> int:
    """
    Returns the lowest significant digit of num that has sig_fig significant
    figures.

    >>> a = D('1310')
    >>> get_low_digit(a, 3)
    1
    >>> b = D('0.0100')
    >>> get_low_digit(b, 3)
    -4
    """

    return get_high_digit(num) - sig_fig + 1


def get_high_digit_val(num: D) -> int:
    """
    Returns the number of the highest significant digit of num.


    >>> a = D('1314')
    >>> get_high_digit_val(a)
    1
    >>> b = D('0.0100')
    >>> get_high_digit_val(b)
    1
    """

    return int(num * D('10') ** D(-get_high_digit(num)))


class ExData():
    """
    Experimental data with uncertainty.
    """

    def __init__(self,  data: str = None, unc: str = None) -> None:
        """
        Creates a new ExData with value and uncertainty data.
        If data is left blank, returns a blank ExData.
        If unc is left blank, all digits are considered significant and
        uncertainty is taken to be the lowest digit.

        >>> length = ExData('1.00')
        >>> length.value
        Decimal('1.00')
        >>> length.sig_fig
        Decimal('3')
        >>> length.unc
        Decimal('0.01')
        >>> mass = ExData('13.010')
        >>> mass.value
        Decimal('13.010')
        >>> mass.sig_fig
        Decimal('5')
        >>> mass.unc
        Decimal('0.001')
        >>> time = ExData('5.21', '0.2')
        >>> time.value
        Decimal('5.21')
        >>> time.sig_fig
        Decimal('2')
        >>> time.unc
        Decimal('0.2')
        """

        if data is not None and unc is None:
            self.value = D(str(data))
            self.sig_fig = D(len(data.replace('.', '')))
            low_digit = get_low_digit(self.value, self.sig_fig)
            self.unc = D('10') ** D(str(low_digit))
        elif data is not None and unc is not None:
            self.value = D(str(data))
            self.sig_fig = D(get_high_digit(self.value) -
                             get_high_digit(unc) + 1)
            self.unc = D(unc)
        elif data is None:
            self.value = D('0')
            self.sig_fig = D('0')
            self.unc = D('0')

    def from_specified(data_value: str, data_sig_fig: int, data_unc: str) -> \
            'ExData':
        """
        Returns a new ExData with specified value, significant figures, and
        uncertainty as data_value, data_sig_fig, and data_unc respectively.

        >>> absurd_length = ExData.from_specified('123', 100, '0.001')
        >>> absurd_length.value
        Decimal('123')
        >>> absurd_length.sig_fig
        Decimal('100')
        >>> absurd_length.unc
        Decimal('0.001')
        """

        E = ExData()
        E.value = D(data_value)
        E.sig_fig = D(data_sig_fig)
        E.unc = D(data_unc)
        return E

    def __str__(self) -> str:
        """
        Returns a string representation of ExData.

        >>> length = ExData('1.230')
        >>> str(length)
        'length: value 1.230, sig. figs. 4, uncertainty 0.001'
        """

        return '{0}: value {1}, sig. figs. {2}, uncertainty {3}'.format(
            retrieve_name(self), str(self.value), str(self.sig_fig),
            str(self.unc))

    def __eq__(self, __o: object) -> bool:
        """
        Returns True if and only if the value being compared to is also ExData
        with the same value, sig_fig, and unc.

        >>> length_a = ExData('1.230')
        >>> length_b = ExData('1.23', '0.001')
        >>> length_a == length_b
        True
        """

        if type(__o) != type(self):
            return False
        if self.value == __o.value and self.sig_fig == __o.sig_fig and \
                self.unc == __o.unc:
            return True
        else:
            return False

    def __mul__(self, __o: object) -> 'ExData':
        """
        Returns a new ExData with factor.

        >>> reduced_length = ExData('1.0010')
        >>> actual_length = reduced_length * (10 ** 4)
        >>> actual_length.value
        Decimal('10010.0000')
        >>> actual_length.sig_fig
        Decimal('5')
        >>> actual_length.unc
        Decimal('1.0000')
        """

        if isinstance(__o, (int, float)):
            factor = D(__o)
            factor_value = self.value * factor
            factor_unc = self.unc * factor
            return ExData.from_specified(
                str(factor_value), self.sig_fig, str(factor_unc))


class EDLst(list):
    """
    A list of ExData.
    """

    def __init__(self, *data_set: 'ExData') -> None:
        """
        Creates a new ExDataLst from some ExData.

        >>> length_a = ExData('1.210')
        >>> length_b = ExData('1.301')
        >>> length_lst = EDLst(length_a, length_b)
        >>> length_lst[0].value
        Decimal('1.210')
        """

        list.__init__([])
        for data in data_set:
            self.append(data)

    def record(self, unc: float, *data_set: float) -> None:
        """
        Modifies self using data_set. The lowest significant figure is
        specified by unc.

        >>> length_lst = EDLst()
        >>> length_lst.record(0.02, 1.01, 1.30, 1.23, 1.40)
        >>> length_lst[1].value
        Decimal('1.3')
        >>> length_lst[1].sig_fig
        Decimal('3')
        >>> length_lst[1].unc
        Decimal('0.02')
        """

        for data in data_set:
            self.append(ExData.from_specified(
                str(data), get_high_digit(data) - get_high_digit(unc) + 1, str(unc)))

    def get_stats(self) -> 'ExData':
        """
        Returns a new ExData representing the mean of data_lst.

        >>> length_lst = EDLst()
        >>> length_lst.record(0.1, 1.0, 1.1, 1.2, 1.1)
        >>> mean_length = length_lst.get_stats()
        The mean of length_lst is 1.10; the uncertainty in the mean is 0.11.
        >>> mean_length.value
        Decimal('1.10')
        >>> mean_length.unc
        Decimal('0.11')
        >>> mean_length.sig_fig
        Decimal('3')
        """

        mean = sum(data.value for data in self) / D(len(self))
        unc_a = (sum((data.value - mean) ** D('2') for data in self) /
               D(len(self) - 1)) ** D('0.5') / (D(len(self))) ** D('0.5')
        unc_b = D(self[0].unc)
        unc = (unc_a ** D('2') + unc_b ** D('2')) ** D('0.5')

        if get_high_digit_val(unc) != 1:
            u_sig_fig = 1
        else:
            u_sig_fig = 2
        unc = get_ceil(unc, get_high_digit(unc) - u_sig_fig + 1)
        
        mean = get_round(mean, get_low_digit(unc, u_sig_fig))
        sig_fig = get_high_digit(mean) - get_high_digit(unc) + u_sig_fig
        print('The mean of {0} is {1}; the uncertainty in the mean is {2}.'.format(
            retrieve_name(self), str(mean), str(unc)))
        return ExData.from_specified(str(mean), sig_fig, str(unc))

    def u_prop(self, func: str, var_names: str = 'x y z a b c d') -> 'ExData':
        """
        Returns a new ExData representing evaluation of function func
        with variables in self. Variables in self have symbols in var_names
        assigned to them respectively, and the same symbols are used in func.


        >>> kinetic_energy = EDLst()
        >>> dens = ExData.from_specified('1.5', 2, '0.2')
        >>> vol = ExData.from_specified('1.3', 2, '0.1')
        >>> vel = ExData.from_specified('2.7', 2, '0.1')
        >>> kinetic_energy.extend([dens, vol, vel])
        >>> kinetic_energy = kinetic_energy.u_prop('0.5 * x * y * z ** 2')
        The value of kinetic_energy is 7.1; the uncertainty is 1.3.
        >>> kinetic_energy.value
        Decimal('7.1')
        >>> kinetic_energy.unc
        Decimal('1.3')
        >>> kinetic_energy.sig_fig
        Decimal('2')
        """

        expr = sp.parsing.sympy_parser.parse_expr(func)
        subs_dict = {}
        for i in range(len(self)):
            subs_dict[sp.symbols(var_names[2 * i: 2 * i + 1])] = self[i].value
        value = expr.subs(subs_dict)
        unc_square = D(0)
        for i in range(len(self)):
            unc_square += (D(str(sp.diff(expr, sp.symbols(
                var_names[2 * i: 2 * i + 1])).subs(subs_dict)))
                * self[i].unc) ** D('2')
        unc = unc_square ** D('0.5')
        if get_high_digit_val(unc) != 1:
            u_sig_fig = 1
        else:
            u_sig_fig = 2
        unc = get_ceil(unc, get_high_digit(unc) - u_sig_fig + 1)
        value = get_round(value, get_low_digit(unc, u_sig_fig))
        sig_fig = get_high_digit(value) - get_high_digit(unc) + u_sig_fig
        value = remove_exponent(value)
        unc = remove_exponent(unc)
        print('The value of {0} is {1}; the uncertainty is {2}.'.format(
            retrieve_name(self), value, unc))
        return ExData.from_specified(str(value), sig_fig, str(unc))


def create_graph(var_x_lst: EDLst, var_y_lst: EDLst, g_title: str) -> None:
    """
    Creates a graph from var_x_lst and var_y_lst with title."""
    # fig = plt.figure(tight_layout=True)
    x_max = -np.inf
    x_min = np.inf
    y_max = -np.inf
    y_min = np.inf
    for i in range(len(var_x_lst)):
        x_val = var_x_lst[i].value
        y_val = var_y_lst[i].value
        x_unc = var_x_lst[i].unc
        y_unc = var_y_lst[i].unc
        x_max = max(x_max, x_val + x_unc)
        x_min = min(x_min, x_val - x_unc)
        y_max = max(y_max, y_val + y_unc)
        y_min = min(y_min, y_val - y_unc)
        plt.scatter(x_val, y_val, 3, color='#000000')
        unc_box_x = [x_val - x_unc, x_val + x_unc,
                     x_val + x_unc, x_val - x_unc, x_val - x_unc]
        unc_box_y = [y_val + y_unc, y_val + y_unc,
                     y_val - y_unc, y_val - y_unc, y_val + y_unc]
        plt.plot(unc_box_x, unc_box_y, color='#ffaadd')
    plt.xlim([x_min - (x_max - x_min) / 8, x_max + (x_max - x_min) / 8])
    plt.ylim([y_min - (y_max - y_min) / 8, y_max + (y_max - y_min) / 8])
    plt.xlabel(retrieve_name(var_x_lst))
    plt.ylabel(retrieve_name(var_y_lst))
    plt.title(g_title)
    plt.show()


if __name__ == '__main__':
    doctest.testmod()
    print('Hello world! I am 方曼宜 (Fang Manyi), your lab partner for today.')
