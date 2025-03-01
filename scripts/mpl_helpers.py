# This file is modified from the mpl_tools.py file from William Black:
# https://bitbucket.org/wkblack/dendro-gr-analysis-tools/src/master/mpl_tools.py

# set up matplotlib and tools to aid it


########################################################################
# set up golden aspect ratio
import numpy as np

ones = np.ones(2)  # handly little shortcut

golden_ratio = (1 + np.sqrt(5)) / 2.0
golden_aspect = 5 * np.array([golden_ratio, 1])
double_height = np.array([1, 2]) * golden_aspect
double_width = np.array([2, 1]) * golden_aspect
aspect_square = ones * 6.374
aspect_rect = np.array([940 / 900.0, 1]) * 6.374


########################################################################
# set up pyplot defaults
import matplotlib as mpl
from matplotlib import pyplot as plt

plt.rcParams["errorbar.capsize"] = 2
plt.rcParams["scatter.edgecolors"] = "none"
plt.rcParams["image.cmap"] = "cividis"
plt.rcParams["font.size"] = 15
# parameters for saving the figure
plt.rcParams["figure.autolayout"] = "True"
plt.rcParams["figure.figsize"] = golden_aspect
plt.rcParams["savefig.dpi"] = 1000
plt.rcParams["savefig.format"] = "jpg"

# grab color methods & data
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap as LSC

# taken directly from Seaborn's source code here: https://github.com/mwaskom/seaborn/blob/master/seaborn/palettes.py#L21
seaborn_deep = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
]
seaborn_dark = [
    "#001C7F",
    "#B1400D",
    "#12711C",
    "#8C0800",
    "#591E71",
    "#592F0D",
    "#A23582",
    "#3C3C3C",
    "#B8850A",
    "#006374",
]
seaborn_dark6 = ["#001C7F", "#12711C", "#8C0800", "#591E71", "#B8850A", "#006374"]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=seaborn_dark)
mpl_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# axes tools
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable


def color_vp(vp, col):
    for key in vp.keys():
        if key == "bodies":  # fix coloring of transparencies
            vp[key][0].set_facecolor(col)
        else:  # fix coloring of various lines
            vp[key].set_color(col)


########################################################################
# other tools / shortcuts

# returns first iteration of arithmeticâ€“geometric mean of `a` and `b`
agm = lambda a, b: [(a + b) / 2.0, np.sign(a) * np.sqrt(a * b)]
# returns more iterations than are necessary of the agm to converge
AGM = lambda a, b: np.mean(agm(*agm(*agm(*agm(*agm(*agm(*agm(a, b))))))))
# returns mean and error of the mean on a variable x
mnerr = lambda x: [np.mean(x), np.std(x) / np.sqrt(len(x))]
# variance wrapped in single function for bootstrapping purposes
variance = lambda x: np.std(x) ** 2
# commonly paired functions, e.g. `mn,std = [f(x,axis=0) for f in mnst]`
mnst = [np.mean, np.std]
nanmnst = [np.nanmean, np.nanstd]


def print_pars(popt, perr=None, lbls=None, units=None, N_sig=False):
    """
    print_pars(popt,perr,lbls=None,N_sig=False)
    print pretty parameters with errors in LaTeX style
    """
    if perr is None:  # usually required; assume it's in first argument
        popt, perr = popt
    if not hasattr(popt, "__len__"):  # force into array for easy handling
        popt = np.array([popt])
        perr = np.array([perr])
    if len(np.shape(perr)) == 2:  # assume covariance matrix input
        perr = np.sqrt(np.diag(perr))
    N_pars = len(popt)
    if lbls is None:
        # tmp = r'$p_{%i} = %g \pm %g$'
        lbls = [f"p[{ii}]" for ii in range(N_pars)]
    else:
        if np.sum(np.shape(lbls)) == 0:
            lbls = np.repeat(lbls, N_pars)
    if units is None:
        units = np.repeat("", N_pars)
    else:
        if isinstance(units, str):
            units = np.repeat(units, N_pars)
        elif len(units) != N_pars:
            units = np.repeat(units, N_pars)
        else:
            pass  # should already be formatted correctly if here.
    # print out
    for ii in range(N_pars):
        if (np.abs(perr[ii]) == 0) + (np.isinf(perr[ii])) + (np.isnan(perr[ii])):
            N = 100  # just return the number w/o rounding
        else:
            N = int(np.round(1.5 - np.log10(np.abs(perr[ii]))))
        # format report
        left = np.round(popt[ii], N)
        right = np.round(perr[ii], N)
        if N < 1:  # remove trailing ".0" by converting to int.
            left = int(left)
        report = f"{lbls[ii]} = {left} \pm {right}"
        if len(units[0]) > 0:
            report += f" {units[ii]}"
        if N_sig:
            report += f" ({popt[ii]/perr[ii]:0.2g})"
        print(report)


def get_polyfit_err_simple(xv, zf, V):
    if not hasattr(xv, "__len__"):
        xv = np.array([xv])
    else:
        xv = np.array(xv)
    deg = len(zf) - 1
    X = np.array(
        [
            [xv ** (deg - ii) * xv ** (deg - jj) for ii in range(deg + 1)]
            for jj in range(deg + 1)
        ]
    )
    err = np.sqrt(np.sum(X * V[:, :, np.newaxis], axis=(0, 1)))
    return err


def get_polyfit_err(xv, zf, V, m=0):
    """
    get_polyfit_err(xv,zf,V,m=0)
    inputs:
     > xv: x-values at which to evalutate
     > zf: polynomial's coefficients in decreasing powers
     > V: coefficient's covariance matrix
     > m=0: order of derivative to evaluate
    """
    if not hasattr(xv, "__len__"):
        xv = np.array([xv])
    else:
        xv = np.array(xv)
    deg = len(zf) - 1
    # check whether this can all be skipped
    if m > deg:
        return 0 * xv

    def get_deriv_coeff(deg, m=0):
        val = np.flip(np.arange(deg + 1)).astype(int)
        report = np.ones(deg + 1).astype(int)
        for ii in range(m):
            report *= val - ii
        return report

    coeff = get_deriv_coeff(deg, m)

    X = np.array(
        [
            [
                xv ** (deg - ii - m) * xv ** (deg - jj - m) * coeff[ii] * coeff[jj]
                for ii in range(deg + 1)
            ]
            for jj in range(deg + 1)
        ]
    )
    err = np.sqrt(np.sum(X * V[:, :, np.newaxis], axis=(0, 1)))
    return err


def get_quant(x):
    """
    for each value in x, return which quantile corresponds to it
    Note: if multiple instances of a value is in the input array,
          the report defaults to the lowest corresponding quantile,
          so there may be slight discrepancy between this and np.quantile
    """
    return np.array([np.count_nonzero(x < x_i) / (len(x) - 1) for x_i in x])


def get_boot_quant(x, N_boot=50):
    """
    for each value in x, return quantile spread according to bootstrap
    """
    L = len(x)
    xv, yv = np.zeros((2, L * N_boot))
    for ii in range(N_boot):
        M = np.random.choice(L, L)
        xv[ii * L : (ii + 1) * L] = x[M]
        yv[ii * L : (ii + 1) * L] = get_quant(x[M])

    xx = np.array(sorted(set(x)))
    yy = np.array([np.mean(yv[xv == xi]) for xi in xx])
    yy_err = np.array([np.std(yv[xv == xi]) for xi in xx])
    return xx, yy, yy_err


def boot_fun(fun, x, N_boot=50, **kwargs):
    "return bootstrap uncertainties of fun(x,**kwargs)"
    x = np.array(x)
    xv0 = fun(x, **kwargs)  # use to get shape of output
    if hasattr(xv0, "__len__"):
        xv = np.zeros((N_boot, len(xv0)))
    else:
        xv = np.zeros(N_boot)
    L = len(x)
    for ii in range(N_boot):
        xv[ii] = fun(x[np.random.choice(L, L)], **kwargs)
    return [f(xv, axis=0) for f in mnst]


def jack_fun(fun, x, **kwargs):
    "return jackknife uncertainties of fun(x,**kwargs)"
    L = len(x)
    report = np.zeros(L)
    for ii in range(L):
        report[ii] = fun(np.delete(x, ii), **kwargs)
    return [f(report) for f in mnst]


def mid(x):
    "return an array of the midpoints of the input"
    return np.array(x[1:] + x[:-1]) / 2.0


def avg_neighbors(arr):
    "return an array averaging the neighbors of the input"
    report = np.copy(arr)
    report[:-1] = np.nansum(np.dstack((report[:-1], arr[1:])), 2)
    report[1:] = np.nansum(np.dstack((report[1:], arr[:-1])), 2)
    div = 3 * np.ones(len(arr))
    div[0] = div[-1] = 2
    return report / div


def find_peak(arr):
    """
    find the highest point of an array
    if array has plateau, return value closest to its center
    """
    if len(set(arr)) == 1:  # all identical
        return 0  # leftmost value
    idxs = np.argwhere(arr == np.max(arr)).T[0]
    if len(idxs) > 2:
        arr_new = avg_neighbors(arr)
        return find_peak(arr_new)
    else:
        return np.min(idxs)


def is_jagged(x):
    "returns boolean as to whether a /\/ trend exists in input"
    signs = np.sign(np.diff(x))
    for ii in range(len(signs) - 2):
        p = np.all([-1, +1, -1] == -signs[ii : ii + 3])
        m = np.all([-1, +1, -1] == +signs[ii : ii + 3])
        if p or m:
            return True
    return False


def smooth(x):
    "returns an array without any /\/ trends, smoothed out"
    report = np.copy(x)
    while is_jagged(report):
        report = avg_neighbors(report)
    return report


def mask_overwrite(t):
    """
    returns mask which overwrites overwritten values,
    e.g. turning [0,1,2,1,2,...][mask] -> [0,1,2,...]
    """
    mask = np.ones_like(t)
    for ii in range(len(t) - 1):
        new = t[ii + 1]
        jj = 0
        while new <= t[ii - jj]:
            mask[ii - jj] = 0
            jj += 1
    # return as boolean array
    return mask.astype(bool)


def safe_divide_by(x, eps=1e-10):
    """
    input `x`, return essentially 1/(x + eps),
    but account for potentially negative x values
    """
    sgn = np.sign(x)
    if hasattr(sgn, "__len__"):
        sgn[sgn == 0] = 1
    else:  # scalar
        if sgn == 0:
            sgn = 1
    return 1 / (x + sgn * eps)


def get_Scott(data):
    """
    return parameters for Scott's normal reference rule
    returns (h,k), where
    h = bin width
    k = number of bins, spanning min to max of data
    """
    h = 3.49 * np.nanstd(data) / len(data) ** (1 / 3)
    k = np.ceil((np.nanmax(data) - np.nanmin(data)) / h).astype(int)
    return h, k


########################################################################
# string helper functions

import os  # to remove common prefixes


def remove_common_prefix(strings):
    "remove a common prefix among a list of strings"
    # Find the common prefix of the strings
    prefix = os.path.commonprefix(strings)
    # Remove the common prefix from each string
    result_list = [string[len(prefix) :] for string in strings]
    return result_list


def remove_common_suffix(strings):
    "remove a common suffix among a list of strings"
    # Find the common suffix of the strings
    suffix = os.path.commonprefix([string[::-1] for string in strings])[::-1]
    # Remove the common suffix from each string
    result_list = [string[: -len(suffix)] for string in strings]
    return result_list


def remove_common_parts(strings):
    """
    given a list of strings potentially sharing prefixes and suffixes,
    remove those common parts and return the cleaned list of strings.
    """
    return remove_common_prefix(remove_common_suffix(strings))
