import numpy as np
import matplotlib.pyplot as plt
import numba


import mpl_helpers as mpl_helpers


PLOT_CLOSURES = True
ELEORDER = 4

DERIV_ORDER_FOR_COMPARISON = 4

XLIM_PLOT = [-23, 23]
YLIM_PLOT = [1e-11, 1e-5]

linestyles = [
    ("solid", "solid"),  # Same as (0, ()) or '-'
    ("dotted", "dotted"),  # Same as (0, (1, 1)) or ':'
    ("dashed", "dashed"),  # Same as '--'
    ("dashdot", "dashdot"),  # Same as '-.'
    ("loosely dotted", (0, (1, 10))),
    ("dotted", (0, (1, 1))),
    ("densely dotted", (0, (1, 1))),
    ("long dash with offset", (5, (10, 3))),
    ("loosely dashed", (0, (5, 10))),
    ("dashed", (0, (5, 5))),
    ("densely dashed", (0, (5, 1))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]


@numba.jit(nopython=True)
def deriv_1st_x_2nd(u, dx, padding_width=3, bflag=0):
    """First order derivative, second order accuracy (2 points)"""
    nx = u.shape[0]
    xstart = 0
    xend = nx

    dxdxu = np.zeros_like(u)

    # inside bounds
    dxdxu[xstart + 1 : xend - 1] = (
        (
            -1.0 * u[xstart : xend - 2]
            # + 0.0 * u[ystart:yend, xstart:xend]
            + 1.0 * u[xstart + 2 : xend]
        )
        / (2 * dx)
    )

    if bflag & 1:
        dxdxu[xstart] = (-1.0 * u[xstart] + 1.0 * u[xstart + 1]) / (dx)
    if bflag & 2:
        dxdxu[xend - 1] = (1.0 * u[xend - 1] - 1.0 * u[xend - 2]) / (dx)

    return dxdxu


@numba.jit(nopython=True)
def deriv_1st_x_4th(u, dx, padding_width=3, bflag=0):
    """Assume that we have *at least* 2 padding points, and ignore the boundaries"""
    nx = u.shape[0]
    xstart = padding_width
    xend = nx - padding_width

    dxdxu = np.zeros_like(u)

    idx_12 = 1 / (dx * 12.0)
    idx_2 = 1 / (dx * 2.0)

    # inside bounds
    dxdxu[xstart:xend] = (
        1.0 * u[xstart - 2 : xend - 2]
        - 8.0 * u[xstart - 1 : xend - 1]
        + 0.0 * u[xstart:xend]
        + 8.0 * u[xstart + 1 : xend + 1]
        - 1.0 * u[xstart + 2 : xend + 2]
    ) * idx_12

    if bflag & 1:
        dxdxu[xstart] = (
            -3.0 * u[xstart] + 4.0 * u[xstart + 1] - 1.0 * u[xstart + 2]
        ) * idx_2

        dxdxu[xstart + 1] = (-1.0 * u[xstart] + 1.0 * u[xstart + 2]) * idx_2

    if bflag & 2:
        dxdxu[xend - 1] = (
            1.0 * u[xend - 3] - 4.0 * u[xend - 2] + 3.0 * u[xend - 1]
        ) * idx_2

        dxdxu[xend - 2] = (-1.0 * u[xend - 3] + 1.0 * u[xend - 1]) * idx_12

    return dxdxu


def deriv_1st_x_6th(u, dx, padding_width=3, bflag=0):
    """Assume that we have *at least* 2 padding points, and ignore the boundaries"""
    nx = u.shape[0]
    xstart = padding_width
    xend = nx - padding_width

    dxdxu = np.zeros_like(u)

    idx_60 = 1 / (dx * 60.0)
    idx_12 = 1 / (dx * 12.0)

    # inside bounds
    dxdxu[xstart:xend] = (
        -1.0 * u[xstart - 3 : xend - 3]
        + 9.0 * u[xstart - 2 : xend - 2]
        - 45.0 * u[xstart - 1 : xend - 1]
        + 0.0 * u[xstart:xend]
        + 45.0 * u[xstart + 1 : xend + 1]
        - 9.0 * u[xstart + 2 : xend + 2]
        + 1.0 * u[xstart + 3 : xend + 3]
    ) * idx_60

    if bflag & 1:
        dxdxu[xstart] = (
            -25.0 * u[xstart]
            + 48.0 * u[xstart + 1]
            - 36.0 * u[xstart + 2]
            + 16.0 * u[xstart + 3]
            - 3.0 * u[xstart + 4]
        ) * idx_12

        dxdxu[xstart + 1] = (
            -3.0 * u[xstart]
            - 10.0 * u[xstart + 1]
            + 18.0 * u[xstart + 2]
            - 6.0 * u[xstart + 3]
            + 1.0 * u[xstart + 4]
        ) * idx_12

        dxdxu[xstart + 2] = (
            1.0 * u[xstart]
            - 8.0 * u[xstart + 1]
            + 8.0 * u[xstart + 3]
            - 1.0 * u[xstart + 4]
        ) * idx_12

    if bflag & 2:
        dxdxu[xend - 1] = (
            25.0 * u[xend - 1]
            - 48.0 * u[xend - 2]
            + 36.0 * u[xend - 3]
            - 16.0 * u[xend - 4]
            + 3.0 * u[xend - 5]
        ) * idx_12

        dxdxu[xend - 2] = (
            3.0 * u[xend - 1]
            + 10.0 * u[xend - 2]
            - 18.0 * u[xend - 3]
            + 6.0 * u[xend - 4]
            - 1.0 * u[xend - 5]
        ) * idx_12

        dxdxu[xstart - 3] = (
            1.0 * u[xend - 5]
            - 8.0 * u[xend - 4]
            + 8.0 * u[xend - 2]
            - 1.0 * u[xend - 1]
        ) * idx_12

    return dxdxu


class BlocksTemp:
    def __init__(self, depth_list, xbeg, xend, eleorder):
        self.eleorder = eleorder
        self.pw = eleorder // 2
        self.depths = depth_list

        # find the minimum depth
        self.mindepth = min(depth_list)
        self.maxdepth = max(depth_list)

        self.xbeg = xbeg
        self.xend = xend
        self.dx_0 = (xend - xbeg) / eleorder

        # build up "base" array
        # if it was 0, we'd have eleorder + 1
        min_blocks = 2**self.mindepth
        min_points = min_blocks * eleorder + 1
        x = np.linspace(xbeg, xend, min_points)

        blocks = []
        self.dx_list = []
        x0 = xbeg
        for depth in depth_list:
            xe = x0 + eleorder * self.dx_from_level(depth)
            self.dx_list.append(self.dx_from_level(depth))
            # calc dx from the block level
            blocks.append(np.linspace(x0, xe, eleorder + 1))
            x0 = xe

        self.blocks_zipped = blocks
        self.blocks_unzipped = []

        print("Minimum DX", np.min(self.dx_list))
        print("Maximum DX", np.max(self.dx_list))

        for blk in self.blocks_zipped:
            dx = blk[2] - blk[1]
            tmp = np.pad(blk, self.pw)
            for ii in range(self.pw):
                tmp[ii] = tmp[self.pw] - dx * (self.pw - ii)
                tmp[-(ii + 1)] = tmp[-(self.pw + 1)] + dx * (self.pw - ii)

            self.blocks_unzipped.append(tmp)

    def apply_initial_data(self, f):
        self.data_zipped = []

        for blk in self.blocks_zipped:
            data = f(blk)

            self.data_zipped.append(data)

        self.data_unzipped = []
        for blk in self.blocks_unzipped:
            data = f(blk)
            self.data_unzipped.append(data)

    def plot_zipped_data(self, axes=None):
        if axes is None:
            fig, axes = plt.subplots()
        else:
            fig = None
        last_depth = 0
        for ii, blk in enumerate(self.blocks_zipped):
            depth = self.depths[ii] - self.mindepth
            # if depth != last_depth:
            # plot a refinement boundary as a vline

            if ii == 0:
                axes.plot(blk, self.data_zipped[ii], color="k", label="Initial Data")
            else:
                axes.plot(blk, self.data_zipped[ii], color="k")

        return fig, axes

    def block_boundary_axvline(self, axes):
        for ii, blk in enumerate(self.blocks_zipped):
            axes.axvline(x=blk[0], linestyle="dotted", color="k", alpha=0.3)
        axes.axvline(x=blk[-1], linestyle="dotted", color="k", alpha=0.3)

    def dx_from_level(self, level):
        return self.dx_0 / 2**level

    def plot_zipped_blocks(self):
        plt.figure()
        for ii, blk in enumerate(self.blocks_zipped):
            plt.scatter(blk, np.zeros_like(blk), label=f"blk {ii}")
        plt.legend()

    def plot_unzipped_blocks(self):
        plt.figure()
        for ii, blk in enumerate(self.blocks_unzipped):
            plt.scatter(blk, np.zeros_like(blk), label=f"blk {ii}")
        plt.legend()

    def plot_unique(self):
        uq = self.unique_pts()
        print(uq)
        plt.figure()
        plt.scatter(uq, np.zeros_like(uq))

    def unique_pts(self):
        uq = np.unique(blks.blocks_zipped)
        uq.sort()
        return uq

    def count_up_shifts(self):
        count = 0
        for ii, depth in enumerate(self.depths):
            if ii == 0:
                prev_depth = depth
            else:
                if depth > prev_depth:
                    count += 1
                prev_depth = depth
        return count

    def count_down_shifts(self):
        count = 0
        for ii, depth in enumerate(self.depths):
            if ii == 0:
                prev_depth = depth
            else:
                if depth < prev_depth:
                    count += 1
                prev_depth = depth
        return count

    def calculate_derivs(self, order=2):
        derivs = []

        if order == 2:
            deriv_use = deriv_1st_x_2nd
        elif order == 4:
            deriv_use = deriv_1st_x_4th
        elif order == 6:
            deriv_use = deriv_1st_x_6th

        for iblk, blk in enumerate(self.data_unzipped):
            if iblk == 0:
                bflag = 1
            elif iblk == len(self.data_unzipped) - 1:
                bflag = 2
            else:
                bflag = 0
            deriv_temp = deriv_use(blk, self.dx_list[iblk], self.pw, bflag)

            # zip them up...
            derivs.append(deriv_temp[self.pw : -self.pw])

        self.derivs = derivs

        # now smush them together into a single array?
        for ii, deriv in enumerate(derivs):
            if ii == 0:
                deriv_comb = deriv[:-1]
            elif ii == len(derivs) - 1:
                deriv_comb = np.concatenate([deriv_comb, deriv])
            else:
                deriv_comb = np.concatenate([deriv_comb, deriv[:-1]])

        return deriv_comb

    def calculate_blockwise_cfd(self, R_main, R_left, R_right, R_both):
        derivs = []

        for iblk, blk in enumerate(self.data_unzipped):
            if iblk == 0 and iblk == len(self.data_unzipped) - 1:
                R_use = R_both
            if iblk == 0:
                R_use = R_left
            elif iblk == len(self.data_unzipped) - 1:
                R_use = R_right
            else:
                R_use = R_main
            deriv_temp = np.dot(R_use, blk) * (1 / self.dx_list[iblk])

            # zip them up...
            derivs.append(deriv_temp[self.pw : -self.pw])

        self.cfd_derivs = derivs

        # now smush them together into a single array?
        for ii, deriv in enumerate(derivs):
            if ii == 0:
                deriv_comb = deriv[:-1]
            elif ii == len(derivs) - 1:
                deriv_comb = np.concatenate([deriv_comb, deriv])
            else:
                deriv_comb = np.concatenate([deriv_comb, deriv[:-1]])

        return deriv_comb


def p1o4_x_vec(blks):
    uqs = blks.unique_pts()
    up_shifts = blks.count_up_shifts()
    down_shifts = blks.count_down_shifts()

    nP = len(uqs) + up_shifts + down_shifts
    print(nP, "is the number of points")
    eleorder = blks.eleorder

    x_arr = np.zeros(nP)
    var_arr = np.zeros(nP)
    dx_arr = np.zeros(nP)

    curr_counter = 0

    for iblk, blk in enumerate(blks.blocks_unzipped):
        # print("on block", iblk, "and curr_counter is", curr_counter)
        points_through = eleorder
        block_counter = blks.pw

        if iblk > 0:
            if blks.depths[iblk - 1] > blks.depths[iblk]:
                x_arr[curr_counter] = blk[block_counter]
                var_arr[curr_counter] = blks.data_unzipped[iblk][block_counter]
                dx_arr[curr_counter] = blks.dx_list[iblk - 1]
                curr_counter += 1
                block_counter += 1
                points_through -= 1

                # then get the data and point from last block *or* interpolate
                x_arr[curr_counter] = blks.blocks_unzipped[iblk - 1][blks.pw * 3 + 1]
                var_arr[curr_counter] = blks.data_unzipped[iblk - 1][blks.pw * 3 + 1]
                dx_arr[curr_counter] = blks.dx_list[iblk - 1]
                curr_counter += 1
                # print(iblk, "prev was deeper!")
            if blks.depths[iblk - 1] < blks.depths[iblk]:
                # print(iblk, "prev was coarser!")
                x_arr[curr_counter] = blk[block_counter - 1]
                var_arr[curr_counter] = blks.data_unzipped[iblk][block_counter - 1]
                dx_arr[curr_counter] = blks.dx_list[iblk]
                curr_counter += 1

        if iblk < len(blks.blocks_zipped) - 1:
            if blks.depths[iblk + 1] > blks.depths[iblk]:
                # print(iblk, "next is deeper!")
                # the last one needs to be adjusted
                pass
            if blks.depths[iblk + 1] < blks.depths[iblk]:
                # print(iblk, "next is coarser!")
                pass

        if iblk == 0:
            pass

        elif iblk == len(blks.blocks_unzipped) - 1:
            # also add the last point
            x_arr[nP - 1] = blk[blks.pw * 3]
            var_arr[nP - 1] = blks.data_unzipped[iblk][blks.pw * 3]
            dx_arr[nP - 1] = blks.dx_list[iblk]
            print(blks.pw * 3, x_arr[nP - 1])
            pass

        for ii in range(points_through):
            x_arr[curr_counter] = blk[block_counter]
            var_arr[curr_counter] = blks.data_unzipped[iblk][block_counter]
            dx_arr[curr_counter] = blks.dx_list[iblk]

            block_counter += 1
            curr_counter += 1

    return x_arr, var_arr, dx_arr


def block_wise_p1o4_r(eleorder, is_left=False, is_right=False, use_closure="none"):
    true_pts = 2 * eleorder + 1
    nP = true_pts
    pw = eleorder // 2

    if use_closure == "closure" and pw < 3:
        raise ValueError("Padding width needs to be 3 for these closures!")

    # create and build 3 matrices
    a = [-0.75, 0.0, 0.75]
    alpha = [0.25, 1.0, 0.25]
    Ld = 1
    Rd = 1
    Lf = 1
    Rf = 1

    if is_left:
        nP -= pw
    if is_right:
        nP -= pw

    P = np.zeros((nP, nP))
    Q = np.zeros((nP, nP))

    start_idx = 0
    end_idx = nP

    if use_closure == "closure":
        for ii in range(pw):
            P[ii, ii] = 1.0
            Q[ii, ii] = 1.0

        t1 = 1.0 / 72.0
        start_idx = pw
        Q[start_idx, start_idx - 3] = -t1
        Q[start_idx, start_idx - 2] = 10.0 * t1
        Q[start_idx, start_idx - 1] = -53.0 * t1
        Q[start_idx, start_idx] = 0.0
        Q[start_idx, start_idx + 1] = 53.0 * t1
        Q[start_idx, start_idx + 2] = -10.0 * t1
        Q[start_idx, start_idx + 3] = t1
        P[start_idx, start_idx] = 1.0
        start_idx += 1

        for ii in range(nP - pw, nP):
            P[ii, ii] = 1.0
            Q[ii, ii] = 1.0

        end_idx = end_idx - pw - 1
        Q[end_idx, end_idx - 3] = -t1
        Q[end_idx, end_idx - 2] = 10.0 * t1
        Q[end_idx, end_idx - 1] = -53.0 * t1
        Q[end_idx, end_idx] = 0.0
        Q[end_idx, end_idx + 1] = 53.0 * t1
        Q[end_idx, end_idx + 2] = -10.0 * t1
        Q[end_idx, end_idx + 3] = t1
        P[end_idx, end_idx] = 1.0

    else:
        # NOTE: DIRICHLET CLOSURES EVERYWHERE
        # then we build up the matrix
        P[0, 0] = 1.0
        P[0, 1] = 3.0

        Q[0, 0] = -17.0 / 6.0
        Q[0, 1] = 3.0 / 2.0
        Q[0, 2] = 3.0 / 2.0
        Q[0, 3] = -1.0 / 6.0
        start_idx += 1

        P[nP - 1, nP - 1] = 1.0
        P[nP - 1, nP - 2] = 3.0

        Q[nP - 1, nP - 1] = 17.0 / 6.0
        Q[nP - 1, nP - 2] = -3.0 / 2.0
        Q[nP - 1, nP - 3] = -3.0 / 2.0
        Q[nP - 1, nP - 4] = 1.0 / 6.0
        end_idx -= 1

    # then the rest of the points
    for ii in range(start_idx, end_idx):
        for kk in range(-Ld, Rd + 1):
            P[ii, ii + kk] = alpha[kk + Ld]
        for kk in range(-Lf, Rf + 1):
            Q[ii, ii + kk] = a[kk + Lf]

    # then if we were left or right, we have to fill in the matrix
    P_true = np.eye(true_pts, true_pts)
    Q_true = np.zeros((true_pts, true_pts))
    if is_left and is_right:
        # then slot in the matrix
        P_true[pw:-pw, pw:-pw] = P
        Q_true[pw:-pw, pw:-pw] = Q
    elif is_left:
        # then slot in the matrix
        P_true[pw:, pw:] = P
        Q_true[pw:, pw:] = Q
    elif is_right:
        # then slot in the matrix
        P_true[:-pw, :-pw] = P
        Q_true[:-pw, :-pw] = Q
    else:
        P_true = P
        Q_true = Q

    # if use_closure == "lopsided":
    #     print(P_true)
    #     print(Q_true)

    # then calculate R
    P_inv = np.linalg.inv(P_true)
    R = np.matmul(P_inv, Q_true)

    return P_true, Q_true, R


def p104_PQ_mat(blks):
    a = [-0.75, 0.0, 0.75]
    alpha = [0.25, 1.0, 0.25]
    Ld = 1
    Rd = 1
    Lf = 1
    Rf = 1

    uqs = blks.unique_pts()
    up_shifts = blks.count_up_shifts()
    down_shifts = blks.count_down_shifts()

    nP = len(uqs) + up_shifts + down_shifts

    P = np.zeros((nP, nP))
    x_arr = np.zeros(nP)

    n = 10
    Q = np.zeros((nP, nP))
    curr_counter = 1

    for iblk, blk in enumerate(blks.blocks_zipped):
        points_through = blks.eleorder
        block_counter = 0

        if iblk > 0:
            if blks.depths[iblk - 1] > blks.depths[iblk]:
                # print(iblk, "prev was deeper!")
                for ii in range(2):
                    for kk in range(-Ld, Rd + 1):
                        P[curr_counter, curr_counter + kk] = alpha[kk + Ld]
                    for kk in range(-Lf, Rf + 1):
                        Q[curr_counter, curr_counter + kk] = a[kk + Lf]
                    # if prev was deeper, then we need to set up a row like "normal" but with a skipped point in the middle
                    # skip a row...
                    curr_counter += 1
                # then fill in with the gap on the first one
                P[curr_counter, curr_counter - 2] = alpha[0]
                P[curr_counter, curr_counter] = alpha[1]
                P[curr_counter, curr_counter + 1] = alpha[2]
                Q[curr_counter, curr_counter - 2] = a[0]
                Q[curr_counter, curr_counter] = a[1]
                Q[curr_counter, curr_counter + 1] = a[2]
                curr_counter += 1
                points_through -= 2

            if blks.depths[iblk - 1] < blks.depths[iblk]:
                # print(iblk, "prev was coarser!")
                # points_through += 1
                pass

        if iblk < len(blks.blocks_zipped) - 1:
            if blks.depths[iblk + 1] > blks.depths[iblk]:
                # print(iblk, "next is deeper!")
                points_through -= 1
                # the last one needs to be adjusted
            if blks.depths[iblk + 1] < blks.depths[iblk]:
                # print(iblk, "next is coarser!")
                pass

        # NOTE: this is the left boundary!
        if iblk == 0:
            P[0, 0] = 1.0
            P[0, 1] = 3.0

            Q[0, 0] = -17.0 / 6.0
            Q[0, 1] = 3.0 / 2.0
            Q[0, 2] = 3.0 / 2.0
            Q[0, 3] = -1.0 / 6.0

            points_through -= 1

        # NOTE: this is the right boundary!
        elif iblk == len(blks.blocks_zipped) - 1:
            P[nP - 1, nP - 1] = 1.0
            P[nP - 1, nP - 2] = 3.0

            Q[nP - 1, nP - 1] = 17.0 / 6.0
            Q[nP - 1, nP - 2] = -3.0 / 2.0
            Q[nP - 1, nP - 3] = -3.0 / 2.0
            Q[nP - 1, nP - 4] = 1.0 / 6.0
            # points_through -= 1

        # then we can fill in the remainder
        for ii in range(points_through):
            for kk in range(-Ld, Rd + 1):
                P[curr_counter, curr_counter + kk] = alpha[kk + Ld]
            for kk in range(-Lf, Rf + 1):
                Q[curr_counter, curr_counter + kk] = a[kk + Lf]

            curr_counter += 1

        if iblk < len(blks.blocks_zipped) - 1:
            if blks.depths[iblk + 1] > blks.depths[iblk]:
                # the last one needs to be adjusted
                # print(iblk, "next is deeper!")
                P[curr_counter, curr_counter - 1] = alpha[0]
                P[curr_counter, curr_counter] = alpha[1]
                P[curr_counter, curr_counter + 2] = alpha[2]
                Q[curr_counter, curr_counter - 1] = a[0]
                Q[curr_counter, curr_counter] = a[1]
                Q[curr_counter, curr_counter + 2] = a[2]
                curr_counter += 1

                P[curr_counter, curr_counter - 1] = alpha[0]
                P[curr_counter, curr_counter] = alpha[1]
                P[curr_counter, curr_counter + 1] = alpha[2]
                Q[curr_counter, curr_counter - 1] = a[0]
                Q[curr_counter, curr_counter] = a[1]
                Q[curr_counter, curr_counter + 1] = a[2]
                curr_counter += 1
            if blks.depths[iblk + 1] < blks.depths[iblk]:
                # print(iblk, "next is coarser!")
                pass

    # now for the other internal points
    return P, Q


def kim_PQ_mat(blks):
    alpha = 0.5862704032801503
    beta = 9.549533555017055e-2

    a1 = 0.6431406736919156
    a2 = 0.2586011023495066
    a3 = 7.140953479797375e-3

    y00 = 0.0
    y10 = 8.360703307833438e-2
    y20 = 3.250008295108466e-2
    y01 = 5.912678614078549
    y11 = 0.0
    y21 = 0.3998040493524358
    y02 = 3.775623951744012
    y12 = 2.058102869495757
    y22 = 0.0
    y03 = 0.0
    y13 = 0.9704052014790193
    y23 = 0.7719261277615860
    y04 = 0.0
    y14 = 0.0
    y24 = 0.1626635931256900

    b10 = -0.3177447290722621
    b20 = -0.1219006056449124
    b01 = -3.456878182643609
    b21 = -0.6301651351188667
    b02 = 5.839043358834730
    b12 = -2.807631929593225e-2
    b03 = 1.015886726041007
    b13 = 1.593461635747659
    b23 = 0.6521195063966084
    b04 = -0.2246526470654333
    b14 = 0.2533027046976367
    b24 = 0.3938843551210350
    b05 = 8.564940889936562e-2
    b15 = -3.619652460174756e-2
    b25 = 1.904944407973912e-2
    b06 = -1.836710059356763e-2
    b16 = 4.080281419108407e-3
    b26 = -1.027260523947668e-3

    b00 = -(b01 + b02 + b03 + b04 + b05 + b06)
    b11 = -(b10 + b12 + b13 + b14 + b15 + b16)
    b22 = -(b20 + b21 + b23 + b24 + b25 + b26)

    uqs = blks.unique_pts()
    up_shifts = blks.count_up_shifts()
    down_shifts = blks.count_down_shifts()

    nP = len(uqs) + up_shifts + down_shifts

    P = np.zeros((nP, nP))
    x_arr = np.zeros(nP)

    n = 10
    Q = np.zeros((n, n))

    # iterate through the blocks
    curr_counter = 3
    for iblk, blk in enumerate(blks.blocks_zipped):
        points_through = blks.eleorder
        block_counter = 0

        if iblk > 0:
            if blks.depths[iblk - 1] > blks.depths[iblk]:
                # print(iblk, "prev was deeper!")
                P[curr_counter, curr_counter - 2] = beta
                P[curr_counter, curr_counter - 1] = alpha
                P[curr_counter, curr_counter] = 1.0
                P[curr_counter, curr_counter + 1] = alpha
                P[curr_counter, curr_counter + 2] = beta

                x_arr[curr_counter] = blk[block_counter]
                x_arr[curr_counter + 1] = 100000
                block_counter += 1
                curr_counter += 2

                # first in block...
                P[curr_counter, curr_counter - 2] = beta
                P[curr_counter, curr_counter - 2] = alpha
                P[curr_counter, curr_counter] = 1.0
                P[curr_counter, curr_counter + 1] = alpha
                P[curr_counter, curr_counter + 2] = beta
                x_arr[curr_counter] = blk[block_counter]
                block_counter += 2
                curr_counter += 1
                # skip two points now...
                points_through -= 2

            if blks.depths[iblk - 1] < blks.depths[iblk]:
                x_arr[curr_counter] = -100000
                curr_counter += 1

                # print(iblk, "prev was coarser!")

        if iblk < len(blks.blocks_zipped) - 1:
            if blks.depths[iblk + 1] > blks.depths[iblk]:
                # print(iblk, "next is deeper!")
                pass
            if blks.depths[iblk + 1] < blks.depths[iblk]:
                # print(iblk, "next is coarser!")
                points_through -= 1

        if iblk == 0:
            P[0, 0] = 1.0
            P[0, 1] = y01
            P[0, 2] = y02

            P[1, 0] = y10
            P[1, 1] = 1.0
            P[1, 2] = y12
            P[1, 3] = y13

            P[2, 0] = y20
            P[2, 1] = y21
            P[2, 2] = 1.0
            P[2, 3] = y23
            P[2, 4] = y24
            x_arr[0] = blk[0]
            x_arr[1] = blk[1]
            x_arr[2] = blk[2]
            points_through -= 3

        elif iblk == len(blks.blocks_zipped) - 1:
            P[nP - 3, nP - 5] = y24
            P[nP - 3, nP - 4] = y23
            P[nP - 3, nP - 3] = 1.0
            P[nP - 3, nP - 2] = y21
            P[nP - 3, nP - 1] = y20

            P[nP - 2, nP - 4] = y13
            P[nP - 2, nP - 3] = y12
            P[nP - 2, nP - 2] = 1.0
            P[nP - 2, nP - 1] = y10

            P[nP - 1, nP - 3] = y02
            P[nP - 1, nP - 2] = y01
            P[nP - 1, nP - 1] = 1.0
            x_arr[nP - 3] = blk[-3]
            x_arr[1] = blk[-2]
            x_arr[2] = blk[-1]
            points_through -= 3
        print(points_through)
        for ii in range(points_through):
            P[curr_counter, curr_counter - 2] = beta
            P[curr_counter, curr_counter - 1] = alpha
            P[curr_counter, curr_counter] = 1.0
            P[curr_counter, curr_counter + 1] = alpha
            P[curr_counter, curr_counter + 2] = beta

            x_arr[curr_counter] = blk[block_counter + ii]

            curr_counter += 1

        if iblk < len(blks.blocks_zipped) - 1:
            if blks.depths[iblk + 1] > blks.depths[iblk]:
                print(iblk, "next is deeper!")
            if blks.depths[iblk + 1] < blks.depths[iblk]:
                print(iblk, "next is coarser!")
                P[curr_counter, curr_counter - 2] = beta
                P[curr_counter, curr_counter - 1] = alpha
                P[curr_counter, curr_counter] = 1.0
                P[curr_counter, curr_counter + 1] = alpha
                P[curr_counter, curr_counter + 2] = beta

                x_arr[curr_counter] = blk[block_counter + ii]

                curr_counter += 1
    return P, Q

    # so our number of P points in our matrix is going to then be...

    n = 10

    # P = np.zeros(

    P = np.zeros((n, n))
    Q = np.zeros((n, n))

    for i in range(3, n - 3):
        P[i, i - 2] = beta
        P[i, i - 1] = alpha
        P[i, i] = 1.0
        P[i, i + 1] = alpha
        P[i, i + 2] = beta

    P[0, 0] = 1.0
    P[0, 1] = y01
    P[0, 2] = y02

    P[1, 0] = y10
    P[1, 1] = 1.0
    P[1, 2] = y12
    P[1, 3] = y13

    P[2, 0] = y20
    P[2, 1] = y21
    P[2, 2] = 1.0
    P[2, 3] = y23
    P[2, 4] = y24

    P[n - 3, n - 5] = y24
    P[n - 3, n - 4] = y23
    P[n - 3, n - 3] = 1.0
    P[n - 3, n - 2] = y21
    P[n - 3, n - 1] = y20

    P[n - 2, n - 4] = y13
    P[n - 2, n - 3] = y12
    P[n - 2, n - 2] = 1.0
    P[n - 2, n - 1] = y10

    P[n - 1, n - 3] = y02
    P[n - 1, n - 2] = y01
    P[n - 1, n - 1] = 1.0

    for i in range(3, n - 3):
        Q[i, i - 3] = -a3
        Q[i, i - 2] = -a2
        Q[i, i - 1] = -a1
        Q[i, i] = 0.0
        Q[i, i + 1] = a1
        Q[i, i + 2] = a2
        Q[i, i + 3] = a3

    Q[0, 0] = b00
    Q[0, 1] = b01
    Q[0, 2] = b02
    Q[0, 3] = b03
    Q[0, 4] = b04
    Q[0, 5] = b05
    Q[0, 6] = b06

    Q[1, 0] = b10
    Q[1, 1] = b11
    Q[1, 2] = b12
    Q[1, 3] = b13
    Q[1, 4] = b14
    Q[1, 5] = b15
    Q[1, 6] = b16

    Q[2, 0] = b20
    Q[2, 1] = b21
    Q[2, 2] = b22
    Q[2, 3] = b23
    Q[2, 4] = b24
    Q[2, 5] = b25
    Q[2, 6] = b26

    Q[n - 3, n - 1] = -b20
    Q[n - 3, n - 2] = -b21
    Q[n - 3, n - 3] = -b22
    Q[n - 3, n - 4] = -b23
    Q[n - 3, n - 5] = -b24
    Q[n - 3, n - 6] = -b25
    Q[n - 3, n - 7] = -b26

    Q[n - 2, n - 1] = -b10
    Q[n - 2, n - 2] = -b11
    Q[n - 2, n - 3] = -b12
    Q[n - 2, n - 4] = -b13
    Q[n - 2, n - 5] = -b14
    Q[n - 2, n - 6] = -b15
    Q[n - 2, n - 7] = -b16

    Q[n - 1, n - 1] = -b00
    Q[n - 1, n - 2] = -b01
    Q[n - 1, n - 3] = -b02
    Q[n - 1, n - 4] = -b03
    Q[n - 1, n - 5] = -b04
    Q[n - 1, n - 6] = -b05
    Q[n - 1, n - 7] = -b06

    return P, Q


# blks = BlocksTemp([2, 3, 3, 3, 3, 2], 0.0, 1.0, 2)


# def initial_data(x):
#     return np.sin(0.5 * np.pi * x) * np.cos(1.5 * np.pi * x)

LAMBDA1 = 0.05
AMPLITUDE = 5.0


def initial_data(x):
    r = np.sqrt(x * x)

    Ephi_up = -8.0 * AMPLITUDE * LAMBDA1 * LAMBDA1 * np.exp(-LAMBDA1 * x**2)
    return x * Ephi_up


def deriv_data(x):
    return (0.5 * np.pi) * np.cos(0.5 * np.pi * x) * np.cos(1.5 * np.pi * x) - (
        1.5 * np.pi
    ) * np.sin(0.5 * np.pi * x) * np.sin(1.5 * np.pi * x)


def deriv_data(x):
    return (
        (2 * LAMBDA1 * x**2 - 1) * 8 * AMPLITUDE * LAMBDA1**2 * np.exp(-LAMBDA1 * x**2)
    )


eleorder = ELEORDER
pw = eleorder // 2

if eleorder == 4:
    DOMAIN = [-33.3333333333333333333333, 33.333333333333333333333333333]
elif eleorder == 6:
    DOMAIN = [-25.0, 25.0]

grid_levels = [
    3,
    3,
    4,
    5,
    5,
    6,
    6,
    6,
    7,
    7,
    7,
    7,
    7,
    7,
    8,
    8,
    7,
    7,
    8,
    8,
    8,
    8,
    7,
    7,
    8,
    8,
    7,
    7,
    7,
    7,
    7,
    7,
    6,
    6,
    6,
    5,
    5,
    4,
    3,
    3,
]

grid_levels_ele4 = [
    3,
    4,
    4,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    5,
    5,
    5,
    4,
    4,
    3,
]

grid_levels_ele6 = [3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 3]

if ELEORDER == 4:
    grid_levels = grid_levels_ele4
else:
    grid_levels = grid_levels_ele6

print(DOMAIN)

blks = BlocksTemp(grid_levels, DOMAIN[0], DOMAIN[1], eleorder)

blks.apply_initial_data(initial_data)
# fig, axs = blks.plot_zipped_data()


P, Q = p104_PQ_mat(blks)
all_x, var_arr, dx_arr = p1o4_x_vec(blks)


# axs.scatter(all_x, var_arr)

if 1:
    plt.figure()
    plt.scatter(all_x, np.zeros_like(all_x), color=[1.0, 0.5, 0.5])
    plt.title("Checking grid")

    fig = plt.figure(figsize=(8, 7))
    plt.matshow(P, cmap="RdYlBu")  # , clim=(0, 1.0))
    plt.colorbar()
    plt.title("P")

    fig = plt.figure(figsize=(8, 7))
    plt.matshow(Q, cmap="RdYlBu")  # , clim=(0, 1.0))
    plt.colorbar()
    plt.title("Q")

# try inverting p for fun

if not np.isclose(0.0, np.linalg.det(P)):
    Pinv = np.linalg.inv(P)

    if 0:
        fig = plt.figure(figsize=(8, 7))
        plt.matshow(Pinv, cmap="RdYlBu", fignum=fig.number)  # , clim=(0, 1.0))
        plt.title("P Inverse")
        plt.colorbar()

    R = np.matmul(Pinv, Q)
    if 0:
        fig = plt.figure(figsize=(8, 7))
        plt.matshow(R, cmap="RdYlBu", fignum=fig.number)  # , clim=(0, 1.0))
        plt.title("R")
        plt.colorbar()
else:
    print("Sorry! Matrix can't be inverted!")


# otherwise we're going to attempt to calculate the solution via an iterative solve
# well, at least with just numpy's linsolve

# first we get our temporary array...

print(
    "Matrix Rank is:",
    np.linalg.matrix_rank(P),
    "shape is",
    P.shape,
    "and determinant is ",
    np.linalg.det(P),
)

inbetween = np.dot(Q, var_arr) * (1 / dx_arr)
# plt.figure()
# plt.plot(all_x, inbetween)

# now we do the matrix solve
deriv_output = np.linalg.solve(P, inbetween)

# FINITE DIFFERENCE STUFF
extra_derivs = blks.calculate_derivs(DERIV_ORDER_FOR_COMPARISON)


# then block-wise CFD's
_, _, R_main = block_wise_p1o4_r(eleorder, is_left=False, is_right=False)

if pw > 2:
    _, _, R_main_closure = block_wise_p1o4_r(
        eleorder, is_left=False, is_right=False, use_closure="closure"
    )
_, _, R_left = block_wise_p1o4_r(eleorder, is_left=True, is_right=False)
_, _, R_right = block_wise_p1o4_r(eleorder, is_left=False, is_right=True)
_, _, R_both = block_wise_p1o4_r(eleorder, is_left=True, is_right=True)
cfd_blockwise_derivs = blks.calculate_blockwise_cfd(R_main, R_left, R_right, R_both)
if pw > 2:
    cfd_blockwise_closure_derivs = blks.calculate_blockwise_cfd(
        R_main_closure, R_left, R_right, R_both
    )

if 0:
    fig = plt.figure(figsize=(8, 7))
    plt.matshow(R_main, cmap="RdYlBu", fignum=fig.number, clim=(-1.0, 1.0))
    plt.colorbar()
    plt.title("R_main")

if pw > 2:
    fig = plt.figure(figsize=(8, 7))
    plt.matshow(R_main_closure, cmap="RdYlBu", fignum=fig.number, clim=(-1.0, 1.0))
    plt.colorbar()
    plt.title("R_closure")


if pw > 2:
    fig = plt.figure(figsize=(8, 7))
    plt.matshow(
        np.abs(R_main_closure[pw:-pw, pw:-pw] - R_main[pw:-pw, pw:-pw]),
        cmap="RdYlBu",
        fignum=fig.number,
    )
    plt.colorbar()
    plt.title("R Differences (closure vs main)")


# END CALCULATIOSN

# plt.figure()
# plt.plot(all_x, deriv_data(all_x), label="True Deriv")
# plt.plot(blks.unique_pts(), extra_derivs, label="Finite Difference Output")
# plt.plot(blks.unique_pts(), cfd_blockwise_derivs, label="Block-wise CFD")
# plt.plot(all_x, deriv_output, label="Global CFD")
# plt.title("Deriv Output")
# plt.legend()
# plt.grid()

# now errors

# compute the error
cfd_error = np.abs(deriv_data(all_x) - deriv_output)
cfd_block_error = np.abs(deriv_data(blks.unique_pts()) - cfd_blockwise_derivs)

if pw > 2:
    cfd_block_closure_error = np.abs(
        deriv_data(blks.unique_pts()) - cfd_blockwise_closure_derivs
    )
fd_error = np.abs(deriv_data(blks.unique_pts()) - extra_derivs)


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=mpl_helpers.double_height)

axes[0].plot()

_, axs = blks.plot_zipped_data(axes[0])
axes[0].plot(
    all_x,
    deriv_data(all_x),
    label="Analytical Derivative",
    color="k",
    linestyle="dashed",
)
blks.block_boundary_axvline(axes[0])
axes[0].set_ylabel(r"${\bf E}_1$ & $\partial {\bf E}_1 / \partial x$")
axes[0].set_xlabel(r"$x$")
axes[0].grid(axis="y")
axes[0].set_xlim(XLIM_PLOT)
axes[0].legend(fontsize="10", loc="lower right")
axes[0].text(-0.1, 1.1, "I)", transform=axes[0].transAxes)

axes[1].plot(blks.unique_pts(), fd_error, linestyle="dashed", label="Finite Difference")
axes[1].plot(all_x, cfd_error, linestyle="solid", label="Global")
axes[1].plot(blks.unique_pts(), cfd_block_error, linestyle="dotted", label="Block-Wise")
if PLOT_CLOSURES:
    if pw > 2:
        axes[1].plot(
            blks.unique_pts(),
            cfd_block_closure_error,
            linestyle="dashdot",
            label="Block-Wise Closure",
        )
axes[1].set_yscale("log")
blks.block_boundary_axvline(axes[1])
axes[1].grid(axis="y")
# axes[1].set_title("Derivative Errors")
axes[1].set_xlabel(r"$x$")
axes[1].set_ylabel(r"Absolute Error")
axes[1].set_xlim(XLIM_PLOT)
axes[1].legend(fontsize="10", loc="lower right")
axes[1].set_ylim(YLIM_PLOT)
axes[1].text(-0.1, 1.1, "II)", transform=axes[1].transAxes)


axes[2].plot(blks.unique_pts(), fd_error, linestyle="dashed", label="Finite Difference")
axes[2].plot(all_x, cfd_error, linestyle="solid", label="Global")
axes[2].plot(blks.unique_pts(), cfd_block_error, linestyle="dotted", label="Block-Wise")
if PLOT_CLOSURES:
    if pw > 2:
        axes[2].plot(
            blks.unique_pts(),
            cfd_block_closure_error,
            linestyle="dashdot",
            label="Block-Wise Closure",
        )
axes[2].set_yscale("log")
blks.block_boundary_axvline(axes[2])
axes[2].grid(axis="y")
# axes[2].set_title("Derivative Errors (Zoomed In)")
axes[2].set_xlim([-3.0, 3.0])
axes[2].set_ylim(YLIM_PLOT)
axes[2].set_xlabel(r"$x$")
axes[2].set_ylabel(r"Absolute Error")
axes[2].legend(fontsize="10", loc="lower right")
axes[2].text(-0.1, 1.1, "III)", transform=axes[2].transAxes)

plt.savefig(
    f"solve_data_comparision_eleorder{ELEORDER}_fdorder{DERIV_ORDER_FOR_COMPARISON}.pdf"
)

plt.show()
