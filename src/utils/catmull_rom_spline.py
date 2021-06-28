# Taken from: https://github.com/vmichals/python-algos/blob/master/catmull_rom_spline.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def catmull_rom_one_point(x, v0, v1, v2, v3):
    """Computes interpolated y-coord for given x-coord using Catmull-Rom.
    Computes an interpolated y-coordinate for the given x-coordinate between
    the support points v1 and v2. The neighboring support points v0 and v3 are
    used by Catmull-Rom to ensure a smooth transition between the spline
    segments.
    Args:
        x: the x-coord, for which the y-coord is needed
        v0: 1st support point
        v1: 2nd support point
        v2: 3rd support point
        v3: 4th support point
    """
    c1 = 1.0 * v1
    c2 = -0.5 * v0 + 0.5 * v2
    c3 = 1.0 * v0 + -2.5 * v1 + 2.0 * v2 - 0.5 * v3
    c4 = -0.5 * v0 + 1.5 * v1 + -1.5 * v2 + 0.5 * v3
    return ((c4 * x + c3) * x + c2) * x + c1


def catmull_rom(p_x, p_y, res):
    """Computes Catmull-Rom Spline for given support points and resolution.
    Args:
        p_x: array of x-coords
        p_y: array of y-coords
        res: resolution of a segment (including the start point, but not the
            endpoint of the segment)
    """
    # create arrays for spline points
    x_intpol = np.empty(res * (len(p_x) - 1) + 1)
    y_intpol = np.empty(res * (len(p_x) - 1) + 1)

    # set the last x- and y-coord, the others will be set in the loop
    x_intpol[-1] = p_x[-1]
    y_intpol[-1] = p_y[-1]

    # loop over segments (we have n-1 segments for n points)
    for i in range(len(p_x) - 1):
        # set x-coords
        x_intpol[i * res : (i + 1) * res] = np.linspace(
            p_x[i], p_x[i + 1], res, endpoint=False
        )
        if i == 0:
            # need to estimate an additional support point before the first
            y_intpol[:res] = np.array(
                [
                    catmull_rom_one_point(
                        x,
                        p_y[0] - (p_y[1] - p_y[0]),  # estimated start point,
                        p_y[0],
                        p_y[1],
                        p_y[2],
                    )
                    for x in np.linspace(0.0, 1.0, res, endpoint=False)
                ]
            )
        elif i == len(p_x) - 2:
            # need to estimate an additional support point after the last
            y_intpol[i * res : -1] = np.array(
                [
                    catmull_rom_one_point(
                        x,
                        p_y[i - 1],
                        p_y[i],
                        p_y[i + 1],
                        p_y[i + 1] + (p_y[i + 1] - p_y[i]),  # estimated end point
                    )
                    for x in np.linspace(0.0, 1.0, res, endpoint=False)
                ]
            )
        else:
            y_intpol[i * res : (i + 1) * res] = np.array(
                [
                    catmull_rom_one_point(x, p_y[i - 1], p_y[i], p_y[i + 1], p_y[i + 2])
                    for x in np.linspace(0.0, 1.0, res, endpoint=False)
                ]
            )

    return (x_intpol, y_intpol)
