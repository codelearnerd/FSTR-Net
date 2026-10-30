import numpy as np
import scipy.integrate as integrate
import torch
from tools.plotbox import plot_one_box

def is_point_inside_fourier_curve(fourier_coeffs, x0, y0):
    """
     (x0, y0)
    :param fourier_coeffs: np, shape [4n-2]
                           2a0, b0, 4(n-1)1n-1
    :param x0:  x0
    :param y0:  y0
    :return:  True False
    """

    n = (len(fourier_coeffs) + 2) // 4
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    a[0] = fourier_coeffs[0]
    c[0] = fourier_coeffs[1]
    for i in range(1, n):
        a[i] = fourier_coeffs[2 + 4 * (i - 1)]
        b[i] = fourier_coeffs[3 + 4 * (i - 1)]
        c[i] = fourier_coeffs[4 + 4 * (i - 1)]
        d[i] = fourier_coeffs[5 + 4 * (i - 1)]


    z0 = x0 + 1j * y0

    def fourier_series_curve(a, b, c, d, n, t):
        X = np.sum([a[i] * np.cos(i * t) + b[i] * np.sin(i * t) for i in range(n)], axis=0)
        Y = np.sum([c[i] * np.cos(i * t) + d[i] * np.sin(i * t) for i in range(n)], axis=0)
        return X, Y

    def fourier_series_derivative(a, b, c, d, n, t):
        X_prime = np.sum([i * (-a[i] * np.sin(i * t) + b[i] * np.cos(i * t)) for i in range(n)], axis=0)
        Y_prime = np.sum([i * (-c[i] * np.sin(i * t) + d[i] * np.cos(i * t)) for i in range(n)], axis=0)
        return X_prime, Y_prime

    def integrand(t):
        X, Y = fourier_series_curve(a, b, c, d, n, t)
        X_prime, Y_prime = fourier_series_derivative(a, b, c, d, n, t)
        z = X + 1j * Y
        dz_dt = X_prime + 1j * Y_prime
        return dz_dt / (z - z0)


    W, _ = integrate.quad(lambda t: integrand(t).real, 0, 2 * np.pi, limit=100)
    W_imag, _ = integrate.quad(lambda t: integrand(t).imag, 0, 2 * np.pi, limit=100)
    W = (W + 1j * W_imag) / (2 * np.pi * 1j)


    WA = abs(W)
    inside = WA > 0.0001
    if inside:
        print(f'W={W}, {WA}')

    return inside

def is_point_inside_fourier_curve_fast(fourier_coeffs, x0, y0):
    """
     (x0, y0)
    :param fourier_coeffs: np, shape [4n-2]
                           2a0, b0, 4(n-1)1n-1
    :param x0:  x0
    :param y0:  y0
    :return:  True False
    """

    n = (len(fourier_coeffs) + 2) // 4
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    a[0] = fourier_coeffs[0]
    c[0] = fourier_coeffs[1]
    S = 0
    for k in range(1, n):
        a[k] = fourier_coeffs[2 + 4 * (k - 1)]
        b[k] = fourier_coeffs[3 + 4 * (k - 1)]
        c[k] = fourier_coeffs[4 + 4 * (k - 1)]
        d[k] = fourier_coeffs[5 + 4 * (k - 1)]
        ad_bc = a[k]*d[k] - c[k]*b[k]
        ab_cd = a[k]*b[k] - c[k]*d[k]
        S += k*((1+x0)*ad_bc + y0*ab_cd)
    S/=(x0*x0+y0*y0)
    print(S)

    return np.abs(S) > 0.001

def render_fourier_curve(im0, objs, cls):
    """

    :param im0:
    :param objs: npshape [nt, 4n-2]
    :param cls: npshape [nt]
    :return:
    """

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
    ]

    height, width, _ = im0.shape

    cls_int = cls.to(torch.int)


    for y in range(height):
        for x in range(width):
            for i, obj in enumerate(objs):
                if is_point_inside_fourier_curve(obj, x, y):
                    category = cls_int[i]
                    color = colors[category % len(colors)]
                    im0[y, x] = color
                    break

    return im0


import numpy as np
import torch
import cv2
import random
def draw_x(image, center, color, size=5, thickness=1):
    """

    :param image:
    :param center:  (x, y)
    :param color:  (B, G, R)
    :param size:
    :param thickness:
    """
    x, y = center
    cv2.line(image, (x - size, y - size), (x + size, y + size), color, thickness)
    cv2.line(image, (x - size, y + size), (x + size, y - size), color, thickness)

def render_fourier_curve2(img_inside, objs, cls, n):
    """

    :param img_inside:
    :param objs: npshape [nt, 4n-2]
    :param cls: npshape [nt]
    :param n:
    :return:
    """

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
        (120, 120, 0),
    ]

    height, width, _ = img_inside.shape

    cls_int = cls.to(torch.int)


    bboxs=[]
    for i, obj in enumerate(objs):
        category = cls_int[i]
        color = colors[category % (len(colors)-1)]
        fourier_curve,bbox = compute_fourier_curve(obj)
        bboxs.append(bbox)
        plot_one_box(None, img_inside,
                    color=colors[category%len(colors)],
                    ft_label=obj.cpu().numpy(),line_thickness=1,show_amp=0)

    if 0:
        for _ in range(n):

            if random.random() < 0.8:
                idx = random.randint(0, len(objs) - 1)
                obj = objs[idx]
                category = cls_int[idx]
                color = colors[category % len(colors)]
                x_min, y_min, x_max, y_max = bboxs[idx]
                x = random.randint(x_min, x_max)
                y = random.randint(y_min, y_max)
            else:
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                color = colors[-1]


            category = -1
            for i, obj in enumerate(objs):
                if is_point_inside_fourier_curve(obj, x, y):
                    category = cls_int[i]
                    break
            if category>=0:
                cv2.circle(img_inside, (x, y), radius=4, color=colors[category % len(colors)], thickness=-1)
            else:
                draw_x(img_inside, (x, y), color=colors[-1], size=4, thickness=2)

    return img_inside

def compute_fourier_curve(obj):
    """

    :param obj:
    :return:
    """
    t = np.linspace(0, 2 * np.pi, num=100)
    curve = np.zeros((len(t), 2), dtype=np.int32)
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    for i in range(len(t)):
        x, y = 0, 0
        for k in range(len(obj) // 2):
            x += obj[2 * k] * np.cos(k * t[i]) - obj[2 * k + 1] * np.sin(k * t[i])
            y += obj[2 * k] * np.sin(k * t[i]) + obj[2 * k + 1] * np.cos(k * t[i])
        curve[i, 0] = int(x)
        curve[i, 1] = int(y)

        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
    return curve,bbox


if __name__ == "__main__":

    n = 10
    fourier_coeffs = np.concatenate([np.random.randn(2), np.random.randn(4 * (n - 1))])


    x0, y0 = 0.5, 0.5


    is_inside = is_point_inside_fourier_curve(fourier_coeffs, x0, y0)
    print(f"The point ({x0}, {y0}) is {'inside' if is_inside else 'outside'} the closed curve.")
