import cv2
import random
import os
import numpy as np
import torch

from utils.ft_utils import ft2vector

import json
from PIL import ImageDraw, ImageFont, Image
font_path = os.path.dirname(os.path.dirname(__file__)) + '/font/simhei.ttf'

def ft2xy(an,bn,cn,dn,theta_fine,term):
    term = an.shape[0] if term<=0 else term
    x_approx = sum([an[i]*np.cos(i*theta_fine) + bn[i]*np.sin(i*theta_fine) for i in range(term)])
    y_approx = sum([cn[i]*np.cos(i*theta_fine) + dn[i]*np.sin(i*theta_fine) for i in range(term)])
    return x_approx,y_approx

def fft_area(ft_label):
    an,bn,cn,dn = np.split(ft_label[2:].reshape(-1, 4), 4, axis=-1)
    an, bn, cn, dn = an.squeeze(), bn.squeeze(), cn.squeeze(), dn.squeeze()

    assert an.shape[0]==len(an)==bn.shape[0]==cn.shape[0]==dn.shape[0]
    return np.pi * ((an*dn-bn*cn)*np.arange(1,1+an.shape[0])).sum()
def fft_areas(ft_labels):


    n_obj, dim = ft_labels.shape
    n = (dim - 2) // 4

    coeffs = ft_labels[:, 2:].reshape(n_obj, n, 4)
    an, bn, cn, dn = coeffs[..., 0], coeffs[..., 1], coeffs[..., 2], coeffs[..., 3]

    k = np.arange(1, n+1, dtype=ft_labels.dtype)

    areas = np.pi * ((an * dn - bn * cn) * k).sum(axis=1)
    return areas
def reverse_ffts(ft_label):
    ft_label[:,3::2] *= -1

def plot_one_box(x, img, color=None, label=None, line_thickness=None, ft_label=None, amp_stat=None, show_amp=0):

    tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    tf = max(tl - 1, 1)
    if ft_label is None:
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    else:
        theta_fine = np.linspace(0, 2*np.pi, 200)
        an,bn,cn,dn = np.split(ft_label[2:].reshape(-1, 4), 4, axis=-1)
        an,bn,cn,dn = np.insert(an,0,ft_label[0]),np.insert(bn,0,0),np.insert(cn,0,ft_label[1]),np.insert(dn,0,0)


        x_approx,y_approx = ft2xy(an,bn,cn,dn,theta_fine,0)
        contour_approx = np.array(list(zip(x_approx, y_approx)), dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [contour_approx], isClosed=True, color=color, thickness=line_thickness)


        c1 = (round(ft_label[0]), round(ft_label[1] - 15))
        amp = [0 for i in range(1, an.shape[0])]
        if amp_stat is None:
            amp_stat = [0 for i in range(0, an.shape[0])]
        for i_ in range(1, an.shape[0]):
            amp[i_ - 1] = np.sqrt(an[i_] ** 2 + bn[i_] ** 2 + cn[i_] ** 2 + dn[i_] ** 2)
            amp_stat[i_ - 1] += amp[i_ - 1]
            amp[i_ - 1] = amp[i_ - 1] / max(img.shape) * 30
            amp_stat[-1] += 1


        if label:
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=max(tf//2,1))[0]

            cv2.putText(img, label, (round(ft_label[0]), round(ft_label[1] - 2)), 0, tl / 6, [50, 50, 50], thickness=max(tf//2,1), lineType=cv2.LINE_AA)

        if show_amp:
            cv2.rectangle(img, (c1[0], c1[1] + 3), (c1[0] + len(amp) * 6 + 3, int(c1[1] - max(amp) * 5 - 3)), (255,255,255), -1)
            for i_ in range(1, an.shape[0]):
                offset_amp = i_ * 6 - 3
                cv2.rectangle(img, (c1[0] + offset_amp, c1[1]), (c1[0] + offset_amp + 3, int(c1[1] - amp[i_ - 1] * 5)), (255,255,0), -1)
            ft_area = fft_area(ft_label)
            str_area = f'{ft_area:.1f}'
            t_size = cv2.getTextSize(str_area, 0, fontScale=tl / 8, thickness=max(tf//2,1))[0]
            cv2.putText(img, str_area, (c1[0] + len(amp) * 6 - t_size[0], int(c1[1] - max(amp) * 5 + t_size[1])), 0, tl / 8, color=color, thickness=max(tf//2,1), lineType=cv2.LINE_AA)

def plot_one_box_with_ft(x, img, color=None, label=None, line_thickness=None, ft_label=None, amp_stat=None, show_amp=0):

    tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    tf = max(tl - 1, 1)
    if ft_label is None:
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    else:
        theta_fine = np.linspace(0, 2*np.pi, 200)
        an,bn,cn,dn = np.split(ft_label[2:].reshape(-1, 4), 4, axis=-1)
        an,bn,cn,dn = np.insert(an,0,ft_label[0]),np.insert(bn,0,0),np.insert(cn,0,ft_label[1]),np.insert(dn,0,0)


        x_approx,y_approx = ft2xy(an,bn,cn,dn,theta_fine,0)
        contour_approx = np.array(list(zip(x_approx, y_approx)), dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [contour_approx], isClosed=True, color=color, thickness=line_thickness)


        c1 = (round(ft_label[0]), round(ft_label[1] - 15))
        amp = [0 for i in range(1, an.shape[0])]
        if amp_stat is None:
            amp_stat = [0 for i in range(0, an.shape[0])]
        for i_ in range(1, an.shape[0]):
            amp[i_ - 1] = np.sqrt(an[i_] ** 2 + bn[i_] ** 2 + cn[i_] ** 2 + dn[i_] ** 2)
            amp_stat[i_ - 1] += amp[i_ - 1]
            amp[i_ - 1] = amp[i_ - 1] / max(img.shape) * 30
            amp_stat[-1] += 1


        if label:
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=max(tf//2,1))[0]

            cv2.putText(img, label, (round(ft_label[0]), round(ft_label[1] - 2)), 0, tl / 6, [50, 50, 50], thickness=max(tf//2,1), lineType=cv2.LINE_AA)

        if show_amp:
            cv2.rectangle(img, (c1[0], c1[1] + 3), (c1[0] + len(amp) * 6 + 3, int(c1[1] - max(amp) * 5 - 3)), (255,255,255), -1)
            for i_ in range(1, an.shape[0]):
                offset_amp = i_ * 6 - 3
                cv2.rectangle(img, (c1[0] + offset_amp, c1[1]), (c1[0] + offset_amp + 3, int(c1[1] - amp[i_ - 1] * 5)), (255,255,0), -1)
            ft_area = fft_area(ft_label)
            str_area = f'{ft_area:.1f}'
            t_size = cv2.getTextSize(str_area, 0, fontScale=tl / 8, thickness=max(tf//2,1))[0]
            cv2.putText(img, str_area, (c1[0] + len(amp) * 6 - t_size[0], int(c1[1] - max(amp) * 5 + t_size[1])), 0, tl / 8, color=color, thickness=max(tf//2,1), lineType=cv2.LINE_AA)


def plot_one_box_with_cms(x, img, color=None, label=None, line_thickness=None, cms_sub=None, cms_value=None, ft_label=None):

    tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        cms_name = ['(cm): ', '(cm): ', '(mm): ', '(cm^2): ']
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        font1 = ImageFont.truetype(font_path, 30)
        t_size = font1.getsize(label)

        up_label = True
        if (int(x[1]) - t_size[1] - 1) > 0:
            c1 = (int(x[0]), int(x[1]) - t_size[1] - 1)
            c2 = (int(x[0]) + t_size[0] + 4, int(x[1]) - 1)
            lb1 = (c1[0] + 3, c1[1])
        else:
            c1 = (int(x[0]), int(x[1]) + tl + 1)
            c2 = (int(x[0]) + t_size[0] + 3, int(x[1]) + tl + 1 + t_size[1])
            lb1 = (c1[0] + 3, c1[1])
            up_label = False

        if cms_sub or cms_value:
            length = 0
            if cms_sub:
                length += 1
            if cms_value:
                length += len(cms_value)
            t1_size = 0
            for i, j in enumerate(cms_value):
                t1_size = max(font1.getsize(f'{cms_name[i]}{j:.2f}')[0], t1_size)
            if (int(x[3]) + (t_size[1] + 10) * (length + 1)) < img.shape[0]:
                c3 = (int(x[0]), int(x[3]))
                c4 = (int(x[0]) + t1_size + 4, int(x[3]) + (t_size[1] + 10) * length)
                lb2 = [int(x[0]) + 3, int(x[3]) - tl]
            else:
                c3 = (int(x[0]), int(x[1]) + tl)
                c4 = (int(x[0]) + t1_size + 4, int(x[1]) + 10 + (t_size[1] + 10) * len(cms_value))
                if not up_label:
                    c3 = (c3[0], c3[1] + t_size[1])
                    c4 = (c4[0], c4[1] + t_size[1])
                lb2 = [c3[0] + 4, c3[1] + 10]

        offset = t_size[1] + 10
        draw.rectangle((c1, c2), fill=tuple(color), outline=tuple(color))
        if cms_sub or cms_value:
            draw.rectangle((c3, c4), fill=tuple(color), outline=tuple(color))
        draw.text(lb1, label, font=font1,fill=(225, 255, 255))
        if cms_sub is not None:
            draw.text((lb2[0], lb2[1]), f'cms_s: {cms_sub}', font=font1,fill=(225, 255, 255))
            lb2[1] += offset
        if cms_value is not None:
            for i, j in enumerate(cms_value):
                if i <= 4:
                    draw.text((lb2[0], lb2[1]), f'{cms_name[i]}{j:.2f}', font=font1,fill=(225, 255, 255))
                    lb2[1] += offset
                else:
                    draw.text((lb2[0], lb2[1]), '......', font=font1,fill=(225, 255, 255))
        if ft_label is not None:
            '''
            ft_label[0:2] *= np.array([img.shape[1],img.shape[0]])
            for i in range((ft_label.shape[0]-2)//4):
                start=2+i*4
                ft_label[start:start+4] *= np.array([img.shape[1],img.shape[1],img.shape[0],img.shape[0]])
            '''
            theta_fine = np.linspace(0, 2*np.pi, 200)
            an,bn,cn,dn = np.split(ft_label[2:].reshape(-1, 4), 4, axis=-1)

            x_approx = sum([an[i]*np.cos((i+1)*theta_fine) + bn[i]*np.sin((i+1)*theta_fine) for i in range(an.shape[0])])
            y_approx = sum([cn[i]*np.cos((i+1)*theta_fine) + dn[i]*np.sin((i+1)*theta_fine) for i in range(an.shape[0])])
            xy = np.vstack([x_approx, y_approx]).T
            xy[:, 0] += ft_label[0]
            xy[:, 1] += ft_label[1]
            xy = xy.astype(np.int32).reshape(-1).tolist()
            draw.polygon(xy, outline=tuple(color), width=tl)
        img[:, :, :] = np.array(pil_img)

def drawRect(img, pos, **kwargs):
    transp = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(transp, "RGBA")
    draw.rectangle(pos, **kwargs)
    img.paste(Image.alpha_composite(img, transp))

def plot_one_rot_box(x, img, color=None, label=None, line_thickness=None, leftop=False, radius=3, dir_line=False):
    tl = line_thickness or round(0.001 * max(img.shape[0:2])) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    x = np.int32(x)
    leftop_x = (x[0], x[1])
    if dir_line:
        x1, y1 = (x[0] + x[2]) / 2, (x[1]+x[3]) / 2
        x2, y2 = (x[4] + x[6]) / 2, (x[5] + x[7]) / 2
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cv2.arrowedLine(img, (int(cx), int(cy)), (int(x1), int(y1)), (0, 0, 255), thickness=tl)
    x = x.reshape((-1, 1, 2))
    cv2.polylines(img, [x], True, color, thickness=tl)
    if leftop:
        cv2.circle(img, leftop_x, radius, color, tl)
    if label:
        tf = max(tl - 1, 1)
        cv2.putText(img, label, leftop_x, 0, tl/3, color, thickness=tf, lineType=cv2.LINE_AA)


def show_ft(img, cls_id=(),xy_rect=(),ft_labels=(),colors=None):

    new_img = img.copy()
    theta_fine = np.linspace(0, 2*np.pi, 200)
    for i, label in enumerate(ft_labels):
        if len(ft_labels) > i:
            cls = int(cls_id[i])
            an,bn,cn,dn = [abcd.reshape(-1) for abcd in np.split(label[2:].reshape(-1, 4), 4, axis=-1)]
            x_approx = sum([an[i]*np.cos((i+1)*theta_fine) + bn[i]*np.sin((i+1)*theta_fine) for i in range(an.shape[0])])
            y_approx = sum([cn[i]*np.cos((i+1)*theta_fine) + dn[i]*np.sin((i+1)*theta_fine) for i in range(an.shape[0])])
            xy = np.vstack([x_approx + label[0], y_approx + label[1]]).T
            xy = xy.astype(np.int32)
            color = colors(cls%colors.n) if colors is not None else (0,0,255)
            cv2.polylines(new_img, [xy], True, color, 2)

        if len(xy_rect) > 0:
            xy_label = xy_rect[i]
            if len(xy_label)==4:
                cv2.rectangle(new_img, (int(xy_label[0]), int(xy_label[1])), (int(xy_label[2]), int(xy_label[3])), (0,255,0), 1)
            elif len(xy_label)==12:
                pts_ = xy_label[4:].reshape([-1, 2]).astype(np.int32)
                cv2.polylines(new_img, [pts_], True, (0,0,255), 2)
    return new_img
