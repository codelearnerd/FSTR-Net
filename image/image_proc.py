import cv2
from skimage.transform import resize

def resize_to_max_area(obj_img, max_area=640*640):


    h, w = obj_img.shape[:2]
    current_area = h * w


    if current_area <= max_area:
        return obj_img.copy()


    scale_ratio = (max_area / current_area) ** 0.5


    new_w = int(w * scale_ratio)
    new_h = int(h * scale_ratio)


    new_w = max(1, new_w)
    new_h = max(1, new_h)


    resized_img = cv2.resize(obj_img, (new_w, new_h),
                           interpolation=cv2.INTER_AREA if scale_ratio < 1 else cv2.INTER_LINEAR)

    return resized_img

def resize_multichannel(im, new_h, new_w, mode='reflect', order=1):


    return resize(
        im,
        (new_h, new_w, im.shape[2]),
        mode=mode,
        order=order,
        preserve_range=True
    ).astype(im.dtype)
