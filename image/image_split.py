import cv2
import numpy as np

def cal_overlay(W,w,lm):
    if W>w:
        nx_1 = (W-w) // (w-lm)
        if (W-w) % (w-lm) > 0:
            nx_1+=1
        return w - (W-w) / nx_1
    else:
        return lm

class CutImages:
    def __init__(self,  subsize=[640,640], over_lap=200, xyoff=[0,0], resize=0):
        self.subsize = subsize
        self.over_lap = over_lap
        self.xyoff = xyoff
        self.resize = resize

    def cut_images(self, image, imgsz):
        if isinstance(image, str):
            img_big = cv2.imread(image)
        else:
            img_big = image
        if img_big is not None:

            cut_affs = self._affine_imgs(img_big.shape[:2], imgsz) if (self.resize==0) else self._cut_imgs(img_big.shape[:2])
        else:
            cut_affs = []
        return img_big, cut_affs

    def _affine_imgs(self, img_shape, imgsz):
        height, width = img_shape
        subsize = self.subsize


        self.over_lapxy = [cal_overlay(width,self.subsize[1],self.over_lap), cal_overlay(height,self.subsize[0],self.over_lap)]
        assert self.over_lapxy[0]>=self.over_lap and self.over_lapxy[1]>=self.over_lap

        cut_results = []
        ds = 0.5
        sx,sy = (imgsz[1]-ds) / (subsize[1]-ds), (imgsz[0]-ds) / (subsize[0]-ds)


        dp = 0.1
        y = self.xyoff[1]
        while 1:
            x = self.xyoff[0]
            while 1:


                A23 = np.array([[sx, 0, -sx*(x - dp)], [0, sy, -sy*(y - dp)]], dtype=np.float32)


                A23_1 = np.array([[1.0/sx, 0, x-dp], [0, 1.0/sy, y-dp]], dtype=np.float32)


                cut = {
                    'A23': A23,
                    'A23_1': A23_1,

                    'cxy': np.array([[1, 0, x+subsize[1]/2], [0, 1, y+subsize[0]/2]], dtype=np.float32)
                }
                cut_results.append(cut)
                if x + subsize[1] >= width:
                    break
                x += subsize[1] - self.over_lapxy[0]
            if y + subsize[0] >= height:
                break
            y += subsize[0] - self.over_lapxy[1]
        return cut_results

    def _cut_imgs(self, img_shape):
        height, width = img_shape
        subsize = self.subsize
        over_lap = self.over_lap
        cut_results = []

        x, y = 0, 0
        h_last = False
        while y < height:
            if y + subsize[0] >= height:
                y = height - subsize[0]
                h_last = True
            w_last = False
            x = 0
            while x < width:
                if x + subsize[1] >= width:
                    x = width - subsize[1]
                    w_last = True

                cut = {
                    'xy': [x, y],

                }
                cut_results.append(cut)
                if w_last:
                    break
                x += subsize[1] - over_lap
            if h_last:
                break
            y += subsize[0] - over_lap
        return cut_results


    def stretch_16to8_2(self, bands, lower_percent=2, higher_percent=98):
        h, w = bands.shape
        a = 0
        b = 255
        c = np.percentile(bands, lower_percent)
        d = np.percentile(bands, higher_percent)
        rate = (b - a) / (d - c)
        bands = np.ravel(bands)
        size = bands.size // 2
        t1 = a + (bands[:size] - c) * rate
        t1[t1 < a] = a
        t1 = t1.astype(np.uint8)
        t2 = a + (bands[size:] - c) * rate
        t2[t2 > b] = b
        t2 = t2.astype(np.uint8)

        bands[:size] = t1[:]
        bands[size:] = t2[:]
        return bands.reshape((h, w))
