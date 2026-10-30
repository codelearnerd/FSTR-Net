import numpy as np
import torch
import torch.nn.functional as F

def scale_coords_ft(img1_shape, coords, img0_shape, ratio_pad=None):

    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, 0] -= pad[0]
    coords[:, 1] -= pad[1]
    coords /= gain
    return coords

def fft_1(coeffs,npts):
    n = (len(coeffs) - 2) // 4
    t = np.linspace(0, 1, npts, endpoint=False)
    x = np.ones_like(t) * coeffs[0]
    y = np.ones_like(t) * coeffs[1]
    offset = 2
    for i in range(n):
        assert offset + 4 <= len(coeffs)
        a, b, c, d = coeffs[offset : offset + 4]
        x += a * np.cos(2*np.pi*(i+1)*t) + b * np.sin(2*np.pi*(i+1)*t)
        y += c * np.cos(2*np.pi*(i+1)*t) + d * np.sin(2*np.pi*(i+1)*t)
        offset+=4
    return x,y
def fft_pts(coeffs, npts):


    n = (len(coeffs) - 2) // 4
    t = np.linspace(0, 1, npts, endpoint=False)


    x = np.ones_like(t) * coeffs[0]
    y = np.ones_like(t) * coeffs[1]

    offset = 2
    for i in range(n):
        assert offset + 4 <= len(coeffs)
        a, b, c, d = coeffs[offset : offset + 4]
        x += a * np.cos(2*np.pi*(i+1)*t) + b * np.sin(2*np.pi*(i+1)*t)
        y += c * np.cos(2*np.pi*(i+1)*t) + d * np.sin(2*np.pi*(i+1)*t)
        offset += 4


    xy = np.column_stack((x, y))
    return xy


def ffts_1(coeffs, npts):


    if isinstance(coeffs,np.ndarray):
        coeffs = coeffs.astype(np.float64)
        nt, total_len = coeffs.shape
        n = (total_len - 2) // 4
        t = np.linspace(0, 1, npts, endpoint=False)


        x = coeffs[:, 0:1] + np.zeros((nt, npts), dtype=np.float64)
        y = coeffs[:, 1:2] + np.zeros((nt, npts), dtype=np.float64)

        offset = 2
        for i in range(n):
            a = coeffs[:, offset + 0][:, None]
            b = coeffs[:, offset + 1][:, None]
            c = coeffs[:, offset + 2][:, None]
            d = coeffs[:, offset + 3][:, None]

            angle = 2 * np.pi * (i + 1) * t
            cos_t = np.cos(angle)[None, :]
            sin_t = np.sin(angle)[None, :]

            x += a * cos_t + b * sin_t
            y += c * cos_t + d * sin_t

            offset += 4
    else:

        coeffs = coeffs.to(torch.float64)
        device, dtype = coeffs.device, coeffs.dtype

        nt, total_len = coeffs.shape
        n = (total_len - 2) // 4


        t = torch.arange(0, npts, device=device, dtype=dtype) / npts


        x = coeffs[:, 0:1] + torch.zeros((nt, npts), device=device, dtype=dtype)
        y = coeffs[:, 1:2] + torch.zeros((nt, npts), device=device, dtype=dtype)

        offset = 2
        for i in range(n):
            a = coeffs[:, offset + 0][:, None]
            b = coeffs[:, offset + 1][:, None]
            c = coeffs[:, offset + 2][:, None]
            d = coeffs[:, offset + 3][:, None]

            angle = 2 * torch.pi * (i + 1) * t
            cos_t = torch.cos(angle)[None, :]
            sin_t = torch.sin(angle)[None, :]

            x = x + a * cos_t + b * sin_t
            y = y + c * cos_t + d * sin_t

            offset += 4

    return x, y

def ffts_1_slow(coeffs_batch, npts):


    nt = coeffs_batch.shape[0]
    x_list = []
    y_list = []
    for i in range(nt):
        x, y = fft_1(coeffs_batch[i], npts)
        x_list.append(x)
        y_list.append(y)
    x_arr = np.stack(x_list, axis=0)
    y_arr = np.stack(y_list, axis=0)
    return x_arr, y_arr

def ft_correct_clockwise(ft_code, offset=2):


    nt, m = ft_code.shape
    nf = (m - offset) // 4


    an = ft_code[:, offset+0::4]
    bn = ft_code[:, offset+1::4]
    cn = ft_code[:, offset+2::4]
    dn = ft_code[:, offset+3::4]

    idx = torch.arange(1, nf+1, device=ft_code.device).float().view(1, -1)


    area = torch.pi * torch.sum(idx * (an * dn - bn * cn), dim=1)


    corrected = area < 0


    ft_code[:, offset+1::4][corrected] *= -1
    ft_code[:, offset+3::4][corrected] *= -1

    return corrected


def _get_covariance_matrix_pts4A(boxes,method=1,ft=False):
    assert boxes.dim() == 3 and boxes.size(-1) == 8
    nt,nf = boxes.shape[:2]
    pts = boxes.view(nt, nf, 4, 2)


    cxy = pts.mean(dim=-2)


    pts_centered = pts - cxy[:,:, None, :]

    if method==0:


        S = torch.matmul(pts_centered.transpose(-2, -1), pts_centered).to(torch.float32) / 4.0


        evals, evecs = torch.linalg.eigh(S)
        v = evecs[..., :, 1]

        v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
        vx, vy = v[..., 0], v[..., 1]


        v_perp = torch.stack([-vy, vx], dim=-1)
        proj_x = (pts_centered * v.unsqueeze(-2)).sum(dim=-1)
        proj_y = (pts_centered * v_perp.unsqueeze(-2)).sum(dim=-1)
        w = proj_x.max(dim=-1).values - proj_x.min(dim=-1).values
        h = proj_y.max(dim=-1).values - proj_y.min(dim=-1).values


        a = (w ** 2) / 12.0
        b = (h ** 2) / 12.0


        cos = vx
        sin = vy
        cos2 = cos * cos
        sin2 = sin * sin

        sigma_xx = a * cos2 + b * sin2
        sigma_yy = a * sin2 + b * cos2
        sigma_xy = (a - b) * cos * sin
    elif method==1:

        cov = torch.matmul(pts_centered.transpose(-2,-1), pts_centered).to(torch.float32) / 4.0

        sigma_xx = cov[..., 0, 0]
        sigma_yy = cov[..., 1, 1]
        sigma_xy = cov[..., 0, 1]
    else:
        assert method==2


        a = (pts_centered[..., 0].pow(2).mean(dim=-1))
        b = (pts_centered[..., 1].pow(2).mean(dim=-1))
        c = (pts_centered[..., 0] * pts_centered[..., 1]).mean(dim=-1)


        x_min = pts[..., 0].min(dim=-1)[0]
        x_max = pts[..., 0].max(dim=-1)[0]
        y_min = pts[..., 1].min(dim=-1)[0]
        y_max = pts[..., 1].max(dim=-1)[0]

        width = x_max - x_min
        height = y_max - y_min


        eigenvalues, eigenvectors = torch.linalg.eigh(torch.stack([
            torch.stack([a, c], dim=-1),
            torch.stack([c, b], dim=-1)
        ], dim=-2))


        angle = torch.atan2(eigenvectors[..., 1, 0], eigenvectors[..., 0, 0])


        a_cov = width.pow(2) / 12
        b_cov = height.pow(2) / 12

        if ft:
            cos = (1 / (1 + angle.pow(2))).sqrt()
            sin = angle * cos
        else:
            cos = angle.cos()
            sin = angle.sin()

        cos2 = cos.pow(2)
        sin2 = sin.pow(2)

        sigma_xx = a_cov * cos2 + b_cov * sin2
        sigma_yy = a_cov * sin2 + b_cov * cos2
        sigma_xy = (a_cov - b_cov) * cos * sin

    return sigma_xx, sigma_yy, sigma_xy, cxy

def probiou_pts4(obb1, obb2, eps=1e-3, method=1):
    an1, bn1, cn1, cxy1 = _get_covariance_matrix_pts4A(obb1,method)
    cx1, cy1 = cxy1[..., 0], cxy1[..., 1]
    an2, bn2, cn2, cxy2 = _get_covariance_matrix_pts4A(obb2,method)
    cx2, cy2 = cxy2[..., 0], cxy2[..., 1]

    det = (an1 + an2) * (bn1 + bn2) - (cn1 + cn2).pow(2)
    assert torch.all(det != 0).item()

    t1 = (((an1 + an2) * (cy1 - cy2).pow(2) + (bn1 + bn2) * (cx1 - cx2).pow(2)) / (det + eps)) * 0.25
    t2 = (((cn1 + cn2) * (cx2 - cx1) * (cy1 - cy2)) / (det + eps)) * 0.5

    det1 = (an1 * bn1 - cn1.pow(2))
    det2 = (an2 * bn2 - cn2.pow(2))

    t3_arg = det / (4 * (det1.clamp_(0) * det2.clamp_(0) + 10*eps).sqrt())

    t3 = (t3_arg.clamp(min=eps, max=10.0)).log() * 0.5

    bd = (t1 + t2 + t3).clamp(min=eps, max=100.0)
    assert torch.all((1.0 - (-bd).exp() + eps) >= 0.0).item()
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd

    return iou

def solve_cos_sin_closed_form(coef0: torch.Tensor, coef: torch.Tensor, eps: float = 1e-8, normalize: bool = True):


    assert coef0.ndim == 2 and coef0.shape[1] == 4
    assert coef.ndim == 2 and coef.shape[1] == 4
    an0, bn0, cn0, dn0 = coef0.unbind(dim=1)
    an,  bn,  cn,  dn  = coef.unbind(dim=1)

    if normalize:

        norm_ab0 = torch.sqrt(an0**2 + bn0**2 + eps)
        norm_ab  = torch.sqrt(an**2  + bn**2  + eps)

        scale_ab = norm_ab0 / norm_ab
        an = an * scale_ab
        bn = bn * scale_ab

        norm_cd0 = torch.sqrt(cn0**2 + dn0**2 + eps)
        norm_cd  = torch.sqrt(cn**2  + dn**2  + eps)

        scale_cd = norm_cd0 / norm_cd
        cn = cn * scale_cd
        dn = dn * scale_cd


    if 1:

        S = an0 * an0 + bn0 * bn0 + cn0 * cn0 + dn0 * dn0
        S = S + eps


        Atb0 = an0 * an + bn0 * bn + cn0 * cn + dn0 * dn
        Atb1 = bn0 * an - an0 * bn + dn0 * cn - cn0 * dn

        cos = Atb0 / S
        sin = Atb1 / S

        cos_sin = torch.stack([cos, sin], dim=1)
    else:
        S = an0 * an0 + bn0 * bn0

        Atb0 = an0 * an + bn0 * bn
        Atb1 = bn0 * an - an0 * bn
        S = S + eps
        cos_ab = Atb0 / S
        sin_ab = Atb1 / S

        S = cn0 * cn0 + dn0 * dn0

        Atb0 = cn0 * cn + dn0 * dn
        Atb1 = dn0 * cn - cn0 * dn
        S = S + eps
        cos_cd = Atb0 / S
        sin_cd = Atb1 / S

        cos = (cos_ab+cos_cd)*0.5
        sin = (sin_ab+sin_cd)*0.5
        cos_sin = torch.stack([cos, sin], dim=1)

    if normalize:
        norm = torch.norm(cos_sin, dim=1, keepdim=True).clamp_min(eps)
        cos_sin = cos_sin / norm

    return cos_sin
def solve_cos_sin_closed_form2(coef0: torch.Tensor, coef: torch.Tensor,
                              eps: float = 1e-4, normalize: bool = True):


    assert coef0.ndim == 2 and coef.ndim == 2
    assert coef0.shape == coef.shape, "coef0 and coef must have same shape"
    nt, four_nf = coef0.shape
    assert four_nf % 4 == 0, "second dim must be multiple of 4"
    nf = four_nf // 4
    device = coef0.device
    dtype = coef0.dtype


    coef0 = coef0.view(nt, nf, 4)
    coef  = coef.view(nt, nf, 4)

    a0 = coef0[..., 0]
    b0 = coef0[..., 1]
    c0 = coef0[..., 2]
    d0 = coef0[..., 3]

    a = coef[..., 0]
    b = coef[..., 1]
    c = coef[..., 2]
    d = coef[..., 3]


    if normalize:

        norm_ab0 = torch.sqrt(a0 * a0 + b0 * b0)
        norm_ab  = torch.sqrt(a  * a  + b  * b)
        scale_ab = norm_ab0 / (norm_ab + eps)
        a = a * scale_ab
        b = b * scale_ab

        norm_cd0 = torch.sqrt(c0 * c0 + d0 * d0)
        norm_cd  = torch.sqrt(c  * c  + d  * d)
        scale_cd = norm_cd0 / (norm_cd + eps)
        c = c * scale_cd
        d = d * scale_cd


    real_dtype = torch.float32
    a0f = a0.to(real_dtype)
    b0f = b0.to(real_dtype)
    c0f = c0.to(real_dtype)
    d0f = d0.to(real_dtype)

    af = a.to(real_dtype)
    bf = b.to(real_dtype)
    cf = c.to(real_dtype)
    df = d.to(real_dtype)


    u1 = torch.complex(a0f, b0f)
    v1 = torch.complex(af, bf)

    u2 = torch.complex(c0f, d0f)
    v2 = torch.complex(cf, df)


    den = (a0f * a0f + b0f * b0f) + (c0f * c0f + d0f * d0f)
    den = den + eps

    num = torch.conj(u1) * v1 + torch.conj(u2) * v2
    x = num / den


    if nf == 1:
        r = x[:, 0]
    else:
        xk  = x[:, :-1]
        xk1 = x[:, 1:]
        wk  = den[:, :-1]


        num_r = torch.sum(wk * torch.conj(xk) * xk1, dim=1)
        den_r = torch.sum(wk * (xk.real * xk.real + xk.imag * xk.imag), dim=1) + eps

        r = num_r / den_r


    if normalize:
        r = r / (torch.abs(r) + eps)

    cos = r.real.to(dtype)
    sin = r.imag.to(dtype)
    cos_sin = torch.stack([cos, sin], dim=1)


    if 1:
        norm = torch.norm(cos_sin, dim=1, keepdim=True).clamp_min(eps)
        cos_sin = cos_sin / norm

    return cos_sin

def rotate_high_orders(abcd1, coefs: torch.Tensor, cos_sin: torch.Tensor, offset1=2):


    offset_cs = 1 if abcd1 is not None else 0
    cos, sin = cos_sin[:,offset_cs:, 0], cos_sin[:,offset_cs:, 1]
    offset = offset1+4 if abcd1 is not None else offset1
    coefs4 = coefs[:,offset:]
    nt, nf4 = coefs4.shape
    assert nf4 % 4 == 0
    nf = nf4 // 4


    coef_split = coefs4.view(nt, nf, 4)
    an0, bn0, cn0, dn0 = coef_split.unbind(-1)


    an = an0 * cos + bn0 * sin
    bn = -an0 * sin + bn0 * cos
    cn = cn0 * cos + dn0 * sin
    dn = -cn0 * sin + dn0 * cos

    rolled4f = torch.stack([an, bn, cn, dn], dim=-1).reshape(nt, nf4)
    if abcd1 is not None:
        rolled = torch.cat([coefs[:,:offset1], abcd1, rolled4f], dim=-1)
    else:
        rolled = torch.cat([coefs[:,:offset1], rolled4f], dim=-1)
    assert rolled.shape[-1]==coefs.shape[-1]
    return rolled

def ft2dir(coefs,cen=0):

    abcd = coefs[:, 2:6]
    a1, b1, c1, d1 = torch.split(abcd, 1, dim=-1)


    if cen==0:
        a1,b1,c1,d1 = a1.float(),b1.float(),c1.float(),d1.float()
        cos_sin2t = torch.stack([a1**2 + c1**2 - b1**2 - d1**2, 2*(a1*b1 + c1*d1)],dim=1).squeeze(2)
        cos_sin2t = F.normalize(cos_sin2t,p=2,dim=-1)

        cos_t = torch.sqrt((1+cos_sin2t[:,0])/2)
        sin_t = torch.sqrt((1-cos_sin2t[:,0])/2)
        sin_t[cos_sin2t[:,1]<0] *= -1

        cos_sin = torch.stack([cos_t,sin_t],dim=1)
    elif cen==1:
        tan2t = (2*(a1*b1 + c1*d1))/(a1**2 + c1**2 - b1**2 - d1**2)


        cos2t = torch.sqrt(1 / (1 + tan2t**2)).view(-1,1)
        cos_sin = cos2t @ torch.tensor([[0.5, -0.5]], dtype=abcd.dtype, device=coefs.device) + 0.5
        cos_sin = torch.sqrt(cos_sin)
        index = torch.where(tan2t < 0)[0]
        cos_sin[index, 1] *= -1
    else:

        x = a1**2 + c1**2 - b1**2 - d1**2
        y = 2*(a1*b1 + c1*d1)


        phi2 = torch.atan2(y + 0.0, x + 0.0)
        phi  = 0.5 * phi2

        cos_t = torch.cos(phi)
        sin_t = torch.sin(phi)
        cos_sin = torch.cat([cos_t,sin_t],dim=1)

    return cos_sin

def ft2dir_k(coefs):
    assert (coefs.shape[1]-2) % 4 == 0
    nt,nf = coefs.shape[0],(coefs.shape[1]-2)//4
    abcd = coefs[:, 2:].view(nt,nf,4)
    an, bn, cn, dn = abcd[..., 0], abcd[..., 1], abcd[..., 2], abcd[..., 3]

    if 1:

        x = an**2 + cn**2 - bn**2 - dn**2
        y = 2*(an*bn + cn*dn)


        phi2 = torch.atan2(y + 0.0, x + 0.0)
        phi  = 0.5 * phi2

        cos_t = torch.cos(phi)
        sin_t = torch.sin(phi)
    else:
        cos_sin2t = torch.stack([an**2 + cn**2 - bn**2 - dn**2, 2*(an*bn + cn*dn)],dim=2)
        cos_sin2t = F.normalize(cos_sin2t,p=2,dim=-1)

        cos_t = torch.sqrt((1+cos_sin2t[...,0])/2)
        sin_t = torch.sqrt((1-cos_sin2t[...,0])/2).clone()
        sin_t[cos_sin2t[...,1]<0] *= -1

    cos_sin = torch.stack([cos_t,sin_t],dim=2)

    return cos_sin

def ft2vector(coefs,cen=0):

    if not torch.is_tensor(coefs):
        coefs = torch.from_numpy(coefs)
    cos_sin = ft2dir(coefs,cen)
    cos_t,sin_t = cos_sin[:,0],cos_sin[:,1]
    a1,b1,c1,d1 = torch.chunk(coefs[:,2:6],4,dim=1)

    ap= a1*cos_t+b1*sin_t
    cp= c1*cos_t+d1*sin_t
    bp=-a1*sin_t+b1*cos_t
    dp=-c1*sin_t+d1*cos_t
    return ap,bp,cp,dp

def angle_multiple(cs: torch.Tensor, m: int):


    c, s = cs[:, 0], cs[:, 1]

    cos_k = c.clone()
    sin_k = s.clone()

    outs = [torch.stack([cos_k, sin_k], dim=-1)]

    for _ in range(1, m):
        new_cos = cos_k * c - sin_k * s
        new_sin = sin_k * c + cos_k * s
        outs.append(torch.stack([new_cos, new_sin], dim=-1))
        cos_k, sin_k = new_cos, new_sin

    return torch.stack(outs, dim=1)

def ft2pts(coefs,cen=0):
    cos_sin = ft2dir(coefs,cen)
    abcd = coefs[:, 2:6]

    xc, yc = torch.split(coefs[:, :2].clone(), 1, dim=-1)
    an1, bn1, cn1, dn1 = torch.split(abcd, 1, dim=-1)
    m_sin = cos_sin[:, 1:2]
    m_cos = cos_sin[:, 0:1]
    scale=1.0
    a1 = scale*(an1 * m_cos + bn1 * m_sin)
    b1 = scale*(-an1 * m_sin + bn1 * m_cos)
    c1 = scale*(cn1 * m_cos + dn1 * m_sin)
    d1 = scale*(-cn1 * m_sin + dn1 * m_cos)
    P0=torch.cat([xc-a1-b1,yc-c1-d1], dim=-1).view(-1, 2)
    P1=torch.cat([xc-a1+b1,yc-c1+d1], dim=-1).view(-1, 2)
    P2=torch.cat([xc+a1+b1,yc+c1+d1], dim=-1).view(-1, 2)
    P3=torch.cat([xc+a1-b1,yc+c1-d1], dim=-1).view(-1, 2)
    points = torch.cat([P0, P1, P2, P3], dim=1)
    return points
def ft_rot_normlize(coefs):
    cos_sin1 = ft2dir(coefs)
    assert ((coefs.shape[-1]-2) % 4)==0
    nf = (coefs.shape[-1]-2)//4
    cos_sin = angle_multiple(cos_sin1, nf)

    nt = coefs.shape[0]
    abcd = coefs[:, 2:].view(nt,nf,4)

    an0, bn0, cn0, dn0 = abcd[..., 0], abcd[..., 1], abcd[..., 2], abcd[..., 3]
    m_cos = cos_sin[..., 0]
    m_sin = cos_sin[..., 1]
    an =  an0 * m_cos + bn0 * m_sin
    bn = -an0 * m_sin + bn0 * m_cos
    cn =  cn0 * m_cos + dn0 * m_sin
    dn = -cn0 * m_sin + dn0 * m_cos
    coefs_ft = torch.stack([an,bn,cn,dn],dim=-1).view(nt,nf*4)
    return torch.cat([coefs[:,:2],coefs_ft],dim=-1),cos_sin

def ft2pts_k(coefs,use_center=True):


    nf = (coefs.shape[-1]-2)//4

    coefs2, cos_sin = ft_rot_normlize(coefs)

    nt = coefs2.shape[0]
    abcd = coefs2[:, 2:].view(nt,nf,4)

    if 1:
        an, bn, cn, dn = abcd[..., 0], abcd[..., 1], abcd[..., 2], abcd[..., 3]

    else:
        an0, bn0, cn0, dn0 = abcd[..., 0], abcd[..., 1], abcd[..., 2], abcd[..., 3]
        m_cos = cos_sin[..., 0]
        m_sin = cos_sin[..., 1]
        an =  an0 * m_cos + bn0 * m_sin
        bn = -an0 * m_sin + bn0 * m_cos
        cn =  cn0 * m_cos + dn0 * m_sin
        dn = -cn0 * m_sin + dn0 * m_cos

    P0=torch.stack([-an-bn,-cn-dn], dim=-1)
    P1=torch.stack([-an+bn,-cn+dn], dim=-1)
    P2=torch.stack([+an+bn,+cn+dn], dim=-1)
    P3=torch.stack([+an-bn,+cn-dn], dim=-1)
    if use_center:
        xc, yc = torch.split(coefs2[:, :2], 1, dim=-1)
        center = torch.stack([xc,yc],dim=-1)
        P0[:,0:1,:] += center
        P1[:,0:1,:] += center
        P2[:,0:1,:] += center
        P3[:,0:1,:] += center
    points = torch.cat([P0, P1, P2, P3], dim=-1)
    if 0:
        v = P1 - P0
        norm = torch.norm(v, dim=2, keepdim=True)
        v_unit = v / (norm + 1e-8)
    return points,cos_sin,coefs2
