import numpy as np

def yuv_import8(file_path, dims, num_frames=1, start_frame=0, frames=None, yuv444=False):
    """
    Import frame images from a YUV file.
    :param file_path: Path of the file.
    :param dims: (height, width) of the frames.
    :param num_frames: Number of the consecutive frames to be imported.
    :param start_frame: Index of the frame to be started. The first frame is indexed as 0.
    :param frames: Indexes of the frames to be imported. Inconsecutive frames are supported.
    :param yuv444: Whether the YUV file is in YUV444 mode.
    :return: Y, U, V, all as the numpy ndarray.
    """

    fp = open(file_path, 'rb')
    ratio = 3 if yuv444 else 1.5
    blk_size = np.uint64((np.prod(dims) * ratio))
    if frames is None:
        assert num_frames > 0
        fp.seek(np.uint64(blk_size * start_frame), 0)

    height, width = dims
    Y = []
    U = []
    V = []
    if yuv444:
        height_half = height
        width_half = width
    else:
        height_half = height // 2
        width_half = width // 2

    if frames is not None:
        previous_frame = -1
        for frame in frames:
            fp.seek(blk_size * (frame - previous_frame - 1), 1)
            Yt = np.fromfile(fp, dtype=np.uint8, count=width * height).reshape((height, width))
            Ut = np.fromfile(fp, dtype=np.uint8, count=width_half * height_half).reshape((height_half, width_half))
            Vt = np.fromfile(fp, dtype=np.uint8, count=width_half * height_half).reshape((height_half, width_half))
            previous_frame = frame
            Y = Y + [Yt]
            U = U + [Ut]
            V = V + [Vt]

    else:
        for i in range(num_frames):
            Yt = np.fromfile(fp, dtype=np.uint8, count=width * height).reshape((height, width))
            Ut = np.fromfile(fp, dtype=np.uint8, count=width_half * height_half).reshape((height_half, width_half))
            Vt = np.fromfile(fp, dtype=np.uint8, count=width_half * height_half).reshape((height_half, width_half))
            Y = Y + [Yt]
            U = U + [Ut]
            V = V + [Vt]


    fp.close()
    return np.array(Y), np.array(U), np.array(V)

def yuv_import(file_path, dims, num_frames=1, start_frame=0, frames=None, yuv444=False):
    """
    Import frame images from a YUV file.
    :param file_path: Path of the file.
    :param dims: (height, width) of the frames.
    :param num_frames: Number of the consecutive frames to be imported.
    :param start_frame: Index of the frame to be started. The first frame is indexed as 0.
    :param frames: Indexes of the frames to be imported. Inconsecutive frames are supported.
    :param yuv444: Whether the YUV file is in YUV444 mode.
    :return: Y, U, V, all as the numpy ndarray.
    """

    fp = open(file_path, 'rb')
    ratio = 3 if yuv444 else 1.5
    blk_size = np.uint64((np.prod(dims) * ratio*2))
    if frames is None:
        assert num_frames > 0
        fp.seek(np.uint64(blk_size * start_frame), 0)

    height, width = dims
    Y = []
    U = []
    V = []
    if yuv444:
        height_half = height
        width_half = width
    else:
        height_half = height // 2
        width_half = width // 2

    if frames is not None:
        previous_frame = -1
        for frame in frames:
            fp.seek(blk_size * (frame - previous_frame - 1), 1)
            Yt = np.fromfile(fp, dtype=np.uint16, count=width * height).reshape((height, width))
            Ut = np.fromfile(fp, dtype=np.uint16, count=width_half * height_half).reshape((height_half, width_half))
            Vt = np.fromfile(fp, dtype=np.uint16, count=width_half * height_half).reshape((height_half, width_half))
            previous_frame = frame
            Y = Y + [Yt]
            U = U + [Ut]
            V = V + [Vt]

    else:
        for i in range(num_frames):
            Yt = np.fromfile(fp, dtype=np.uint16, count=width * height).reshape((height, width))
            Ut = np.fromfile(fp, dtype=np.uint16, count=width_half * height_half).reshape((height_half, width_half))
            Vt = np.fromfile(fp, dtype=np.uint16, count=width_half * height_half).reshape((height_half, width_half))
            Y = Y + [Yt]
            U = U + [Ut]
            V = V + [Vt]


    fp.close()
    return np.array(Y), np.array(U), np.array(V)

def yuv_import_8bits(file_path, dims, num_frames=1, start_frame=0, frames=None, yuv444=False):
    """
    Import frame images from a YUV file.
    :param file_path: Path of the file.
    :param dims: (height, width) of the frames.
    :param num_frames: Number of the consecutive frames to be imported.
    :param start_frame: Index of the frame to be started. The first frame is indexed as 0.
    :param frames: Indexes of the frames to be imported. Inconsecutive frames are supported.
    :param yuv444: Whether the YUV file is in YUV444 mode.
    :return: Y, U, V, all as the numpy ndarray.
    """

    fp = open(file_path, 'rb')
    ratio = 3 if yuv444 else 1.5
    blk_size = np.uint64(int(np.prod(dims) * ratio))
    if frames is None:
        assert num_frames > 0
        fp.seek(np.uint64(blk_size * start_frame), 0)

    height, width = dims
    Y = []
    U = []
    V = []
    if yuv444:
        height_half = height
        width_half = width
    else:
        height_half = height // 2
        width_half = width // 2

    if frames is not None:
        previous_frame = -1
        for frame in frames:
            fp.seek(blk_size * (frame - previous_frame - 1), 1)
            Yt = np.fromfile(fp, dtype=np.uint16, count=width * height).reshape((height, width))
            Ut = np.fromfile(fp, dtype=np.uint16, count=width_half * height_half).reshape((height_half, width_half))
            Vt = np.fromfile(fp, dtype=np.uint16, count=width_half * height_half).reshape((height_half, width_half))
            previous_frame = frame
            Y = Y + [Yt]
            U = U + [Ut]
            V = V + [Vt]

    else:
        for i in range(num_frames):
            Yt = np.fromfile(fp, dtype=np.uint16, count=width * height).reshape((height, width))
            Ut = np.fromfile(fp, dtype=np.uint16, count=width_half * height_half).reshape((height_half, width_half))
            Vt = np.fromfile(fp, dtype=np.uint16, count=width_half * height_half).reshape((height_half, width_half))
            Yt=Yt/1023*255
            Ut=Ut/1023*255
            Vt=Vt/1023*255
            Yt = Yt.clip(0, 255).round().astype(np.uint8)
            Ut = Ut.clip(0, 255).round().astype(np.uint8)
            Vt = Vt.clip(0, 255).round().astype(np.uint8)
            Y = Y + [Yt]
            U = U + [Ut]
            V = V + [Vt]

    fp.close()
    return np.array(Y), np.array(U), np.array(V)


def yuv_export(Yt, Ut, Vt, file_path, dims, num_frames=1, start_frame=0, frames=None, yuv444=False):
    """
    Export frame images from a YUV file.
    :param file_path: Path of the file.
    :param dims: (height, width) of the frames.
    :param num_frames: Number of the consecutive frames to be imported.
    :param start_frame: Index of the frame to be started. The first frame is indexed as 0.
    :param frames: Indexes of the frames to be imported. Inconsecutive frames are supported.
    :param yuv444: Whether the YUV file is in YUV444 mode.
    :return: Y, U, V, all as the numpy ndarray.
    """

    fp = open(file_path, 'a')


    height, width = dims
    Y = []
    U = []
    V = []
    if yuv444:
        height_half = height
        width_half = width
    else:
        height_half = height // 2
        width_half = width // 2

    for i in range(num_frames):
            Yt[i].tofile(fp)
            Ut[i].tofile(fp)
            Vt[i].tofile(fp)
  

    fp.close()
    

def y400_export(Yt, file_path, dims, num_frames=1, start_frame=0, frames=None, yuv444=False):
    """
    Export frame images from a YUV file.
    :param file_path: Path of the file.
    :param dims: (height, width) of the frames.
    :param num_frames: Number of the consecutive frames to be imported.
    :param start_frame: Index of the frame to be started. The first frame is indexed as 0.
    :param frames: Indexes of the frames to be imported. Inconsecutive frames are supported.
    :param yuv444: Whether the YUV file is in YUV444 mode.
    :return: Y, U, V, all as the numpy ndarray.
    """

    fp = open(file_path, 'a')


    height, width = dims
    Y = []
    
    if yuv444:
        height_half = height
        width_half = width
    else:
        height_half = height // 2
        width_half = width // 2

    for i in range(num_frames):
            Yt[i].tofile(fp)
            
    fp.close()


def yuv2rgb(Y, U, V):
    """
    Convert YUV to RGB.
    """

    if not Y.shape == U.shape:
        U = U.repeat(2, axis=1).repeat(2, axis=2).astype(np.float64)
        V = V.repeat(2, axis=1).repeat(2, axis=2).astype(np.float64)

    Y = Y.astype(np.float64)
    U = U.astype(np.float64)
    V = V.astype(np.float64)
    U -= 128.0
    V -= 128.0

    rr = 1.001574765442552 * Y + 0.002770649292941 * U + 1.574765442551769 * V
    gg = 0.999531875325065 * Y - 0.188148872370914 * U - 0.468124674935631 * V
    bb = 1.000000105739993 * Y + 1.855609881994441 * U + 1.057399924810358e-04 * V

    rr = rr.clip(0, 255).round().astype(np.uint8)
    gg = gg.clip(0, 255).round().astype(np.uint8)
    bb = bb.clip(0, 255).round().astype(np.uint8)

    return np.stack((rr, gg, bb), axis=1)


def YUV2RGB( yuv ):
      
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
    
    rgb = np.dot(yuv,m)
    rgb[:,:,0]-=179.45477266423404
    rgb[:,:,1]+=135.45870971679688
    rgb[:,:,2]-=226.8183044444304

    rgb = rgb.astype(np.uint8)
    return rgb

def yuv2rgb_10bits(Y, U, V):
    """
    Convert YUV to RGB.
    """

    if not Y.shape == U.shape:
        U = U.repeat(2, axis=1).repeat(2, axis=2).astype(np.float64)
        V = V.repeat(2, axis=1).repeat(2, axis=2).astype(np.float64)
    
    Y = np.interp(Y, (0, 1023), (0, +255))
    U = np.interp(U, (0, 1023), (0, +255))
    V = np.interp(V, (0, 1023), (0, +255))

    yuv = np.array([Y[0], U[0], V[0]])

    #rgb = YUV2RGB( np.transpose(yuv, (1,2,0)) )

    Y = Y.astype(np.float64)
    U = U.astype(np.float64)
    V = V.astype(np.float64)
    U -= 128.0
    V -= 128.0

    #rr = 1.001574765442552 * Y + 0.002770649292941 * U + 1.574765442551769 * V
    #gg = 0.999531875325065 * Y - 0.188148872370914 * U - 0.468124674935631 * V
    #bb = 1.000000105739993 * Y + 1.855609881994441 * U + 1.057399924810358e-04 * V

    a = 0.2627
    b = 0.6780
    c = 0.0593
    d = 1.8814
    e= 1.4747

    rr = Y + e * V
    gg = Y - (a * e / b) * V - (c * d / b) * U
    bb = Y + d * U

    rr = rr.clip(0, 255).round().astype(np.uint8)
    gg = gg.clip(0, 255).round().astype(np.uint8)
    bb = bb.clip(0, 255).round().astype(np.uint8)

    return np.stack((bb, gg, rr), axis=1)

def yuv2rgb_8_bt2020(Y, U, V):
    """
    Convert YUV to RGB.
    """

    if not Y.shape == U.shape:
        U = U.repeat(2, axis=1).repeat(2, axis=2).astype(np.float64)
        V = V.repeat(2, axis=1).repeat(2, axis=2).astype(np.float64)
    
    Y = np.interp(Y, (0, 1023), (0, +255))
    U = np.interp(U, (0, 1023), (0, +255))
    V = np.interp(V, (0, 1023), (0, +255))

    yuv = np.array([Y[0], U[0], V[0]])

    #rgb = YUV2RGB( np.transpose(yuv, (1,2,0)) )

    Y = Y.astype(np.float64)
    U = U.astype(np.float64)
    V = V.astype(np.float64)
    U -= 128.0
    V -= 128.0

    #rr = 1.001574765442552 * Y + 0.002770649292941 * U + 1.574765442551769 * V
    #gg = 0.999531875325065 * Y - 0.188148872370914 * U - 0.468124674935631 * V
    #bb = 1.000000105739993 * Y + 1.855609881994441 * U + 1.057399924810358e-04 * V

    a = 0.2627
    b = 0.6780
    c = 0.0593
    d = 1.8814
    e= 1.4747

    rr = Y + e * V
    gg = Y - (a * e / b) * V - (c * d / b) * U
    bb = Y + d * U

    rr = rr.clip(0, 255).round().astype(np.uint8)
    gg = gg.clip(0, 255).round().astype(np.uint8)
    bb = bb.clip(0, 255).round().astype(np.uint8)

    return np.stack((bb, gg, rr), axis=1)


def yuv2rgb_10_bt2020(Y, U, V):
    """
    Convert YUV to RGB.
    """

    if not Y.shape == U.shape:
        U = U.repeat(2, axis=1).repeat(2, axis=2).astype(np.float64)
        V = V.repeat(2, axis=1).repeat(2, axis=2).astype(np.float64)
    
    #Y = np.interp(Y, (0, 1023), (0, +255))
    #U = np.interp(U, (0, 1023), (0, +255))
    #V = np.interp(V, (0, 1023), (0, +255))

    #yuv = np.array([Y[0], U[0], V[0]])

    #rgb = YUV2RGB( np.transpose(yuv, (1,2,0)) )

    Y = Y.astype(np.float64)
    U = U.astype(np.float64)
    V = V.astype(np.float64)
    U -= 512.0
    V -= 512.0

    #rr = 1.001574765442552 * Y + 0.002770649292941 * U + 1.574765442551769 * V
    #gg = 0.999531875325065 * Y - 0.188148872370914 * U - 0.468124674935631 * V
    #bb = 1.000000105739993 * Y + 1.855609881994441 * U + 1.057399924810358e-04 * V

    a = 0.2627
    b = 0.6780
    c = 0.0593
    d = 1.8814
    e= 1.4747

    rr = Y + e * V
    gg = Y - (a * e / b) * V - (c * d / b) * U
    bb = Y + d * U

    rr = rr.clip(0, 1023).round().astype(np.uint16)
    gg = gg.clip(0, 1023).round().astype(np.uint16)
    bb = bb.clip(0, 1023).round().astype(np.uint16)

    return np.stack((bb, gg, rr), axis=1)

def rgb2yuv_8_bt2020(R, G, B):
    """
    Convert RGB to YUV.
    """
      
    a = 0.2627
    b = 0.6780
    c = 0.0593
    d = 1.8814
    e= 1.4747

    Y  = a * R + b * G + c * B
    U = (B - Y) / d + 128
    V = (R - Y) / e + 128

    return np.array(Y), np.array(U), np.array(V)