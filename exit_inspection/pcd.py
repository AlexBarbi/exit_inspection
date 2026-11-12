"""
PCD reading with numpy only (ascii, binary, binary_compressed/LZF), copied from map_point_picker.py
(src/FAST_LIO_SLAM_ROS2/SC_PGO_ROS2/utils/map_point_picker) so that the runtime nodes need neither open3d nor pcl.
"""

import io
import struct

import numpy as np


def lzf_decompress(src, out_len):
    # LZF as written by PCL's binary_compressed: a control byte < 32 starts a literal run of
    # ctrl + 1 bytes, anything else is a back-reference of length (ctrl >> 5) + 2 (7 means
    # "read one more length byte") at distance ((ctrl & 0x1f) << 8 | next byte) + 1.
    out = bytearray(out_len)
    ip = op = 0
    n = len(src)
    while ip < n:
        ctrl = src[ip]
        ip += 1
        if ctrl < 32:
            ln = ctrl + 1
            out[op:op + ln] = src[ip:ip + ln]
            ip += ln
            op += ln
        else:
            ln = ctrl >> 5
            if ln == 7:
                ln += src[ip]
                ip += 1
            ref = op - ((ctrl & 0x1f) << 8) - src[ip] - 1
            ip += 1
            ln += 2
            d = op - ref
            if d >= ln:
                out[op:op + ln] = out[ref:ref + ln]
            else:  # the reference overlaps the output: it repeats with period d
                out[op:op + ln] = (out[ref:op] * (ln // d + 1))[:ln]
            op += ln
    if op != out_len:
        raise ValueError("corrupt binary_compressed data: got %d bytes, expected %d" % (op, out_len))
    return bytes(out)


def read_pcd_xyz(path):
    """Return the x, y, z columns of a PCD file as an (N, 3) float32 array, NaN points dropped."""
    with open(path, "rb") as f:
        raw = f.read()
    header = {}
    pos = 0
    while True:
        end = raw.index(b"\n", pos)
        line = raw[pos:end].decode("ascii", "replace").strip()
        pos = end + 1
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(" ")
        header[key.upper()] = value.split()
        if key.upper() == "DATA":
            break

    fields = header["FIELDS"]
    sizes = [int(s) for s in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
    num = int(header["POINTS"][0])
    data = header["DATA"][0].lower()
    for axis in "xyz":
        if axis not in fields:
            raise ValueError("%s has no %s field (fields: %s)" % (path, axis, " ".join(fields)))

    kind = {"F": "f", "I": "i", "U": "u"}
    names = [name if name != "_" else "_pad%d" % i for i, name in enumerate(fields)]
    dtypes = [(n, "<%s%d" % (kind[t], s), (c,)) if c > 1 else (n, "<%s%d" % (kind[t], s))
              for n, t, s, c in zip(names, types, sizes, counts)]

    if data == "ascii":
        cols = [fields.index(a) for a in "xyz"]
        # COUNT > 1 fields take several columns; find the column where each axis starts
        starts = np.cumsum([0] + counts[:-1])
        text = raw[pos:].decode("ascii", "replace")
        xyz = np.loadtxt(io.StringIO(text), usecols=[starts[c] for c in cols], dtype=np.float64, ndmin=2)
        xyz = xyz[:num].astype(np.float32)
    elif data == "binary":
        cloud = np.frombuffer(raw, dtype=np.dtype(dtypes), count=num, offset=pos)
        xyz = np.stack([cloud[a].astype(np.float32) for a in "xyz"], axis=1)
    elif data == "binary_compressed":
        comp_size, raw_size = struct.unpack("<II", raw[pos:pos + 8])
        buf = lzf_decompress(raw[pos + 8:pos + 8 + comp_size], raw_size)
        # binary_compressed stores the fields one after the other (all x, then all y, ...)
        cols = {}
        off = 0
        for (name, t, s, c) in zip(names, types, sizes, counts):
            nbytes = s * c * num
            if name in ("x", "y", "z"):
                cols[name] = np.frombuffer(buf, dtype="<%s%d" % (kind[t], s), count=num * c, offset=off)[::c]
            off += nbytes
        xyz = np.stack([cols[a].astype(np.float32) for a in "xyz"], axis=1)
    else:
        raise ValueError("unsupported PCD DATA type: %s" % data)

    return xyz[np.isfinite(xyz).all(axis=1)]
