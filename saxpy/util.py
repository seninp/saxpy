"""Some helpers."""
import re


def read_ucr_data(fname):
    data = []

    with open(fname, 'r') as fp:
        read_lines = fp.readlines()
        for line in read_lines:
            tokens = re.split("\\s+", line.strip())

            t0 = tokens.pop(0)
            if re.match("^\d+?\.\d+?e[\+\-]?\d+?$", t0) is None:
                class_label = t0
            else:
                class_label = str(int(float(t0)))

            data_line = []
            for token in tokens:
                data_line.append(float(token))

            data.append((class_label, data_line))

    res = {}
    for key, arr in data:
        if key in res.keys():
            res[key].append(arr)
        else:
            dat = [arr]
            res[key] = dat

    return res