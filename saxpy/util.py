"""Some helpers."""
import re


def read_data_file(fname):
    data = []

    with open(fname, 'r') as fp:
        read_lines = fp.readlines()
        for line in read_lines:
            tokens = re.split("\\s+", line.strip())
            data_line = []
        for token in tokens:
            data_line.append(float(token))
        class_label = data_line.pop(0)
        data.append((class_label, data_line))

    res = {}
    for key, arr in data:
        if key in res.keys():
            res[key].append(arr)
        else:
            dat = [arr]
            res[key] = dat

    return res