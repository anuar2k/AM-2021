#!/usr/bin/env python3

import numpy as np
import os
from subprocess import check_output
from tempfile import NamedTemporaryFile as tmp

CSV_DELIM = ","


def write_matrix(matrix, file):
    rows, cols = np.shape(matrix)
    file.write(f"{rows}x{cols}\n")
    np.savetxt(file, matrix, fmt="%f", delimiter=CSV_DELIM)
    file.flush()


def read_matrix(file):
    rows, cols = (int(x) for x in file.readline().split("x"))
    res = np.loadtxt(file, delimiter=CSV_DELIM)

    assert rows, cols == np.shape(res)
    return res


def run_alg(input_name, out_name, alg):
    command = [
        "./prog.out",
        input_name,
        out_name,
        alg
    ]

    output = check_output(command, encoding="utf-8")

    clk_per_sec, clk_calc = (int(x) for x in output.split("\n")[:2])
    with open(out_name) as out_file:
        res = read_matrix(out_file)

    return clk_calc / clk_per_sec, res


def calc(input, alg):
    with tmp(mode="w+", delete=False) as input_file, tmp(mode="w+", delete=False) as out_file:
        write_matrix(input, input_file)
        input_name = input_file.name
        out_name = out_file.name

    res = run_alg(input_name, out_name, alg)
    os.remove(input_name)
    os.remove(out_name)
    return res


if __name__ == "__main__":
    input = np.array([
        [1, -1, 2, 2],
        [2, -2, 1, 0],
        [-1, 2, 1, -2],
        [2, -1, 4, 0]
    ])

    time1, result1 = calc(input, "dense")
    time2, result2 = calc(input, "coord")
    time3, result3 = calc(input, "dense_skip")
    print(time1, result1)
    print(time2, result2)
    print(time3, result3)
    print(np.array_equal(result1, result2))
    print(np.array_equal(result2, result3))
    print(np.array_equal(result1, result3))
