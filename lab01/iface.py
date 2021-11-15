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

def run_matmul(a_name, b_name, out_name, order, block_size):
    command = [
        "./prog.out",
        a_name, 
        b_name,
        out_name,
        order, 
        str(block_size),
        "4"
    ]

    output = check_output(command, encoding="utf-8")

    clk_per_sec, clk = (int(x) for x in output.split("\n")[:2])
    with open(out_name) as out_file:
        res = read_matrix(out_file)
    
    return clk / clk_per_sec, res

def calc(a, b, order, block_size=0):
    with tmp(mode="w+", delete=False) as a_file, tmp(mode="w+", delete=False) as b_file, tmp(mode="w+", delete=False) as out_file:
        write_matrix(a, a_file)
        write_matrix(b, b_file)
        a_name = a_file.name
        b_name = b_file.name
        out_name = out_file.name
    
    res = run_matmul(a_name, b_name, out_name, order, block_size)
    os.remove(a_name)
    os.remove(b_name)
    os.remove(out_name)
    return res
        
if __name__ == "__main__":
    a = np.array([
        [0, 1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6, 7],
        [0, 5, 2, 6, 7, 2, 3]
    ])

    b = np.array([
        [2, 1, 3, 7],
        [4, 2, 0, 6],
        [5, 5, 3, 3],
        [1, 3, 3, 7],
        [8, 0, 0, 0],
        [9, 9, 7, 7],
        [1, 2, 3, 4]
    ])

    expected = a @ b
    time1, result1 = calc(a, b, "pij")
    time2, result2 = calc(a, b, "ipj", 2)
    print(time1, result1)
    print(time2, result2)
    print(np.array_equal(expected, result1), np.array_equal(expected, result2))
