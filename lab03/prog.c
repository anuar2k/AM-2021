#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CELL(mat, row, col) ((mat)->values[(row) * (mat)->n + (col)])
#define CSV_DELIM ","

typedef struct {
    size_t m;
    size_t n;
    float values[];
} dense_matrix_t;

typedef struct {
    size_t row;
    size_t col;
    float value;
} row_maj_coord_cell_t;

typedef struct {
    size_t m;
    size_t n;
    size_t len;
    row_maj_coord_cell_t values[];
} row_maj_coord_matrix_t;

dense_matrix_t *make_dense_matrix(size_t m, size_t n) {
    dense_matrix_t *result = calloc(sizeof(*result) + m * n * sizeof(*result->values), 1);
    if (!result) {
        return NULL;
    }
    result->m = m;
    result->n = n;
    return result;
}

row_maj_coord_matrix_t *dense_to_row_maj_coord(dense_matrix_t *input) {
    size_t nonzeros = 0;
    for (size_t i = 0; i < input->m * input->n; i++) {
        if (input->values[i] != .0) {
            nonzeros++;
        }
    }

    row_maj_coord_matrix_t *result = calloc(sizeof(*result) + nonzeros * sizeof(*result->values), 1);
    if (!result) {
        return NULL;
    }
    result->m = input->m;
    result->n = input->n;
    result->len = nonzeros;

    size_t next = 0;
    for (size_t i = 0; i < input->m; i++) {
        for (size_t j = 0; j < input->n; j++) {
            if (CELL(input, i, j) != .0) {
                result->values[next++] = (row_maj_coord_cell_t){
                    .row = i,
                    .col = j,
                    .value = CELL(input, i, j)};
            }
        }
    }

    return result;
};

int read_dense_matrix(const char *path, dense_matrix_t **out_result) {
    FILE *file = NULL;
    dense_matrix_t *result = NULL;
    int ret = -EINVAL;

    file = fopen(path, "r");
    if (!file) {
        goto cleanup;
    }

    size_t m;
    size_t n;
    if (fscanf(file, " %zu x %zu ", &m, &n) != 2) {
        goto cleanup;
    }

    result = make_dense_matrix(m, n);
    if (!result) {
        goto cleanup;
    }

    for (size_t i = 0; i < m; i++) {
        if (fscanf(file, " %f ", &CELL(result, i, 0)) != 1) {
            goto cleanup;
        }
        for (size_t j = 1; j < n; j++) {
            if (fscanf(file, " " CSV_DELIM "%f ", &CELL(result, i, j)) != 1) {
                goto cleanup;
            }
        }
    }

    *out_result = result;
    ret = 0;

cleanup:
    if (file) {
        fclose(file);
    }
    if (ret) {
        free(result);
    }
    return ret;
}

void write_dense_matrix(dense_matrix_t *matrix, FILE *file) {
    fprintf(file, "%zux%zu\n", matrix->m, matrix->n);
    for (size_t i = 0; i < matrix->m; i++) {
        fprintf(file, "%f", CELL(matrix, i, 0));
        for (size_t j = 1; j < matrix->n; j++) {
            fprintf(file, ",%f", CELL(matrix, i, j));
        }
        fprintf(file, "\n");
    }
}

int dense_dense_matmul(dense_matrix_t *a, dense_matrix_t *b, dense_matrix_t **out_result) {
    if (a->n != b->m) {
        return -EINVAL;
    }

    size_t m = a->m;
    size_t n = b->n;
    size_t k = a->n;

    dense_matrix_t *result = make_dense_matrix(m, n);
    if (!result) {
        return -ENOMEM;
    }

    result->m = m;
    result->n = n;

    clock_t measurement_begin = clock();

    for (size_t i = 0; i < m; i++) {
        for (size_t p = 0; p < k; p++) {
            for (size_t j = 0; j < n; j++) {
                CELL(result, i, j) += CELL(a, i, p) * CELL(b, p, j);
            }
        }
    }

    printf("0\n");
    printf("%ld\n", clock() - measurement_begin);

    *out_result = result;
    return 0;
}

int coord_dense_matmul(dense_matrix_t *a, dense_matrix_t *b, dense_matrix_t **out_result) {
    int ret;
    if (a->n != b->m) {
        return -EINVAL;
    }

    size_t m = a->m;
    size_t n = b->n;
    size_t k = a->n;

    dense_matrix_t *result = make_dense_matrix(m, n);

    clock_t measurement_begin = clock();
    row_maj_coord_matrix_t *a_sparse = dense_to_row_maj_coord(a);
    printf("%ld\n", clock() - measurement_begin);

    if (!result || !a_sparse) {
        free(result);
        free(a_sparse);
        return -ENOMEM;
    }

    result->m = m;
    result->n = n;

    measurement_begin = clock();

    for (size_t i_cell = 0; i_cell < a_sparse->len; i_cell++) {
        for (size_t j = 0; j < n; j++) {
            row_maj_coord_cell_t *cell = &a_sparse->values[i_cell];
            size_t i = cell->row;
            size_t p = cell->col;
            CELL(result, i, j) += cell->value * CELL(b, p, j);
        }
    }

    printf("%ld\n", clock() - measurement_begin);

    free(a_sparse);

    *out_result = result;
    return 0;
}

// ./prog.out a_matrix b_matrix out_matrix type
// ./prog.out m1.csv   m2.csv   /dev/null  dense
// ./prog.out m1.csv   m2.csv   m3.csv     coord
int main(int argc, char **argv) {
    dense_matrix_t *a = NULL;
    dense_matrix_t *b = NULL;
    dense_matrix_t *c = NULL;
    FILE *out = NULL;
    int ret = 1;

    if (argc != 5) {
        goto cleanup;
    }
    const char *a_path = argv[1];
    const char *b_path = argv[2];
    const char *c_path = argv[3];
    const char *type = argv[4];

    if (read_dense_matrix(a_path, &a)) {
        goto cleanup;
    }
    if (read_dense_matrix(b_path, &b)) {
        goto cleanup;
    }

    out = fopen(c_path, "w");
    if (!out) {
        goto cleanup;
    }

    printf("%ld\n", CLOCKS_PER_SEC);

    if (!strcmp(type, "dense")) {
        if (dense_dense_matmul(a, b, &c)) {
            goto cleanup;
        }
    } else if (!strcmp(type, "coord")) {
        if (coord_dense_matmul(a, b, &c)) {
            goto cleanup;
        }
    } else {
        goto cleanup;
    }

    write_dense_matrix(c, out);
    ret = 0;

cleanup:
    free(a);
    free(b);
    free(c);
    if (out) {
        fclose(out);
    }
    return ret;
}
