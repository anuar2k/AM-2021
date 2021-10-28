#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <errno.h>

#define CELL(mat, row, col) ((mat)->data[(row) * (mat)->n + (col)])

typedef struct matrix_struct {
    size_t m;
    size_t n;
    float data[];
} matrix_t;

matrix_t *make_matrix(size_t m, size_t n) {
    matrix_t *result = calloc(sizeof(*result) + m * n * sizeof(*result->data), 1);
    if (!result) {
        return NULL;
    }
    result->m = m;
    result->n = n;
    return result;
}

void print_matrix(matrix_t *matrix) {
    for (size_t i = 0; i < matrix->m; i++) {
        for (size_t j = 0; j < matrix->n; j++) {
            printf("%.1f\t", CELL(matrix, i, j));
        }
        printf("\n");
    }
}

int matmul(matrix_t *a, matrix_t *b, matrix_t **out_result) {
    if (a->n != b->m) {
        return -EINVAL;
    }

    size_t m = a->m;
    size_t n = b->n;
    size_t k = a->n;

    matrix_t *result = make_matrix(m, n);
    if (!result) {
        return -ENOMEM;
    }

    result->m = m;
    result->n = n;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t p = 0; p < k; p++) {
                CELL(result, i, j) += CELL(a, i, p) * CELL(b, p, j);
            }
        }
    }

    *out_result = result;
    return 0;
}

int main(int argc, char **argv) {
    matrix_t *a = make_matrix(2, 4);
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 4; j++) {
            CELL(a, i, j) = i + j;
        }
    }
    matrix_t *b = make_matrix(4, 3);
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 3; j++) {
            CELL(b, i, j) = i + j;
        }
    }
    print_matrix(a);
    printf("\n");
    print_matrix(b);
    printf("\n");
    matrix_t *c;
    matmul(a, b, &c);
    print_matrix(c);
    free(a);
    free(b);
    free(c);
}
