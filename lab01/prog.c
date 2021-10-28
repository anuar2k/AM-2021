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

#define matmul(lp0, lp1, lp2) \
int matmul_##lp0##lp1##lp2(matrix_t *a, matrix_t *b, matrix_t **out_result) { \
    if (a->n != b->m) { \
        return -EINVAL; \
    } \
 \
    size_t m = a->m; \
    size_t n = b->n; \
    size_t k = a->n; \
 \
    matrix_t *result = make_matrix(m, n); \
    if (!result) { \
        return -ENOMEM; \
    } \
 \
    result->m = m; \
    result->n = n; \
 \
    size_t i_end = m; \
    size_t j_end = n; \
    size_t p_end = k; \
 \
    for (size_t lp0 = 0; lp0 < lp0##_end; lp0++) { \
        for (size_t lp1 = 0; lp1 < lp1##_end; lp1++) { \
            for (size_t lp2 = 0; lp2 < lp2##_end; lp2++) { \
                CELL(result, i, j) += CELL(a, i, p) * CELL(b, p, j); \
            } \
        } \
    } \
 \
    *out_result = result; \
    return 0; \
}

matmul(i, j, p)
matmul(i, p, j)
matmul(j, i, p)
matmul(j, p, i)
matmul(p, i, j)
matmul(p, j, i)

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
    {
        matrix_t *c;
        matmul_ijp(a, b, &c);
        print_matrix(c);
        printf("\n");
        free(c);
    }

    {
        matrix_t *c;
        matmul_ipj(a, b, &c);
        print_matrix(c);
        printf("\n");
        free(c);
    }

    {
        matrix_t *c;
        matmul_jip(a, b, &c);
        print_matrix(c);
        printf("\n");
        free(c);
    }

    {
        matrix_t *c;
        matmul_jpi(a, b, &c);
        print_matrix(c);
        printf("\n");
        free(c);
    }

    {
        matrix_t *c;
        matmul_pij(a, b, &c);
        print_matrix(c);
        printf("\n");
        free(c);
    }

    {
        matrix_t *c;
        matmul_pji(a, b, &c);
        print_matrix(c);
        printf("\n");
        free(c);
    }

    free(a);
    free(b);
}
