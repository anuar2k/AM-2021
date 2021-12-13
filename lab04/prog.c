#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
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
    size_t len;
    row_maj_coord_cell_t values[];
} row_maj_coord_buf_t;

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

row_maj_coord_buf_t *make_row_maj_coord_buf(size_t max_len) {
    row_maj_coord_buf_t *result = malloc(sizeof(*result) + max_len * sizeof(*result->values));
    if (!result) {
        return NULL;
    }

    result->len = max_len;
    return result;
}

row_maj_coord_matrix_t *make_row_maj_coord_matrix(size_t m, size_t n, size_t max_len) {
    row_maj_coord_matrix_t *result = malloc(sizeof(*result) + max_len * sizeof(*result->values));
    if (!result) {
        return NULL;
    }

    result->m = m;
    result->n = n;
    result->len = 0;
    return result;
}

void *memdup(const void *src, size_t size) {
    void *result = malloc(size);
    if (result) {
        memcpy(result, src, size);
    }
    return result;
}

dense_matrix_t *copy_dense_matrix(const dense_matrix_t *input) {
    return memdup(input, sizeof(*input) + input->m * input->n * sizeof(*input->values));
}

dense_matrix_t *row_maj_coord_to_dense(const row_maj_coord_matrix_t *input) {
    dense_matrix_t *result = make_dense_matrix(input->m, input->n);
    if (!result) {
        return NULL;
    }

    for (size_t i = 0; i < input->len; i++) {
        const row_maj_coord_cell_t *cell = &input->values[i];
        CELL(result, cell->row, cell->col) = cell->value;
    }

    return result;
}

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

void write_dense_matrix(const dense_matrix_t *matrix, FILE *file) {
    fprintf(file, "%zux%zu\n", matrix->m, matrix->n);
    for (size_t i = 0; i < matrix->m; i++) {
        fprintf(file, "%f", CELL(matrix, i, 0));
        for (size_t j = 1; j < matrix->n; j++) {
            fprintf(file, ",%f", CELL(matrix, i, j));
        }
        fprintf(file, "\n");
    }
}

int dense_gauss(const dense_matrix_t *input, dense_matrix_t **out_result) {
    dense_matrix_t *result = copy_dense_matrix(input);
    if (!result) {
        return -ENOMEM;
    }

    clock_t measurement_begin = clock();

    size_t row = 0;
    size_t col = 0;

    while (row < result->m && col < result->n) {
        size_t pivot_row = row;

        // find row with earliest nonzero cell
        while (CELL(result, pivot_row, col) == .0f) {
            if (++pivot_row == result->m) {
                pivot_row = row;

                if (++col == result->n) {
                    goto out;
                }
            }
        }

        // do a swap if neccessary
        if (pivot_row != row) {
            for (size_t swap_col = col; swap_col < result->n; swap_col++) {
                float tmp = CELL(result, row, swap_col);
                CELL(result, row, swap_col) = CELL(result, pivot_row, swap_col);
                CELL(result, pivot_row, swap_col) = tmp;
            }
        }

        // subtract pivot row scaled by coeff
        float nonzero = CELL(result, row, col);
        for (size_t clear_row = row + 1; clear_row < result->m; clear_row++) {
            if (CELL(result, clear_row, col) != .0f) {
                float coeff = CELL(result, clear_row, col) / nonzero;

                CELL(result, clear_row, col) = .0f;
                for (size_t clear_col = col + 1; clear_col < result->n; clear_col++) {
                    CELL(result, clear_row, clear_col) -= coeff * CELL(result, row, clear_col);
                }
            }
        }

        row++;
        col++;
    }
out:

    printf("%ld\n", clock() - measurement_begin);

    *out_result = result;
    return 0;
}

int coord_gauss(const dense_matrix_t *input, row_maj_coord_matrix_t **out_result) {
    if (input->m != input->n) {
        return -EINVAL;
    }

    size_t min_row = MAX(input->n - input->m + 1, 1);
    size_t max_row = input->n;
    size_t row_count = max_row - min_row + 1;
    size_t max_nonzero_cell_count = (min_row + max_row) * row_count / 2;

    // result, in a given moment, will contain only rows up to n-th row
    row_maj_coord_matrix_t *result = make_row_maj_coord_matrix(input->m, input->n, max_nonzero_cell_count);
    // every buf designates a rectangle left to eliminate
    row_maj_coord_buf_t *buf_prev = make_row_maj_coord_buf(input->m * input->n);
    row_maj_coord_buf_t *buf_curr = make_row_maj_coord_buf(input->m * input->n);

    size_t *iter_order = malloc(input->m * input->n * sizeof(*iter_order));

    if (!result || !buf_prev || !buf_curr || !iter_order) {
        free(result);
        free(buf_prev);
        free(buf_curr);
        free(iter_order);
        return -ENOMEM;
    }

    result->m = input->m;
    result->n = input->n;
    buf_prev->len = 0;
    buf_curr->len = 0;

    clock_t measurement_begin = clock();

    // initialize "previous iteration" buffer with input
    for (size_t i = 0; i < input->m; i++) {
        for (size_t j = 0; j < input->n; j++) {
            if (CELL(input, i, j) != .0f) {
                buf_prev->values[buf_prev->len++] = (row_maj_coord_cell_t) {
                    .row = i, 
                    .col = j, 
                    .value = CELL(input, i, j)
                };
            }
        }
    }

    size_t row = 0;
    size_t col = 0;

    while (row < result->m && col < result->n) {
        // previous iteration left no nonzeros -> we're done
        if (buf_prev->len == 0) {
            break;
        }

        // search for row with earlies nonzero cell
        size_t pivot_row_start = 0;
        size_t pivot_row = buf_prev->values[0].row;
        size_t max_pivot_col = buf_prev->values[0].col;

        if (max_pivot_col != col) {
            for (size_t i = 0; i < buf_prev->len; i++) {
                if (buf_prev->values[i].col < max_pivot_col) {
                    pivot_row_start = i;
                    pivot_row = buf_prev->values[i].row;
                    max_pivot_col = buf_prev->values[i].col;

                    if (max_pivot_col == col) {
                        break;
                    }
                }
            }
        }
        col = max_pivot_col;

        // write pivot to result matrix
        size_t result_pivot_start = result->len;
        size_t pivot_cell_idx = pivot_row_start;
        while (pivot_cell_idx < buf_prev->len && buf_prev->values[pivot_cell_idx].row == pivot_row) {
            result->values[result->len] = buf_prev->values[pivot_cell_idx++];
            result->values[result->len++].row = row;
        }

        // program iteration order
        size_t idx = 0;
        size_t curr = 0;
        size_t first_row = 0;
        while (curr < buf_prev->len && buf_prev->values[curr].row == row) {
            buf_prev->values[curr].row = pivot_row;
            first_row++;
            curr++;
        }
        while (curr < buf_prev->len) {
            if (buf_prev->values[curr].row == pivot_row) {
                for (size_t i = 0; i < first_row; i++) {
                    iter_order[idx++] = i;
                }
                while (curr < buf_prev->len && buf_prev->values[curr].row == pivot_row) {
                    curr++;
                }
            }
            else {
                iter_order[idx++] = curr++;
            }
        }

        printf("new\n");
        for (size_t i = 0; i < idx; i++) {
            printf("ord: %zu\n", iter_order[i]);
        }

        buf_curr->len = 0;
        long long prev_row = -1;
        size_t result_pivot_idx;
        float coeff;

        // the secret sauce - pivot and "previous iteration" buffer merging
        for (size_t i = 0; i < idx; i++) {
            row_maj_coord_cell_t *buf_cell = &buf_prev->values[iter_order[i]];
            if (prev_row != buf_cell->row) {
                if (buf_cell->col == col) {
                    result_pivot_idx = result_pivot_start + 1;
                    prev_row = buf_cell->row;
                    coeff = buf_cell->value / result->values[result_pivot_start].value;
                } else {
                    buf_curr->values[buf_curr->len++] = *buf_cell;
                }
            } else {
                while (result_pivot_idx < result->len && result->values[result_pivot_idx].col < buf_cell->col) {
                    row_maj_coord_cell_t *pivot_cell = &result->values[result_pivot_idx];
                    buf_curr->values[buf_curr->len++] = (row_maj_coord_cell_t) {
                        .row = buf_cell->row, 
                        .col = pivot_cell->col, 
                        .value = .0f - (pivot_cell->value * coeff)
                    };
                    result_pivot_idx++;
                }
                if (result_pivot_idx < result->len && result->values[result_pivot_idx].col == buf_cell->col) {
                    row_maj_coord_cell_t *pivot_cell = &result->values[result_pivot_idx++];
                    float result = buf_cell->value - pivot_cell->value * coeff;
                    if (result != .0f) {
                        buf_curr->values[buf_curr->len++] = (row_maj_coord_cell_t) {
                            .row = buf_cell->row, 
                            .col = buf_cell->col, 
                            .value = result
                        };
                    }
                } else {
                    buf_curr->values[buf_curr->len++] = *buf_cell;
                }
                if (i + 1 == idx || buf_prev->values[iter_order[i+1]].row != buf_cell->row) {
                    while (result_pivot_idx < result->len) {
                        row_maj_coord_cell_t *pivot_cell = &result->values[result_pivot_idx];
                        buf_curr->values[buf_curr->len++] = (row_maj_coord_cell_t) {
                            .row = buf_cell->row, 
                            .col = pivot_cell->col, 
                            .value = .0f - (pivot_cell->value * coeff)
                        };
                        result_pivot_idx++;
                    }
                }
                
            }
        }

        row++;
        col++;

        void *tmp = buf_prev;
        buf_prev = buf_curr;
        buf_curr = tmp;
    }

    printf("%ld\n", clock() - measurement_begin);

    *out_result = result;
    free(buf_prev);
    free(buf_curr);
    free(iter_order);

    return 0;
}

// ./prog.out input output     type
// ./prog.out m.csv /dev/null  dense
// ./prog.out m.csv out.csv    coord
int main(int argc, char **argv) {
    dense_matrix_t *input = NULL;
    dense_matrix_t *output_dense = NULL;
    row_maj_coord_matrix_t *output_coord = NULL;
    FILE *out = NULL;
    int ret = 1;

    if (argc != 4) {
        goto cleanup;
    }
    const char *input_path = argv[1];
    const char *output_path = argv[2];
    const char *type = argv[3];

    if (read_dense_matrix(input_path, &input)) {
        goto cleanup;
    }

    out = fopen(output_path, "w");
    if (!out) {
        goto cleanup;
    }

    printf("%ld\n", CLOCKS_PER_SEC);

    if (!strcmp(type, "dense")) {
        if (dense_gauss(input, &output_dense)) {
            goto cleanup;
        }
    } else if (!strcmp(type, "coord")) {
        if (coord_gauss(input, &output_coord)) {
            goto cleanup;
        }

        output_dense = row_maj_coord_to_dense(output_coord);
        if (!output_dense) {
            goto cleanup;
        }
    } else {
        goto cleanup;
    }

    write_dense_matrix(output_dense, out);
    ret = 0;

cleanup:
    free(input);
    free(output_dense);
    free(output_coord);
    if (out) {
        fclose(out);
    }
    return ret;
}
