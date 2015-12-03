#include <vector>
#include <cmath>

#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include <iostream>

#define NO_IMPORT_ARRAY

#include "EM.h"


using std::vector;
using std::max;

bool face_area(int d_h, int d_w, int i, int j, int h, int w) {
    bool cond1 = d_h <= i && i <= d_h + h - 1;
    bool cond2 = d_w <= j && j <= d_w + w - 1;
    return (cond1 && cond2);
}

vector<vector<vector<double> > > get_lpx_d_all(vector<vector<vector<double> > >& X, vector<vector<double> >& F,
    vector<vector<double> >& B, double s) {
    size_t H = X.size();
    size_t W = X[0].size();
    size_t N = X[0][0].size();
    size_t h = F.size();
    size_t w = F[0].size();

    double base = -H * W * log(sqrt(2 * M_PI) * s);
    vector<vector<vector<double> > > result(H - h + 1, vector<vector<double> >(W - w + 1, vector<double>(N, base)));
    for (int k = 0; k < N; ++k) {
        // std::cout << "k == " << k << std::endl;
        for (int d_h = 0; d_h < H - h + 1; ++d_h) {
            for (int d_w = 0; d_w < W - w + 1; ++d_w) {
                for (int i = 0; i < H; ++i) {
                    for (int j = 0; j < W; ++j) {
                        double M;
                        if (face_area(d_h, d_w, i, j, h, w)) {
                            M = F[i - d_h][j - d_w];
                        } else {
                            M = B[i][j];
                        };
                        // std::cout << "M == " << M << std::endl;
                        // std::cout << d_h << d_w << "; " << i << j << std::endl;
                        result[d_h][d_w][k] += (X[i][j][k] - M) * (X[i][j][k] - M) / (2 * s * s);
                    }
                }
            }
        }
    }
    return result;
}

vector<vector<size_t> > e_step(vector<vector<vector<double> > >& X, vector<vector<double> >& F,
    vector<vector<double> >& B, double s, vector<vector<double> >& A) {

    // H, W, N = np.shape(X)
    size_t H = X.size();
    size_t W = X[0].size();
    size_t N = X[0][0].size();
    size_t h = F.size();
    size_t w = F[0].size();
    vector<vector<vector<double> > > q = get_lpx_d_all(X, F, B, s);
    vector<vector<size_t> > best_d(2, vector<size_t>(N));

    vector<vector<size_t> > counts(H, vector<size_t>(W));
    double expected = 1.0 *  H * W / N;
    for (int k = 0; k < N; ++k) {
        int best_i = -1;
        int best_j = -1;
        for (int i = 0; i < H - h + 1; ++i) {
            for (int j = 0; j < W - w + 1; ++j) {
                q[i][j][k] += log(max(1e-300, A[i][j]));
                if (counts[i][j] <= 4 * expected) {
                    if ((best_i == -1) || (q[i][j][k] >= q[best_i][best_j][k])) {
                        best_i = i;
                        best_j = j;
                    }
                }
            }
        }

        best_d[0][k] = best_i;
        best_d[1][k] = best_j;
        ++counts[best_i][best_j];
        // std::cout << best_i << ", " << best_j  << " counts at:" << counts[best_i][best_j] << std::endl;

    }
    return best_d;
}

void m_step(vector<vector<vector<double> > >& X, vector<vector<size_t> >& q, size_t h, size_t w,
    vector<vector<double> >* F, vector<vector<double> >* B, double* s, vector<vector<double> >* A) {

    size_t H = X.size();
    size_t W = X[0].size();
    size_t N = X[0][0].size();

    vector<vector<size_t> > normB(H, vector<size_t>(W, N));

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            for (size_t k = 0; k < N; ++k) {
                (*B)[i][j] += X[i][j][k];
            }
        }
    }
    for (size_t k = 0; k < N; ++k) {
        size_t dh = q[0][k];
        size_t dw = q[1][k];
        (*A)[dh][dw] += 1.0 / N;
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                (*F)[i][j] += 1.0 * X[dh + i][dw + j][k] / N;
                (*B)[dh + i][dw + j] -= X[dh + i][dw + j][k];
                normB[dh + i][dw + j] -= 1;
            }
        }
    }
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            if (normB[i][j] == 0) {
                (*B)[i][j] = 0;
            } else {
                (*B)[i][j] /= normB[i][j];
            }
        }
    }
    double s2 = 0;
    for (size_t k = 0; k < N; ++k) {
        size_t dh = q[0][k];
        size_t dw = q[1][k];
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                double M;
                if (face_area(dh, dw, i, j, h, w)) {
                    M = (*F)[i - dh][j - dw];
                } else {
                    M = (*B)[i][j];
                };
                s2 += (X[i][j][k] - M) * (X[i][j][k] - M);
            }
        }
    }
    (*s) = sqrt(s2 / (H * W * N));
    return;
}
void run_EM(vector<vector<vector<double> > >& X, size_t h, size_t w,
    vector<vector<double> >* F, vector<vector<double> >* B, double* s,
    vector<vector<double> >* A, size_t number_of_iterations) {

    vector<vector<size_t> > q = e_step(X, *F, *B, *s, *A);

    for (int i = 0; i < number_of_iterations; ++i) {
        std::cout << "iter " << i << std::endl;
        // F, B, s, A = m_step(X, q, h, w)
        m_step(X, q, h, w, F, B, s, A);

        vector<vector<size_t> > q = e_step(X, *F, *B, *s, *A);

    }
}
