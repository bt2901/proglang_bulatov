#include <vector>
#include <cmath>

using std::vector;

vector<vector<vector<double> > > calc_lpx(vector<vector<vector<double> > >& X, vector<vector<double> >& F,
    vector<vector<double> >& B, double s);

vector<vector<size_t> > e_step(vector<vector<vector<double> > >& X, vector<vector<double> >& F,
    vector<vector<double> >& B, double s, vector<vector<double> >& A);

void m_step(vector<vector<vector<double> > >& X, vector<vector<size_t> >& q, size_t h, size_t w,
    vector<vector<double> >* F, vector<vector<double> >* B, double* s, vector<vector<double> >* A);

void run_EM(vector<vector<vector<double> > >& X, size_t h, size_t w,
    vector<vector<double> >* F, vector<vector<double> >* B, double* s,
    vector<vector<double> >* A, size_t number_of_iterations);

