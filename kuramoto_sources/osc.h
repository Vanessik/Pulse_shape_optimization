using namespace std;

/*******************************************************/
// this function codes the ordinary differential equations for the coupled system
double Calc_state(double* phi, int idx);
double Zm(double phi);
double Zp(double phi);
void MeanField(double *phi, double *y);
double Calc_R(double* y);
double Calc_Theta(double* y);
void bfpulse_EulerStep(double *phi, double *y, int np); 
void copy_vecs(double *phi, double *phi_copy);
// Clear memory
void free_vector(double *l, long nl, long nh);
// step
void Make_step(double *phi, double *y, int width_p_, int gap_, int width_n_, double pos_Aampl, double neg_Aampl, int skip_steps);
// Initialization
double * init (int nptrans);
double * init_MF ();
double * init_state_copy (double *phi);
