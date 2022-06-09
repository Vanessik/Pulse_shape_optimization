

/*******************************************************/
// this function codes the ordinary differential equations for the coupled system
void ensem(double t, double y[], double ydot[]);
double Calc_mfx(double* y);
double Calc_mfy(double* y);
double Calc_mfz(double* y);
double * Make_shift(double *y, int shift);
double * return_to_sync(double *y, double *y_copy);
double * create_sync_copy(double *y);
void release_memory(double *y_copy);
double Calc_state(double* y, int idx);
double *  Make_step2(double *y);
double Calc_std(double *history, int start_idx, int end_idx);
double Show_history(double *history, int idx);
void Phase_state(double phase_, double *y, double *history, double *phase_oscillator, int hist_length);
void Make_step3(double *y, double *history, double *phase_oscillator, int nsteps, int hist_length);
double Calc_angle(double *history, int hist_length);
void Calc_all_angles(double *history, double *phases, int hist_length);
double Show_phase(double *phases, int idx);
// Initialization
double * init_history (int hist_length);
double * init (int nosc_, double epsilon_, double frrms_, int sdim_);
double * init_phase_oscillator(int hist_length);
void Make_biphasic_step(double *y, double *history, double *phase_oscillator, int width_p_, int gap_, int width_n_, double pos_Aampl, double neg_Aampl, int skip_steps, int hist_length);
int Return_end_idx(double *history, int hist_length);
void Calc_oscillator(double *history, double *phase_oscillator, int hist_length);
double Calc_angle2(double *phase_oscillator, int idx);
// Clear
void Clear_phase_oscillator(double *phase_oscillator, int hist_length);
void Clear_history(double *history, int hist_length);
void Clear_state(double *y, int nosc_, double epsilon_, double frrms_);
// Free memory
void free_vector(double *y, long nl, long nh);