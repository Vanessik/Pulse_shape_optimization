#include <iostream> 
#include <fstream> 
#include <math.h>
#include <cmath> 
#include "my_defs.h"
#include "nrlib.h"
using namespace std;

/*************  important parameters  **************/
int nosc = 5000;                  // number of oscillators
int sdim = 3;                     // system dimension
int neqn = sdim * nosc;           // number of equations
double epsilon = 0.2;             // coupling parameter
double frrms = 0.02;              // width of the distribution of natural frequencies 
const int ntrans = 1000;         // length of the transient and of the main integration
const double tstep = 0.1;         // integration step
int width_p = 25, gap = 0, kfactor = 0;   
int width_n = width_p * kfactor;
int nsteps = 10;                   // debug value
int hist_length = 30000;                   // length of all history values

double mfx, mfy, mfz, t;
double I_extern = 0.;              // extern amplitude
double *omeg;
int mu = 500;
int period = 1500;
double om0_estimate = pi2 / period;
double alpha = 0.4 * om0_estimate;
double phase;
double threshold = 0.01;
double dt = 1.;

void ensem(double t, double y[], double ydot[])
{
  int i;
  double x,mf;
  
  for (i=1, mf=0; i<=neqn; i+=sdim)  mf+=y[i];  mf=mf/nosc;   // Mean field in x 
  for (i=1; i<=nosc; i++)
    {
      x=y[sdim*i-2];
      ydot[sdim*i-2]=y[sdim*i-1]-x*x*x+3.*x*x-y[sdim*i]+omeg[i]
	+epsilon*(mf) + I_extern;
      ydot[sdim*i-1]=1.-5.*x*x-y[sdim*i-1];
      ydot[sdim*i]=0.006*(4.*(x+1.56)-y[sdim*i]);
    }
}   
   

//Calculate mean_field x
 double Calc_mfx(double* y)
{  
  int k;
  for (k=1, mfx=0; k<=neqn; k+=sdim)  mfx+=y[k];  
  return mfx/nosc;
}

//Calculate mean_field y
 double Calc_mfy(double* y)
{  
  int k;
  for (k=2, mfy=0; k<=neqn; k+=sdim)  mfy+=y[k];  
  return mfy/nosc;
}

 double Calc_mfz(double* y)
{  
  int k;
  for (k=3, mfz=0; k<=neqn; k+=sdim)  mfz+=y[k];  
  return mfz/nosc;
}

double * Make_shift(double *y, int shift)
{
  rk(y,neqn,t,tstep,ensem,shift); 
  return y;
}

double * return_to_sync(double *y, double *y_copy)
{
  for(int k=1;k<=neqn;k++) y[k] = y_copy[k];
  return y;
}

double * create_sync_copy(double *y)
{
  double *y_copy;
  y_copy=dvector(1,neqn);
  for(int k=1;k<=neqn;k++) y_copy[k] = y[k];  // store the state before the pulse   
  return y_copy;
}


void release_memory(double *y_copy)
{
  delete y_copy;    
}

double *  Make_step2(double *y)
{
  
  int i, k;
  rk(y,neqn,t,tstep,ensem,1); 
  return y;
}

int Return_end_idx(double *history, int hist_length)
{
  for(int k=1; k <= hist_length; k++)
  {
  if (history[k] == 0) return k;    
  }
  return hist_length;
}

void Clear_history(double *history, int hist_length)
{
  for (int i = 1; i <= hist_length; i++) history[i] = 0;
}

void Clear_phase_oscillator(double *phase_oscillator, int hist_length)
{
  for (int i=0; i<=sdim*hist_length; i++) phase_oscillator[i] = 0.;
}

void Clear_state(double *y, int nosc_, double epsilon_, double frrms_)
{
  long iseed=-73, iseed1=-21;
  nosc = nosc_;           
  neqn = 3*nosc;     
  epsilon = epsilon_;   
  frrms = frrms_; 
  int i, k; // initialization of the random number generator
  double I_extern = 0.;
  
  for (i=1; i<=nosc; i++)
    omeg[i]=3+frrms*gasdev(&iseed);   
  for (i=1; i<=neqn; i++) 
      y[i]=2*ran1(&iseed1)-1.;          // Initial conditions for doff equations: just random 
	
  t=0.;  rk(y,neqn,t,tstep,ensem,ntrans);     // transient: this is time before our system comes to its limit cycle,
  // i.e. all oscillators come to their limit cyces and synchronize
  // Function rk is the Runge-Kutta integrator  
}

double Calc_angle(double *history, int hist_length)
{    
    double u = 0., x = 0., d = 0.;
    double phase;
    int last_idx;
    last_idx = Return_end_idx(history, hist_length);
    for (int i = 1; i <= last_idx; i++)
    {
    u += x * dt;
    x += (-alpha * x - om0_estimate * om0_estimate * u + history[i]) * dt;
    d += (x - d) / mu * dt;
    // auxiliary variable
    double xhat = om0_estimate * mu * d; 
    // phase (-pi, pi)
    phase = atan2(xhat, x);  
    }
    return phase; 
}

void Calc_all_angles(double *history, double *phases, int hist_length)
{    
    hist_length= hist_length;
    double u = 0., x = 0., d = 0.;
    int last_idx;
    last_idx = Return_end_idx(history, hist_length);
    for (int i = 1; i <= last_idx; i++)
    {
    u += x * dt;
    x += (-alpha * x - om0_estimate * om0_estimate * u + history[i]) * dt;
    d += (x - d) / mu * dt;
    // auxiliary variable
    double xhat = om0_estimate * mu * d; 
    // phase (-pi, pi)
    phases[i] = atan2(xhat, x);  
    }
}


void Calc_oscillator(double *history, double *phase_oscillator, int hist_length)
{    
    int last_idx;
    last_idx = Return_end_idx(history, hist_length);
    phase_oscillator[(last_idx-1)*sdim-2] = phase_oscillator[(last_idx-2)*sdim-2] +  phase_oscillator[(last_idx-2)*sdim-1]*dt;
    phase_oscillator[(last_idx-1)*sdim-1] = phase_oscillator[(last_idx-2)*sdim-1] + (-alpha * phase_oscillator[(last_idx-2)*sdim-1] - om0_estimate * om0_estimate * phase_oscillator[(last_idx-1)*sdim-2] + history[last_idx-1]) * dt;
    phase_oscillator[(last_idx-1)*sdim] = phase_oscillator[(last_idx-2)*sdim] + (phase_oscillator[(last_idx-1)*sdim-1] - phase_oscillator[(last_idx-2)*sdim]) / mu * dt; 
}


double Calc_angle2(double *phase_oscillator, int idx)
{    
    double xhat = om0_estimate * mu * phase_oscillator[idx*sdim]; 
    phase = atan2(xhat,  phase_oscillator[idx*sdim-1]);  
    return phase; 
}

double Show_phase(double *phases, int idx)
{  
  return phases[idx];
}

void  Make_step3(double *y, double *history, double *phase_oscillator, int nsteps, int hist_length)
{ 
  int last_idx;
  last_idx = Return_end_idx(history, hist_length);
  for (int i = 1; i <= nsteps; i++)
  {
  rk(y, neqn, t, tstep, ensem, 1); 
  history[i+last_idx-1] = Calc_mfx(y);  
  Calc_oscillator(history, phase_oscillator, hist_length);
  }
}

void Make_biphasic_step(double *y, double *history, double *phase_oscillator, int width_p_, int gap_, int width_n_, double pos_Aampl, double neg_Aampl, int skip_steps, int hist_length)
{    
  width_p = width_p_; gap = gap_; width_n = width_n_; 
  if (width_p > 0) {
      I_extern = pos_Aampl;
      Make_step3(y, history, phase_oscillator,  width_p, hist_length); 
  }
  if (gap > 0){
      I_extern = 0.;
      Make_step3(y, history, phase_oscillator, gap, hist_length);
  }
  if (width_n > 0){
      I_extern = neg_Aampl;
      Make_step3(y, history, phase_oscillator, width_n, hist_length); 
  }
  if (skip_steps > 0){
      I_extern = 0.;
      Make_step3(y, history, phase_oscillator,  skip_steps, hist_length); 
  }
  I_extern = 0.;
}

void * Phase_state(double phase_, double *y, double *history, double *phase_oscillator, int hist_length)
{
  int last_idx;
  last_idx = Return_end_idx(history, hist_length)-1;
  phase = Calc_angle2(phase_oscillator, last_idx); 
  double val = fmod(phase - phase_, pi2);
  while (ABS(val) > threshold)
  {
  Make_step3(y, history, phase_oscillator, 1, hist_length);
  phase = Calc_angle2(phase_oscillator, last_idx++);
  val = fmod(phase - phase_, pi2);
  }
}

double Calc_state(double* y, int idx)
{  
  return y[idx];
}

double Show_history(double *history, int idx)
{  
  return history[idx];
}

double Show_oscillator(double *phase_oscillator, int idx)
{  
  return phase_oscillator[idx];
}

void free_vector(double *y, long nl, long nh)
{
  free_dvector(y, nl, nh);
}

double Calc_std(double *history, int start_idx, int end_idx)
{
  double var = 0., mean = 0.;
  for(int n = start_idx; n <= end_idx; n++ ) mean += history[n];  mean = mean / (end_idx - start_idx + 1);    
  for(int n = start_idx; n <= end_idx; n++ )
  { 
  var += (history[n] - mean) * (history[n] - mean);
  }
  var /= (end_idx - start_idx + 1);
  return sqrt(var);
}


//Init params
double * init (int nosc_, double epsilon_, double frrms_, int sdim_) 
{
  long iseed=-73, iseed1=-21;
  sdim = sdim_;
  nosc = nosc_;           
  neqn = sdim*nosc;     
  epsilon = epsilon_;   
  frrms = frrms_; 
  double *y;
  int i, k; // initializati10000on of the random number generator
  double I_extern = 0.;

  y = dvector(1,neqn);
  omeg = dvector(1,nosc);// reserve space for arrays
  // here I use routines from Numerical Receipes to have arrays with the index
  // running from 1 to neqn or from 1 to nosc 
  
  for (i=1; i<=nosc; i++)
    omeg[i]=3+frrms*gasdev(&iseed);   
  for (i=1; i<=neqn; i++) 
      y[i]=2*ran1(&iseed1)-1.;          // Initial conditions for doff equations: just random 
	
  t=0.;  rk(y,neqn,t,tstep,ensem,ntrans);     // transient: this is time before our system comes to its limit cycle,
  // i.e. all oscillators come to their limit cyces and synchronize
  // Function rk is the Runge-Kutta integrator
  return y;
}

double * init_history(int hist_length)
{
  double *history;
  history = dvector(1, hist_length); 
  for (int i=1; i<=hist_length; i++) history[i] = 0.;
  return history;
}

// double * init_phases(int hist_length)
// {
//   double *phases;
//   phases = dvector(1, hist_length); 
//   for (int i=1; i<=hist_length; i++) phases[i] = 0.;
//   return phases;
// }

double * init_phase_oscillator(int hist_length)
{
  double *phase_oscillator;
  phase_oscillator = dvector(0, sdim*hist_length); 
  for (int i=0; i<=sdim*hist_length; i++) phase_oscillator[i] = 0.;
  return phase_oscillator;
}
