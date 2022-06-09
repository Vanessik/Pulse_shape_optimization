/*******  amplitude response curve for the SL oscillators   ******/
/*******            charge-balanced pulses                  ******/

#include <iostream> 
#include <fstream> 
#include <math.h>
#include "my_defs.h"
#include "nrlib.h"
using namespace std;

/*************  important parameters  **************/
const double omegm=9.9, omegp=10.1;        // frequencies omega minus and omega plus
const int noscm=4000, noscp=noscm*1.5;  // number of oscillators (40% of slow units)
const int neqn = noscm+noscp;          // number of equations
const double epsil=0.4;                  // coupling strength   
int nptrans=500000;               // number of points for transient
int width_p=25, gap=0, kfactor=0;   
int width_n=width_p * kfactor; // width of the 2nd pulse in steps
const double Aampl=30; 
double I_extern=0; 
const double D=0.1;             // noise strength
const double tstep=0.0001, sqstep=sqrt(tstep), noise_factor=sqstep*sqrt(2*D);
long int iseed=-21; 

/******************************************************/
double Zm(double phi)
{
  return (1-cos(phi))*exp(3*(cos(phi-pi/3)-1));
}
/******************************************************/

double Zp(double phi)
{
  return cos(phi)*exp(3*(cos(phi-1.4*pi)-1));
}
/******************************************************/

void MeanField(double *phi, double *y)
{
  double avcos, avsin;
  avcos = avsin = 0;
  for(int i=1; i<=neqn; i++)
    {
      avcos += cos(phi[i]);  avsin += sin(phi[i]); 
    }
  avcos /= neqn; avsin /= neqn;
  y[1] = sqrt(SQR(avcos)+SQR(avsin)); y[2] = atan2(avsin,avcos);
  if (y[2] < 0) y[2] += pi2; 
}

/******************************************************/
double Calc_R(double* y)
{  
  return y[1];
}

/******************************************************/
double Calc_Theta(double* y)
{  
  return y[2];
}

/******************************************************/
double Calc_state(double* phi, int idx)
{  
  return phi[idx];
}

// //******************************************************/
void bfpulse_EulerStep(double *phi, double *y, int np) 
{
  for(int k=1;k<=np;k++)              // cycle over the number of Euler steps 
    {
      MeanField(phi, y);
      for(int i=1;i<=noscm;i++)
	phi[i] = phi[i] + (omegm+epsil*y[1]*sin(y[2]-phi[i])+Zm(phi[i])*I_extern)*tstep
	  +noise_factor*gasdev(&iseed);
      for(int i=noscm+1;i<=neqn;i++)
	phi[i] = phi[i] + (omegp+epsil*y[1]*sin(y[2]-phi[i])+Zp(phi[i])*I_extern)*tstep
	  +noise_factor*gasdev(&iseed);
    }
}

void copy_vecs(double *phi, double *phi_copy)
{
  for(int k=1; k<=neqn; k++) phi[k] = phi_copy[k];
}

void free_vector(double *l, long nl, long nh)
{
  free_dvector(l, nl, nh);
}

void Make_step(double *phi, double *y, int width_p_, int gap_, int width_n_, double pos_Aampl, double neg_Aampl, int skip_steps)
{
  width_p = width_p_; gap = gap_; width_n = width_n_; 
    
  if (width_p > 0){
      I_extern = pos_Aampl;
      bfpulse_EulerStep(phi, y, width_p);  
  }
  if (gap > 0){
      I_extern = 0.;
      bfpulse_EulerStep(phi, y, gap);
  }
  if (width_n > 0){
      I_extern = neg_Aampl;
      bfpulse_EulerStep(phi, y, width_n);
  }
  if (skip_steps > 0){
      I_extern = 0.;
      bfpulse_EulerStep(phi, y, skip_steps); 
  }
  I_extern = 0.;
}

//Init params
double * init_MF () 
{
  double *y;
  y = dvector(1, 2);  
  return y;
}

//Init params
double * init (int nptrans) 
{
  double *phi;
  phi = dvector(1, neqn);
  for(int i=1; i<=neqn; i++) phi[i] = pi2 / neqn * i;  // initial conditions uniformly on the circle
  I_extern = 0.;
  bfpulse_EulerStep(phi, init_MF(), nptrans);    
  return phi;
}

double * init_state_copy (double *phi) 
{
  double *phi_copy;
  phi_copy=dvector(1, neqn);
  for(int k=1;k<=neqn;k++) phi_copy[k] = phi[k];  // store the state before the pulse   
  return phi_copy;
}