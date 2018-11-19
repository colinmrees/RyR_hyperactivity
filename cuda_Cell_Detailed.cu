/*---------------------------------------------------------------
*
* This code is based on the rabbit ventricular myocyte model of Restrepo et al (Biophysical Journal, 2008) and modified for increased physical realism and physiological accuracy.
*
* The code was used to produce the results published in
*Title: Hyperphosphorylation of RyRs Underlies Triggered Activity in Transgenic Rabbit Model of LQT2 Syndrome
*Authors: Dmitry Terentyev, Colin M. Rees, Weiyan Li, Leroy L. Cooper, Hitesh K. Jindal, Xuwen Peng, Yichun Lu, Radmila Terentyeva, Katja E. Odening, Jean Daley, Kamana Bist, Bum-Rak Choi, Alain Karma, and Gideon Koren
*Journal: Circulation Research
*Year: 2014
*
* For more information on this research, please contact:
*
* Center for interdisciplinary research on complex systems
* Departments of Physics, Northeastern University
*
* Alain Karma a.karma (at) northeastern.edu
*--------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>		     //All subroutines 
#include <curand.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
using namespace std;
#define lumgate         //Luminal gating. Necessary in current model

//#define randomlcc     //Option to randomize number of LCC at each dyad.
//#define iso             //Isoproterenol, increases uptake and I_Ca,L 
//#define hyper           //Caffine RyR Hyperactivity. Increases closed->open rate
//#define nopacing      //Option to not pace, but check for wave behavior
//#define perm          //Permeabalized cell. No sarcolemmal ion channels
/*
#define lqt             //Long-QT 2 syndrome simulation. No I_k,r
#define iso2            //weaker isoproterenol, increases uptake and I_Ca,L
*/
//#define tapsi           //thapsgargin, kills uptake

//CURRENT CHANGES IMPLEMENTED -- tau_u changed, additional uptake added


// iso2 5
#define outputlinescan
//#define outputcadist
#define time_iso 4.5
#define nbeat 7	//number of pacing beat
#define tperid 4000.0		//PCL in the unit of ms.
#define DT 0.025
//#define slowncxchange

//#define apclamp       //action potential clamp. numbers from v.txt. need to re-acumulate this file
//#define vclamp        //step function voltage clamp
//#define brownvclamp

#ifdef vclamp
#define clampvoltage atof(argv[3])
#endif
//#define linearapclamp

#define outt 2.5 //0.1 //2.5

#define pi 3.1415926		 
#define Vp (0.00126) 	//Volume of the proximal space
#define Vjsr (0.02*1)		//Volume of the Jsr space
#define Vi (0.5/8.)		//Volume of the Local cytosolic
#define Vnsr (0.025/8.)		//Volume of the Local Nsr space
#define nryr (100)	       //Number of Ryr channels
#define taups (0.0283) //0.032 //0.029 //0.022	//Diffusion time from the proximal to the submembrane
#define taupi (0.1)   //0.07   //0.09  //10000
#define tausi (0.04)  //0.14 //0.12  //0.1	//Diffusion time from the submembrane to the cytosolic
#define tautr (25.0*1.0/4.)	//Diffusion time from NSR to JSR 
#define taunl 6.	        //Longitudinal NSR
#define taunt 1.8	        //Transverse NSR
#define tauil (0.7*2.) //0.4	//Longitudinal cytosolic
#define tauit (0.33*2.) //0.4	//Transverse cytosolic
#define taust 1.42//5.33 //inf  //Submembrane along t-tubules
#define nx 64		        //Number of Units in the x direction
#define ny 28		        //Number of Units in the y direction
#define nz 12		        //Number of Units in the z direction
#define frt (96.485/8.314/308.0)   //0.03767  inv = 26.5	
#define xi 0.7			//diffusive coupling coefficient.
//#define ncc 4			//number of LCC channels in each CRU;

#ifdef perm
#define ci_basal 0.3
#define cjsr_basal 900
#else
#define ci_basal 0.0944
#define cjsr_basal atof(argv[4]) //560.0 for a patch clamp 1 sec wait to get to 645
#endif

#define Vs (0.025)
#define	 svica ((0.666)*(1.2)*(atof(argv[9])))
//#define kprefac ((t<(6*tperid))?1.0:3.0)
#define kprefac 1.
#define icagamma (0.5)
#define Cm 45   //35, 45, 57
 
#define svncp 4

#define taucb 1.
#define taucu 1.

#define sv_iup (1.0+0.75)

#define svncx (2.5*atof(argv[5]))

#define svjmax 11.5        //J_max prefactor
//0.02*
#define tauu (0.2*5*220.)//1100.//(165.0*1.0)/ //Unbinding transition rate factor cmrfactor5
#define taub (1.0)//(5.0*1.0)  //Binding transition rate factor 

#define	 svileak 1.0
#define  svtauxs 0.95 //atof(argv[3])

#ifdef lqt
#define svncx_lqt ((1.0)*(1.0))
#ifdef tapsi
#define sviup_lqt 0.05//((1.5)*(1.0))
#else
#define sviup_lqt 0.//((1.5)*(1.0))
#endif
#define svkpr_lqt atof(argv[6]) 
#define	sviks 1.0 //(atof(argv[5]))
#define	svikr 0
#define svtof 1.0 //(atof(argv[5]))// 1.0
#define svtos 1.0 //(atof(argv[5]))
#define svnak 1.0 //(atof(argv[4]))
#define svk1 (1.0)
#else
#define svncx_lqt 1.0
#define sviup_lqt 0.
#define svkpr_lqt atof(argv[6])
#define	sviks 1.0 //(atof(argv[5]))
#define	svikr 3  //THESE CHANGES MIGHT STILL BE GOOD
#define svtof 1.0 //(atof(argv[5])) //atof(argv[3])// 1.0
#define svtos 1.0 //(atof(argv[5]))
#define svnak 1. //atof(argv[4])
#define svk1 (1.0)
#endif

#define  xnao 140.0

#include "subroutine_cuda.c"
#include "reesica.h"//"microsinglemar_unclamped_cuda.h"	   //Call file where LCC Markov is done
#include "buffers.c"
//#include "LCC_luorudy.h"




// CUDA block size
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 7
#define BLOCK_SIZE_Z 4

#define STRIDE			(nx*ny)
#define WIDTH			 nx
#define pos(x,y,z)	(STRIDE*(z)+WIDTH*(y)+(x))

struct sl_bu{
  double casar;
  double casarh;
  double casarj;
  double casarhj;
};

struct cyt_bu{
  double cacal;
  double catf;
  double cats;
  double casr;
  double camyo;
  double mgmyo;
  double cadye;
};

struct cytosol{
  double xiup;
  double xileak;
};

struct cru2{
  curandState state;
  int nsign;
  int nspark;
  double randomi;
  double cp;
  double cjsr;
  double cjsrb;    //CHANGED FOR CHECKSUM
  
  //double xinaca;
  double xire;
  //double xicat;
  int lcc[8];
  int nl;
  int nou;
  int ncu;
  int nob;
  int ncb;
  double po;
  int sparknum;
  double Ancx;
};
  
struct cru{
  double xinaca;
  double xicat;
};
  
// GPU computing kernels
__global__ void	setup_kernel(unsigned long long seed,cru2 *CRU2);
__global__ void Init( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double *ci, double *cinext, double *cnsr, double *cnsrnext, double *cs, double *csnext, double cjsr_b);
__global__ void Compute( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double *ci, double *cinext, double *cnsr, double *cnsrnext, double *cs, double *csnext, double v, double t, double dt, double Ku, double Ku2, double Kb, double xnai, double r1, double sv_ica, double sv_ncx);




void CheckCudaError(const char *msg,int it)
{
  cudaError_t CudaErr = cudaGetLastError();
  if( CudaErr!=cudaSuccess) 
    {
      printf("Cuda error at iteration #%d:\t%s\t%s...\n",it,msg,cudaGetErrorString(CudaErr));
      //exit(EXIT_FAILURE);
    }                         
}

int main(int argc, char **argv)
{	
  // ------------------- Input arguments ----------------   
  int CudaDevice=0;	
  if(argc>=2) 
    {
      int Device=atoi(argv[1]);	// Argument #1: cuda device (default: 0)
      if(Device>=0)
	CudaDevice=Device;	
    }
  cudaSetDevice(CudaDevice);
  //int histoutput = 20;

  FILE * wholecell_scr;
  char fnametrial[500];
  sprintf(fnametrial,"wholetest_%s",argv[2]);
  wholecell_scr=fopen(fnametrial,"w");

  FILE * linescan_y;
  sprintf(fnametrial,"linescan_%s",argv[2]);
  linescan_y=fopen(fnametrial,"w");

  FILE * cadistfile;
  sprintf(fnametrial,"cadist_%s",argv[2]);
  cadistfile=fopen(fnametrial,"w");

  // Allocate arrays memory in CPU 
  size_t ArraySize_cru = nx*ny*nz*sizeof(cru);
  size_t ArraySize_cru2 = nx*ny*nz*sizeof(cru2);
  size_t ArraySize_cyt = 8*nx*ny*nz*sizeof(cytosol);
  size_t ArraySize_cbu = 8*nx*ny*nz*sizeof(cyt_bu);
  size_t ArraySize_sbu = nx*ny*nz*sizeof(sl_bu);
  size_t ArraySize_dos = nx*ny*nz*sizeof(double);
  size_t ArraySize_dol = 8*nx*ny*nz*sizeof(double);

  cru *h_CRU;
  h_CRU = (cru*) malloc(ArraySize_cru);
  cru2 *h_CRU2;
  h_CRU2 = (cru2*) malloc(ArraySize_cru2);
  cytosol *h_CYT;
  h_CYT = (cytosol*) malloc(ArraySize_cyt);
  cyt_bu *h_CBU;
  h_CBU = (cyt_bu*) malloc(ArraySize_cbu);
  sl_bu *h_SBU;
  h_SBU = (sl_bu*) malloc(ArraySize_sbu);
  double *h_ci;
  h_ci = (double*) malloc(ArraySize_dol);
  //double *h_cinext;
  //h_cinext = (double*) malloc(ArraySize_dol);
  double *h_cnsr;
  h_cnsr = (double*) malloc(ArraySize_dol);
  //double *h_cnsrnext;
  //h_cnsrnext = (double*) malloc(ArraySize_dol);
  double *h_cs;
  h_cs = (double*) malloc(ArraySize_dos);
  //double *h_csnext;
  //h_csnext = (double*) malloc(ArraySize_dos);


  // Set paramaters for geometry of computation
  dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
  dim3 numBlocks(nx / threadsPerBlock.x,
		 ny / threadsPerBlock.y,
		 nz / threadsPerBlock.z);
  //Allocate arrays in GPU

  cru *d_CRU;
  cudaMalloc((void**)&d_CRU, ArraySize_cru);
  cru2 *d_CRU2;
  cudaMalloc((void**)&d_CRU2, ArraySize_cru2);
  cytosol *d_CYT;
  cudaMalloc((void**)&d_CYT, ArraySize_cyt);
  cyt_bu *d_CBU;
  cudaMalloc((void**)&d_CBU, ArraySize_cbu);
  sl_bu *d_SBU;
  cudaMalloc((void**)&d_SBU, ArraySize_sbu);

  double *d_ci;
  cudaMalloc((void**)&d_ci, ArraySize_dol);
  double *d_cinext;
  cudaMalloc((void**)&d_cinext, ArraySize_dol);
  double *d_cnsr;
  cudaMalloc((void**)&d_cnsr, ArraySize_dol);
  double *d_cnsrnext;
  cudaMalloc((void**)&d_cnsrnext, ArraySize_dol);
  double *d_cs;
  cudaMalloc((void**)&d_cs, ArraySize_dos);
  double *d_csnext;
  cudaMalloc((void**)&d_csnext, ArraySize_dos);

  //curandState *devStates;
  //cudaMalloc((void**)&devStates,nx*ny*nz*sizeof(curandState));
        
  double  cproxit;	       //For calculating the average concentration of the proximal space
  double  csubt;	       //For calculating the average concentration of the submembrane space
  double  cit;		       //For calculating the average concentration of the cytosolic space
  double  cjsrt;	       //For calculating the average concentration of the JSR space
  double  csqnbt;              //For calculating the average concentration of Ca bound to CSQN in the JSR space
  double  cnsrt;	       //For calculating the average concentration of the NSR space
  double  xicatto;	       //For calculating the average Lcc calcium current in the proximal space
  double  out_ica;             //LCC Strength averaged over output period
  double  out_ina;             //I_na Strength averaged over output period
  double out_ncx;
  double  xinacato;	       //For calculating the average NCX current in the submembrane space
  double  poto;		       //For calculating the average open probability of RyR channels

  int jx;					 //Loop variable
  int jy;					 //Loop variable
  int jz;					 //Loop variable
  double Ku;
  double Ku2;
  double Kb;

  double dt;		    //pacing time interval in the unit of ms
  double t=0.0;		    //pacing time 

  //!!!!!!!!!!!!!unclamped ini tial conditions!!!!!!!!!!!!!!!!	
  //double xnao=136.0;	//mM   ! external Na
  double xki=140.0;  //mM   ! internal K
  double xko=5.40;   //mM    ! external K
  //double cao=1.8;    //mM     ! external Ca      1.8mM
  double xnai=6.1*((9000)/(tperid+5000));
  double v=-80.00;   // voltage
  double xm=0.0010;  // sodium m-gate
  double xh=1.00;    // sodium h-gate
  double xj=1.00;    // soium  j-gate=
  double xr=0.00;    // ikr gate variable
  double xs1=0.08433669901; // iks gate variable
  double xs2=xs1;//0.1412866149; //removed, and replaced with xs1..not sure why
  double qks=0.20;    // iks gate variable
  double xtos=0.010; // ito slow activation
  double ytos=1.00;  // ito slow inactivation
  double xtof=0.020; // ito fast activation
  double ytof=0.80;  // ito slow inactivation
  double xinak;
  double xina;

  int sparksum = 0;

  double sviks_iso = 1.;
  // Ini tialization
  Init<<<numBlocks, threadsPerBlock>>>(d_CRU, d_CRU2, d_CYT, d_CBU, d_SBU, d_ci, d_cinext, d_cnsr, d_cnsrnext, d_cs, d_csnext, cjsr_basal);
 cudaMemcpy(h_CRU2,d_CRU2,ArraySize_cru2,cudaMemcpyDeviceToHost);
	
  //setup_kernel<<<numBlocks,threadsPerBlock>>>(time(NULL),devStates);
 setup_kernel<<<numBlocks,threadsPerBlock>>>(18,d_CRU2);
	
  //Time loop
  t=0;
  dt=DT;

#ifdef apclamp
  int numsteps = ((int)(tperid/dt + 0.1));
  double varray[numsteps];
  std::ifstream file( "v_1000.txt" );
  for( int ii = 0; ii < numsteps; ii++ ){
    file >> varray[ii];
  }
  /*for( int ii = 0; ii < numsteps; ii+=4 ){
    varray[ii+1] = varray[ii]+(varray[ii+4]-varray[ii])/4.;
    varray[ii+2] = varray[ii]+(varray[ii+4]-varray[ii])/2.;
    varray[ii+3] = varray[ii]+(varray[ii+4]-varray[ii])*3/4.;
    }*/
  file.close();
#endif

  cout << "Initialization Complete" << endl;
  while (t<nbeat*tperid+2100){

    Ku2 = atof(argv[10]);
    Kb = atof(argv[8]);
    Ku = (kprefac) * (svkpr_lqt);

    /*if ( v < -80 ){
      Ku2 = 1.5;
      Ku = 1.5;
      Kb = 1;
      }*/

    //if ( t > 32000)
    //Ku = 0;
    //#ifdef iso2
    //if ( t > time_iso * tperid )
    //sviks_iso = 2.;
    //#endif
	
      //Ci updating

    Compute<<<numBlocks, threadsPerBlock>>>( d_CRU, d_CRU2, d_CYT, d_CBU, d_SBU, d_ci, d_cinext, d_cnsr, d_cnsrnext, d_cs, d_csnext, v, t, dt, Ku, Ku2, Kb, xnai, atof(argv[7]), svica, svncx );

      double *tempci, *tempcs, *tempcnsr;
      tempci = d_cinext; tempcs = d_csnext; tempcnsr = d_cnsrnext;
      d_cinext = d_ci; d_csnext = d_cs; d_cnsrnext = d_cnsr;
      d_ci=tempci; d_cs=tempcs; d_cnsr=tempcnsr;
      
      cudaMemcpy(h_CRU, d_CRU, ArraySize_cru, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_cs, d_cs, ArraySize_dos, cudaMemcpyDeviceToHost);

      cit=0;
      csubt=0;
      xicatto=0;
      xinacato=0;
      //CURRENT FROM FLUX
      for (jz = 1; jz < nz-1; jz++)
	for (jy = 1; jy < ny-1; jy++)
	  for (jx = 1; jx < nx-1; jx++) 
	    {
	      //for( int ii = 0; ii < 8; ++ii){
	      //cit=cit+h_ci[pos(jx,jy,jz)*8+ii];
	      //}
	      csubt += h_cs[pos(jx,jy,jz)];
	      xicatto=xicatto+h_CRU[pos(jx,jy,jz)].xicat;
	      xinacato=xinacato+h_CRU[pos(jx,jy,jz)].xinaca;
	    }
      //cit=cit/((nx-2)*(ny-2)*(nz-2))/8.;
      csubt=csubt/((nx-2)*(ny-2)*(nz-2));

      //xicatto=xicatto/((nx-2)*(ny-2)*(nz-2))*1000*(0.18/4)*Vp*2.0;
      //*25000./50*0.0965*2*Vp == 7660
      //96.5*2.58e-5/3.1e-4*396*2 == 6360      
      xicatto=xicatto/Cm*0.0965*Vp*2.0*(icagamma+1.);

      //xinacato=xinacato/((nx-2)*(ny-2)*(nz-2))*1000*(0.18/4)*Vs;
      //*25000./50.*0.0965*Vs
      xinacato=xinacato/Cm*0.0965*Vs;

      //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Sodium current~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
				
      double ena;
      ena= (1.0/frt)*log(xnao/xnai);        // sodium reversal potential
				
      double am;
      am = 0.32*(v+47.13)/(1.0-exp(-0.1*(v+47.13)));
      double bm;
      bm = 0.08*exp(-v/11.0);
				
      double ah,bh,aj,bj;
				
      if(v < -40.0)
	{
	  ah = 0.135*exp((80.0+v)/(-6.8));
	  bh = 3.56*exp(0.079*v)+310000.0*exp(0.35*v);
	  aj=(-127140.0*exp(0.2444*v)-0.00003474*exp(-0.04391*v))*((v+37.78)/(1.0+exp(0.311*(v+79.23))));
	  bj=(0.1212*exp(-0.01052*v))/(1.0+exp(-0.1378*(v+40.14)));
	  //aj=ah; //make j just as h
	  //bj=bh; //make j just as h
	}
      else
	{
	  ah=0.00;
	  bh=1.00/(0.130*(1.00+exp((v+10.66)/(-11.10))));
	  aj=0.00;
	  bj=(0.3*exp(-0.00000025350*v))/(1.0 + exp(-0.10*(v+32.00)));
	  //aj=ah; //make j just as h
	  //bj=bh; //make j just as h
	}
				
      double tauh;
      tauh=1.00/(ah+bh);
      double tauj;
      tauj=1.00/(aj+bj);
      double taum;
      taum=1.00/(am+bm);
				
      double gna=15;//12.00;            // sodium conductance (mS/micro F)
      double gnaleak = 0.3e-3*5;
      xina = gna*xh*xj*xm*xm*xm*(v-ena) + gnaleak*(v-ena);    //with leak added
				
      xh = ah/(ah+bh)-((ah/(ah+bh))-xh)*exp(-dt/tauh);
      xj = aj/(aj+bj)-((aj/(aj+bj))-xj)*exp(-dt/tauj);
      xm = am/(am+bm)-((am/(am+bm))-xm)*exp(-dt/taum);
      //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      //  -------------- Ikr following Shannon  ------------------

				
      double ek;
      ek = (1.00/frt)*log(xko/xki);                // K reversal potential = -86.26
				
      double gss;
      gss=sqrt(xko/5.40);
      double xkrv1;
      xkrv1=0.001380*(v+7.00)/( 1.0-exp(-0.123*(v+7.00))  );
      double xkrv2;
      xkrv2=0.000610*(v+10.00)/(exp( 0.1450*(v+10.00))-1.00);
      double taukr;
      taukr=1.00/(xkrv1+xkrv2);
				
      double xkrinf;
      xkrinf=1.00/(1.00+exp(-(v+50.00)/7.50));
				
      double rg;
      rg=1.00/(1.00+exp((v+33.00)/22.40));
				
      double gkr=0.0078360;  // Ikr conductance
      double xikr;
      xikr=svikr*gkr*gss*xr*rg*(v-ek);
				
      xr = xkrinf-(xkrinf-xr)*exp(-dt/taukr);
      // -----------------------------------------------------------
      // ----- Iks modified from Shannon, with new Ca dependence
      //------------

      double prnak=0.0183300;
				
      double qks_inf; //0.35
      qks_inf=0.2*(1+0.8/(1+pow((0.28/csubt),3)));//0.60*(1.0*csubt);                                  //should use cs in each unit      used 10*ci instead of submembrane (?) for simplicity
      //qks_inf=0.433*(1+0.8/(1+pow((0.5/cit),3)));	//Mahajan qks
      double tauqks=1000.00;
#ifdef slowncxchange
      qks_inf = 0.2*(1+1.6/(1+pow((0.28/csubt),3)));
      tauqks = 2000;
#endif
				
      double eks;
      eks=(1.00/frt)*log((xko+prnak*xnao)/(xki+prnak*xnai));
      double xs1ss;
      xs1ss=1.0/(1.0+exp(-(v-1.500)/16.700));
				
      double tauxs;
      tauxs=1.00/(0.0000719*(v+30.00)/(1.00-exp(-0.1480*(v+30.0)))+0.0001310*(v+30.00)/(exp(0.06870*(v+30.00))-1.00));
      tauxs=svtauxs*tauxs;
      double gksx=0.2000; // Iks conductance
      double xiks;
      xiks = sviks*sviks_iso*gksx*qks*xs1*xs2*(v-eks);
				
      xs1=xs1ss-(xs1ss-xs1)*exp(-dt/tauxs);
      xs2=xs1ss-(xs1ss-xs2)*exp(-dt/tauxs);
      qks=qks+dt*(qks_inf-qks)/tauqks;
				
				
				
				
				
      //~~~~~~~~~~Iks from Mahajan~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      //double prnak=0.018330;
      //				double eks=(1.0/frt)*log((xko+prnak*xnao)/(xki+prnak*xnai));
      //				double xs1ss=1.0/(1.0+exp(-(v-1.50)/16.70));
      //				double xs2ss=xs1ss;
      //				double tauxs1;
      //				if (fabs(v+30.0)<0.001/0.0687)
      //					tauxs1=1/(0.0000719/0.148+0.000131/0.0687);
      //				else
      //					tauxs1=1.0/(0.0000719*(v+30.0)/(1.0-exp(-0.148*(v+30.0)))+0.000131*(v+30.0)/(exp(0.0687*(v+30.0))-1.0));
      //				
      //				tauxs1=svtauxs*tauxs1;
      //				double tauxs2=4*tauxs1;
      //				double gksx=0.433*(1+0.8/(1+pow((0.5/cit),3)));
      //				double gks=0.32;
      //				double xiks=sviks*gks*gksx*xs1*xs2*(v-eks);
      //				xs1=xs1ss-(xs1ss-xs1)*exp(double(-dt/tauxs1));
      //				xs2=xs2ss-(xs2ss-xs2)*exp(double(-dt/tauxs2));
      //

      //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!				
      //-----------------------------------------------
      //      ------  Ik1 following Luo-Rudy formulation (from Shannon model)
      //     ------

      double gkix=0.600; // Ik1 conductance
      double gki;
      gki=gkix*(sqrt(xko/5.4));
      double aki;
      aki=1.02/(1.0+exp(0.2385*(v-ek-59.215)));
      double bki;
      bki=(0.49124*exp(0.08032*(v-ek+5.476))+exp(0.061750*(v-ek-594.31)))/(1.0+exp(-0.5143*(v-ek+4.753)));
      double xkin;
      xkin=aki/(aki+bki);
      double xik1;
      xik1=svk1*gki*xkin*(v-ek);	 
      //---------------------------------
      // ------- Ito slow following Shannon et. al. 2005 ------------

      double rt1;
      rt1=-(v+3.0)/15.00;
      double rt2;
      rt2=(v+33.5)/10.00;
      double rt3;
      rt3=(v+60.00)/10.00;
      double xtos_inf;
      xtos_inf=1.00/(1.0+exp(rt1));
      double ytos_inf;
      ytos_inf=1.00/(1.00+exp(rt2));
				
      double rs_inf;
      rs_inf=1.00/(1.00+exp(rt2));
				
      double txs;
      txs=9.00/(1.00+exp(-rt1)) + 0.50;
      double tys;
      tys=3000.00/(1.0+exp(rt3)) + 30.00; //cmrchange
				
      double gtos=0.040; // ito slow conductance
				
      double xitos;
      xitos=svtos*gtos*xtos*(ytos+0.50*rs_inf)*(v-ek); // ito slow
				
      xtos = xtos_inf-(xtos_inf-xtos)*exp(-dt/txs);
      ytos = ytos_inf-(ytos_inf-ytos)*exp(-dt/tys);
      //----------------------------------------------------------
      // --------- Ito fast following Shannon et. al. 2005 -----------

      double xtof_inf;
      double ytof_inf;
	
	
      xtof_inf=xtos_inf;
      ytof_inf=ytos_inf;		
      double rt4;
      rt4=-(v/30.00)*(v/30.00);
      double rt5;
      rt5=(v+33.50)/10.00;
      double txf;
      txf=3.50*exp(rt4)+1.50;
      double tyf;
      tyf=20.0/(1.0+exp(rt5))+20.00;
      
      /*xtof_inf = 1/(1+exp((v+5.7)/-11.1));
      ytof_inf = 1/(1+exp((v+40)/5)); //34.7, 7.4
      double txf = 0.84 + 6.6*exp(-pow2((v+29)/25));
      double tyf = 12.5 + 98.5*exp(-pow2((v+90)/70));*/
				
      double gtof=0.10;  //! ito fast conductance
				
      double xitof;
      xitof=svtof*gtof*xtof*ytof*(v-ek);
				
      xtof = xtof_inf-(xtof_inf-xtof)*exp(-dt/txf);
      ytof = ytof_inf-(ytof_inf-ytof)*exp(-dt/tyf);

      //-------------------------------------------------------
      //      -------  Inak (sodium-potassium exchanger) following Shannon
      //        --------------
				
      double xkmko=1.50; // these are Inak constants adjusted to fit
      //                // the experimentally measured dynamic restitution
      //                    curve
      double xkmnai=12.00;
      double xibarnak=1.5000;
      double hh=1.00;  // Na dependence exponent
				
      double sigma;
      sigma = (exp(xnao/67.30)-1.00)/7.00;
      double fnak;
      fnak = 1.00/(1.0+0.1245*exp(-0.1*v*frt)+0.0365*sigma*exp(-v*frt));
      xinak = svnak * xibarnak*fnak*(1.0/(1.0+pow((xkmnai/xnai),hh)))*xko/(xko+xkmko);
				
      //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& previous currents from UCLA model
				
      double stim;
				
      //! -------   stimulation -----------------------------
#ifdef iso
      if(fmod(t,tperid) < 1.0 /*&& v<-40*/ && ( t < tperid*11.9 ) )	
	stim = 80.0;	//80
      else							
	stim= 0.0;
#else
      if(fmod(t+tperid-100,tperid) < 1.0 /*&& v<-40*/ )
	stim = 80.0;	//80
      else						
	stim= 0.0;
#endif
#ifdef nopacing
      stim = 0.0;
#endif
#ifdef perm
      stim = 0.0;
#endif
      //! --------------------------------------------------
      double wcainv;
      wcainv=1.0/50.0;      //! conversion factor between pA to micro //! amps/ micro farads
      //		double Vi=12500.0;			  
      double conv = 0.18*12500;  //conversion from muM/ms to pA    (includes factor of 2 for Ca2+)
      double xinaca;
      xinaca=xinacato;   //convert ion flow to current:  net ion flow = 1/2   calcium flow
      //double xica;
      //xica=xicatto;      //xicatotemp has already the factor of 2
				
      //!  --------  sodium dynamics -------------------------
				
      double xrr;
      double trick=1.0;
      xrr=trick*(1.0/wcainv/conv)/1000.0; // note: sodium is in m molar    
                                             //so need to divide by 1000
      double dnai;
      dnai=-xrr*(xina +3.0*xinak+3.0*xinaca);
      xnai += dnai*dt;
      //if ( t < tperid*(nbeat-5) )
      //xnai += dnai*dt*5;
      //if ( t < tperid*(nbeat-15) )
      //xnai += dnai*dt*50;
				
      //xnai = 1.0*78.0/(1.0 + 10.0*sqrt(tperid/1000.0)); //since it takes so many iterations to equilibrate Na, this function caputures SS sodium.
      //xnai=11;   //Or hold sodium at experimental value


      //! --------  dV/dt ------------------------------------
				
      double dvh;
      dvh=-(xina +xik1+1*xikr+1*xiks+xitos+1*xitof+1.*xinacato+1*xicatto+xinak) + stim; 
	
#ifdef vclamp
      v=-86;
      if( t > 100 )   //allow 1 second of rest before voltage clamp is applied
	v = clampvoltage;
#else
      v += dvh*dt;
#endif

#ifdef brownvclamp
      if( t < 4000 )
	v = -86;
      else if ( t < 4200 )
	v = -10;
      else if (t < 6200 )
	v = 10;
      else if (t < 7200 )
	v= -30;
      else
	v = -86;
#endif

#ifdef apclamp
      //if ( t > tperid*(nbeat-1)+0.001 )
	v = varray[((int)( t/dt+0.1 ))%((int)(tperid/dt+0.1))];
#endif
#ifdef linearapclamp
      if ( t < 100 )
	v = -86;
      else if ( t < 101.865 )
	v = 68*(t-100)-80;
      else if ( t < 119.1 )
	v = -(t-100)+48;
      else if ( t < 138.9 )
	v = 27+0.1*(t-100);
      else if ( t < 364.43 )
	v = 43-(t-100)/3.2;
      else if ( t < 395.43 )
	v = -1.5*(t-100)+357;
      else
	v=-86;
#endif
      out_ica += xicatto;
      out_ncx += xinacato;
      out_ina += xina ;
      if (fmod(t+0.0000001,outt)<dt )
	{
	  
	  cudaMemcpy(h_CRU2,d_CRU2, ArraySize_cru2,cudaMemcpyDeviceToHost);
	  cudaMemcpy(h_CYT,d_CYT, ArraySize_cyt,cudaMemcpyDeviceToHost);
	  cudaMemcpy(h_CBU, d_CBU, ArraySize_cbu, cudaMemcpyDeviceToHost);
	  cudaMemcpy(h_SBU, d_SBU, ArraySize_sbu, cudaMemcpyDeviceToHost);
	  cudaMemcpy(h_ci, d_ci, ArraySize_dol, cudaMemcpyDeviceToHost);
	  cudaMemcpy(h_cnsr, d_cnsr, ArraySize_dol, cudaMemcpyDeviceToHost);
        

	  cproxit=0;
	  //csubt=0;
	  cjsrt=0;
	  csqnbt=0;
	  cnsrt=0;
	  poto=0;
	  double pbcto=0;
	  double csub2t=0;
	  double csub3t=0;
	  double ncxfwd=0;

	  double isit=0;
	  double catft=0;
	  double catst=0;
	  double casrt=0;
	  double camyot=0;
	  double mgmyot=0;
	  double cacalt=0;
	  double cadyet=0;

	  double casart = 0;
	  double casarht = 0;
	  double casarjt = 0;
	  double casarhjt = 0;

	  double leakt=0;
	  double upt=0;
	  double ire=0;
	  
	  int tn1 = 0;
	  int tn2 = 0;
	  int tn3 = 0;
	  int tn4 = 0;
	  int tn5 = 0;
	  int tn6 = 0;
	  int tn7 = 0;
	  //int nltot = 0;	  int nltot2 = 0;
	  
	  int tnou = 0;
	  int tnob = 0;
	  int tncu = 0;
	  int tncb = 0;

	  int tcruo = 0;
	  int tcruo2 = 0;
	  int tcruo3 = 0;
	  int tcruo4 = 0;
	  double icaflux = 0;
	  double ncxflux = 0;

	  double outAncx = 0;
	  for (jz = 1; jz < nz-1; jz++)
	    for (jy = 1; jy < ny-1; jy++)
	      for (jx = 1; jx < nx-1; jx++) 
		{	

		  icaflux=icaflux+h_CRU[pos(jx,jy,jz)].xicat;
		  ncxflux=ncxflux+h_CRU[pos(jx,jy,jz)].xinaca;
		  if (  h_CRU[pos(jx,jy,jz)].xinaca < 0 )
		    ncxfwd=ncxfwd+h_CRU[pos(jx,jy,jz)].xinaca;
		  outAncx+=h_CRU2[pos(jx,jy,jz)].Ancx;


		  cproxit=cproxit+h_CRU2[pos(jx,jy,jz)].cp;
		  //csubt=csubt+h_cs[pos(jx,jy,jz)];
		  csub2t=csub2t+h_cs[pos(jx,jy,jz)]*h_cs[pos(jx,jy,jz)];
		  csub3t=csub3t+h_cs[pos(jx,jy,jz)]*h_cs[pos(jx,jy,jz)]*h_cs[pos(jx,jy,jz)];
		  cjsrt=cjsrt+h_CRU2[pos(jx,jy,jz)].cjsr;
		  csqnbt=csqnbt+h_CRU2[pos(jx,jy,jz)].cjsrb;
		  poto=poto+h_CRU2[pos(jx,jy,jz)].po;
		  pbcto=pbcto+h_CRU2[pos(jx,jy,jz)].ncb;

		  casart += h_SBU[pos(jx,jy,jz)].casar;
		  casarht += h_SBU[pos(jx,jy,jz)].casarh;
		  casarjt += h_SBU[pos(jx,jy,jz)].casarj;
		  casarhjt += h_SBU[pos(jx,jy,jz)].casarhj;

		  for ( int ii = 0; ii < 8; ++ii ){
		    cit+=h_ci[pos(jx,jy,jz)*8+ii]/8.;
		    cnsrt=cnsrt+h_cnsr[pos(jx,jy,jz)*8+ii]/8.;
		    catft= catft+h_CBU[pos(jx,jy,jz)*8+ii].catf/8.;
		    catst= catst+h_CBU[pos(jx,jy,jz)*8+ii].cats/8.;
		    casrt= casrt+h_CBU[pos(jx,jy,jz)*8+ii].casr/8.;
		    camyot= camyot+h_CBU[pos(jx,jy,jz)*8+ii].camyo/8.;
		    mgmyot= mgmyot+h_CBU[pos(jx,jy,jz)*8+ii].mgmyo/8.;
		    cacalt= cacalt+h_CBU[pos(jx,jy,jz)*8+ii].cacal/8.;
		    cadyet= cadyet+h_CBU[pos(jx,jy,jz)*8+ii].cadye/8.;
		    leakt= leakt+h_CYT[pos(jx,jy,jz)*8+ii].xileak/8.;
		    upt=  upt+h_CYT[pos(jx,jy,jz)*8+ii].xiup/8.;

		    if( h_ci[pos(jx,jy,jz)*8+ii] > 1000 ){
		      cout << t << " " << jx << " " << jy << " " << jz << " " << ii << "error!" << endl;
		    }
		  }


		  ire += h_CRU2[pos(jx,jy,jz)].xire;

		  tnou += h_CRU2[pos(jx,jy,jz)].nou;
		  tnob += h_CRU2[pos(jx,jy,jz)].nob;
		  tncu += h_CRU2[pos(jx,jy,jz)].ncu;
		  tncb += h_CRU2[pos(jx,jy,jz)].ncb;

		  if ( h_CRU2[pos(jx,jy,jz)].nou + h_CRU2[pos(jx,jy,jz)].nob > 30 )
		    ++tcruo;
		  if ( h_CRU2[pos(jx,jy,jz)].nou + h_CRU2[pos(jx,jy,jz)].nob > 40 )
		    ++tcruo2;
		  if ( h_CRU2[pos(jx,jy,jz)].nou + h_CRU2[pos(jx,jy,jz)].nob > 30 || 
		       h_CRU2[pos(jx,jy,jz)].nob + h_CRU2[pos(jx,jy,jz)].ncb > 50 )
		    ++tcruo3;
		  if ( h_CRU2[pos(jx,jy,jz)].ncu < 20 )
		    ++tcruo4;

		  sparksum += h_CRU2[pos(jx,jy,jz)].nspark;


		  for( int jj = 0; jj < 8; ++jj ){
		    switch (h_CRU2[pos(jx,jy,jz)].lcc[jj]){
		      case 1: ++tn1; ++tn6; break;
		      case 2: ++tn2; break;
		      case 3: ++tn1; ++tn2; break;
		      case 4: ++tn3; ++tn5; break;
		      case 5: ++tn1; ++tn3; ++tn5; break;
		      case 6: ++tn2; ++tn3; ++tn5; break;
		      case 7: ++tn1; ++tn2; ++tn3; ++tn5; break;
		      case 8: ++tn4; break;
		      case 9: ++tn1; ++tn4; break;
		     case 10: ++tn2; ++tn4; break;
		     case 11: ++tn1; ++tn2; ++tn4; break;
		     case 12: ++tn3; ++tn4; break;
		     case 13: ++tn1; ++tn3; ++tn4; break;
		     case 14: ++tn2; ++tn3; ++tn4; break;
		     case 15: ++tn1; ++tn2; ++tn3; ++tn4; break;
		      case 0: ++tn7; ++tn6; break;
		    }
		  }


		  //nltot += (h_CRU2[pos(jx,jy,jz)].nl?1:0);
		  //nltot2 += h_CRU2[pos(jx,jy,jz)].nl;
				  
		}
	  //
	  cproxit=cproxit/((nx-2)*(ny-2)*(nz-2));
	  //csubt=csubt/((nx-2)*(ny-2)*(nz-2));
	  csub2t=csub2t/((nx-2)*(ny-2)*(nz-2));
	  csub3t=csub3t/((nx-2)*(ny-2)*(nz-2));
	  cjsrt=cjsrt/((nx-2)*(ny-2)*(nz-2));
	  csqnbt=csqnbt/((nx-2)*(ny-2)*(nz-2));
	  cnsrt=cnsrt/((nx-2)*(ny-2)*(nz-2));
	  cit/=((nx-2)*(ny-2)*(nz-2));
	  poto=poto/((nx-2)*(ny-2)*(nz-2));
	  pbcto=pbcto/((nx-2)*(ny-2)*(nz-2))/100.;
	  //ncxavg = 0;//ncx_h(csubt/1000, v, 0, xnai);

	  isit=isit/((nx-2)*(ny-2)*(nz-2));
	  catft= catft/((nx-2)*(ny-2)*(nz-2));
	  catst= catst/((nx-2)*(ny-2)*(nz-2));
	  casrt= casrt/((nx-2)*(ny-2)*(nz-2));
	  camyot/=((nx-2)*(ny-2)*(nz-2));
	  mgmyot/=((nx-2)*(ny-2)*(nz-2));
	  cacalt= cacalt/((nx-2)*(ny-2)*(nz-2));
	  cadyet= cadyet/((nx-2)*(ny-2)*(nz-2));
	  leakt= leakt/((nx-2)*(ny-2)*(nz-2));
	  upt=upt/((nx-2)*(ny-2)*(nz-2));
	  ire=ire/((nx-2)*(ny-2)*(nz-2));
	  ncxflux/=((nx-2)*(ny-2)*(nz-2));
	  ncxfwd/=((nx-2)*(ny-2)*(nz-2));
	  icaflux/=((nx-2)*(ny-2)*(nz-2));
	  outAncx/=((nx-2)*(ny-2)*(nz-2));

	   casart /=((nx-2)*(ny-2)*(nz-2));
	   casarht /=((nx-2)*(ny-2)*(nz-2));
	   casarjt /=((nx-2)*(ny-2)*(nz-2));
	   casarhjt /=((nx-2)*(ny-2)*(nz-2));

	  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~				
	  //~~~~~~~~~~~~~~~~~~~~~~~~~~~ Line Scan ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~		
	  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	
#ifdef outputlinescan
	  for (jx =1; jx < nx-1; jx++)
	  //for ( jy = 1; jy < ny-1; ++jy)
	    {
	      //int jx = 31;
	      int jz = 5;
	      int jy = ny/2;
	      fprintf(linescan_y, "%f %f %f %i %i ""%i %i %i %f %f "
		                  "%f %f %f %f %f ""%f %f %f %f %f "
                                  "%f %f\n",
		      t,(double)jx,
		      h_ci[pos(jx,jy,jz)*8], h_CRU2[pos(jx,jy,jz)].nob, 
		      h_CRU2[pos(jx,jy,jz)].nou, h_CRU2[pos(jx,jy,jz)].ncb, 
		      h_CRU2[pos(jx,jy,jz)].ncu, h_CRU2[pos(jx,jy,jz)].nl, 
		      h_cs[pos(jx,jy,jz)], h_CRU2[pos(jx,jy,jz)].cp,

		      h_SBU[pos(jx,jy,jz)].casar,  h_SBU[pos(jx,jy,jz)].casarh, 
		      h_CBU[pos(jx,jy,jz)*8].camyo,  h_CBU[pos(jx,jy,jz)*8].mgmyo, 
		      h_CBU[pos(jx,jy,jz)*8].casr,  h_CBU[pos(jx,jy,jz)*8].cacal, 
		      h_CBU[pos(jx,jy,jz)*8].cats, h_CBU[pos(jx,jy,jz)*8].catf, 
		      h_CRU2[pos(jx,jy,jz)].xire, h_CRU2[pos(jx,jy,jz)].cjsr,//(h_CRU2[pos(jx,jy,jz)].nl?1:0), 

		      h_CRU2[pos(jx,jy,jz)].Ancx, h_CRU[pos(jx,jy,jz)].xinaca );
	      
	      /*
	      fprintf(linescan_y, "%f %f %f\n",t,(double)jx,h_ci[pos(jx,jy,jz)*8]);
	      fprintf(linescan_y, "%f %f %f\n",t,(double)jx+0.5,h_ci[pos(jx,jy,jz)*8+4]);*/
	      
		      //h_CBU[pos(jx,jy,jz)*8+4].cadye*kdyeoff/kdyeon/(Bdye-h_CBU[pos(jx,jy,jz)*8+4].cadye)  );
	    }
	  fprintf(linescan_y, "\n");
	  fflush(linescan_y);
#endif

#ifdef outputcadist
	  if ( fmod(t+0.0000001,5)<dt &&  fmod(t+0.0000001,tperid) < 500 )
	    {

	      fprintf(cadistfile, "0 header %f %f\n", t, v );
	      for (jx =1; jx < nx-1; jx++){
		for (jy =1; jy < ny-1; jy++){
		  for (jz =1; jz < nz-1; jz++){
		    fprintf(cadistfile, "%f %f %f %f %f %f\n", h_ci[pos(jx,jy,jz)*8], h_cs[pos(jx,jy,jz)], h_CRU2[pos(jx,jy,jz)].cp, h_CRU2[pos(jx,jy,jz)].cjsr, h_CRU[pos(jx,jy,jz)].xinaca/Cm*0.0965*Vs*16120., h_CRU2[pos(jx,jy,jz)].xire );
		  }
		}
	      }
	      //histoutput -= 10;
	    }
	  /*if( t > 28000-dt/10.  && t < 30000-dt/10. ){
	    for (jx =1; jx < nx-1; jx++){
	      for (jy =1; jy < ny-1; jy++){
		for (jz =1; jz < nz-1; jz++){
		  fprintf(cadistfile, "%f ", ( h_ci[pos(jx,jy,jz)*8] + h_ci[pos(jx,jy,jz)*8] + h_ci[pos(jx,jy,jz)*8] + h_ci[pos(jx,jy,jz)*8] + h_ci[pos(jx,jy,jz)*8] + h_ci[pos(jx,jy,jz)*8] + h_ci[pos(jx,jy,jz)*8] + h_ci[pos(jx,jy,jz)*8] )/8. );
		}
	      }
	    }
	    fprintf(cadistfile, "\n");
	    }*/
#endif




	  
	  double Total_Ca = ( cit + catft + catst + casrt + camyot + cacalt ) * Vi*8. +
	    ( csubt + casart + casarht ) * Vs +
	    ( cproxit + casarjt + casarhjt ) * Vp +
	    ( cnsrt ) * Vnsr*8. +
	    ( cjsrt + csqnbt ) * Vjsr;

	  double Total_Ci = ( cit + catft + catst + casrt + camyot + cacalt )* Vi*8. +
	    ( csubt + casart + casarht ) * Vs +
	    ( cproxit + casarjt + casarhjt ) * Vp;
	  
	  fprintf(wholecell_scr,"%f %f %f %f %f  %f %f %f %f %f  "
                                "%f %f %f %f %f  %f %f %f %f %f  "
                                "%f %f %f %f %f  %f %i %i %i %i  "
                                "%i %i %i %i %i  %i %i %f %f %f  "
                                "%i %f %f %f %f\n",
t,cit,
		  v,out_ncx/(outt/dt),
out_ica/(outt/dt),cproxit,
csubt,cjsrt,
cnsrt,poto,

xnai,xiks,
xikr,xik1,
xinak,xitos,
xitof, out_ina/(outt/dt),
xr, ncxfwd*(Vs/Vp), 

pbcto, cacalt, 
catft, leakt, 
upt, ire,
tnou, tnob,
tncu,tncb,

tn1,tn2,
tn3, tn4,
tn5, tn6,
tn7, outAncx,
xs1, qks,

		  sparksum, catst,             //ncxflux*(Vs/Vp),   //##############CHANGE FOR CHECKSUM CODE
		  csqnbt, Total_Ci,       //icaflux, casarjt, //##############CHANGE FOR CHECKSUM CODE
		  Total_Ca); //casarhjt );

	  sparksum = 0;
	  out_ica = 0;
	  out_ncx = 0;
	  out_ina = 0;
	  //fclose(wholecell_scr);
	  fflush( wholecell_scr );
	  //cout << t << " " << h_cs[9480] << " " << h_CRU2[9480].Ancx << " " << h_CRU[9480].xinaca << " " <<  h_CRU2[9480].cp << " " << h_SBU[9480].casar << " " << h_SBU[9480].casarh << " " << h_CRU[9480].xicat << " " << h_cs[9480+STRIDE] << " " << h_cs[9480-STRIDE] << " " << h_CRU2[9480].nl << endl;
	}
      //if ( fmod(t, tperid) < dt )					
      t=t+dt;
	
    }
	
  fclose(wholecell_scr);
  fclose(linescan_y);
  //fclose(linescan_y);

  cudaFree(d_CYT);
  cudaFree(d_CRU);
  cudaFree(d_CRU2);
  cudaFree(d_SBU);
  cudaFree(d_CBU);
	
  free(h_CYT);
  free(h_CRU);
  free(h_CRU2);
  free(h_SBU);
  free(h_CBU);
	
  return EXIT_SUCCESS;
	
}

//setupinit
__global__ void Init( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double *ci, double *cinext, double *cnsr, double *cnsrnext, double *cs, double *csnext, double cjsr_b)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int ac = pos(i,j,k);
#ifdef randomlcc
  curandState localState;
  localState=CRU2[ac].state;
#endif
#ifdef tapsi
  cjsr_b = 10;
#endif
  CRU2[ac].randomi=1.0;
  CRU2[ac].cp=ci_basal;
  cs[ac]=ci_basal;
  csnext[ac]=ci_basal;
  CRU2[ac].cjsr=cjsr_b;
  CRU2[ac].cjsrb=luminal_SS(cjsr_b);
  CRU2[ac].nspark=0;
  CRU2[ac].Ancx = 0.025; //0.05;

  for ( int ii = 0; ii < 8; ++ii ){
    ci[ac*8+ii]=ci_basal;
    cnsr[ac*8+ii]=cjsr_b;
    cinext[ac*8+ii]=ci_basal;
    cnsrnext[ac*8+ii]=cjsr_b;

    CBU[ac*8+ii].catf=ktfon*ci_basal*Btf/(ktfon*ci_basal+ktfoff);
    CBU[ac*8+ii].cats=ktson*ci_basal*Bts/(ktson*ci_basal+ktsoff);
    CBU[ac*8+ii].cacal=kcalon*ci_basal*Bcal/(kcalon*ci_basal+kcaloff);
    CBU[ac*8+ii].cadye=kdyeon*ci_basal*Bdye/(kdyeon*ci_basal+kdyeoff);
    CBU[ac*8+ii].casr=ksron*ci_basal*Bsr/(ksron*ci_basal+ksroff);
    CBU[ac*8+ii].camyo=3.;   //2.800963;
    CBU[ac*8+ii].mgmyo=136.209996;
  }
  SBU[ac].casar=ksaron*ci_basal*Bsar/(ksaron*ci_basal+ksaroff);
  SBU[ac].casarh=ksarhon*ci_basal*Bsarh/(ksarhon*ci_basal+ksarhoff);
  SBU[ac].casarj=ksaron*ci_basal*Bsar/(ksaron*ci_basal+ksaroff);
  SBU[ac].casarhj=ksarhon*ci_basal*Bsarh/(ksarhon*ci_basal+ksarhoff);
  
  int ll;
#ifdef randomlcc
    int randpoint = 0+(curand(&localState))%9;
#endif
#ifndef randomlcc
    int randpoint = svncp;
#endif
  
  for(ll=0; ll<8; ll++){
    if ( ll < randpoint )
      CRU2[ac].lcc[ll]=3;
    else
      CRU2[ac].lcc[ll]=16;
  }

  
  double roo2 = ratedimer/(1.0+pow(kdimer/(cjsr_b),hilldimer));
  double aub=(-1.0+sqrt(1.0+8.0*CSQbers*roo2))/(4.0*roo2*CSQbers)/taub;
  double abu=1.0/tauu;

  double fracbound = 1/(1+abu/aub);

  CRU2[ac].ncb=int(fracbound*nryr);
  CRU2[ac].ncu=nryr-int(fracbound*nryr);
  CRU2[ac].nob=0;
  CRU2[ac].nou=0;

  /*#ifdef nopacing  //This creates a plane of activated CRUs, to test whether a wave propagates.
  if( i == nx/2. ){
    CRU2[ac].ncb=int(0.25*nryr);
    CRU2[ac].nou=int(0.3*nryr);
    }
    #endif*/
}




__global__ void Compute( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double *ci, double *cinext, double *cnsr, double *cnsrnext, double *cs, double *csnext, double v, double t, double dt, double Ku, double Ku2, double Kb, double xnai, double r1, double sv_ica, double sv_ncx)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  int ac = pos(i,j,k);

  double dotcjsr;
  double dotci[8];
  double dotcnsr[8];
  double xicoupn[8];
  double xicoupi[8];
	
  double sviup;
  double s1pref;
  double s1t;
  //double r1;
	
  sviup=sv_iup;
  s1pref=1;
  s1t=1*0.00195;
  //r1=0.3;

  double cat = 0.50*1.;
  double cpt = 1.50*1.;

#ifdef iso
  if (fmod(t+10*tperid,15.5*tperid)<dt){
    for( int ll = 4; ll < 8; ++ll)
      if( CRU2[ac].lcc[ll] == 16 )
	CRU2[ac].lcc[ll]=3;
  }

  if (t>=5.5*tperid)
    {
      sviup=sv_iup*2.;
      //s1pref=0.5;
      //s1t=0.5*0.00195;
      //r1=3.0*0.3;
      cat = 0.5*2;
      cpt = 1.5*2;
    }
#endif
#ifdef iso2
  sviup = sv_iup*1.75;
  /*if (t>=time_iso*tperid)
    {
    }*/
#endif
#ifdef tapsi
  sviup=0;
#endif
  curandState localState;
  localState=CRU2[ac].state;
  //CALCIUM DYNAMICS COMPUTATION
  if((i*j*k)!=0 && i<nx-1 && j<ny-1 && k<nz-1)
    {	
      if (fmod(t-dt,outt)<dt){
	CRU2[ac].nspark=0;
	CRU2[ac].nl=0;
      }
      double icat1 = 0;
#ifndef perm
      int jj;
      int ll;
      int nlcp = 0;
      for (ll=0; CRU2[ac].lcc[ll]<16 && ll < 8; ll++)
	{
	  jj=lcc_markov(&localState,CRU2[ac].lcc[ll],(CRU2[ac].cp+cs[ac]*icagamma)/(1+icagamma),v,dt,r1,s1t,s1pref, cat, cpt);
	  CRU2[ac].lcc[ll]=jj;
	  if (jj==0 ){
	    ++CRU2[ac].nl;
	    ++nlcp;
	  }
	}
      icat1 = sv_ica*(double)(nlcp)*lcccurrent(v,(CRU2[ac].cp+cs[ac]*icagamma)/(1+icagamma));
#endif

      double pmca = 0;//0.18*1.15*cs[ac]/(3.6+cs[ac]);
      double bcgcur = 0;
#ifdef slowncxchange
      bcgcur = 1.0*0.003016/32.*(v-log(180000/cs[ac])/2./frt);
#endif


      CRU[ac].xicat=icat1+pmca+bcgcur*(Vs/Vp);//I_pmca added;

      int prevsp = CRU2[ac].nou+CRU2[ac].nob;
      //int prevsp = CRU2[ac].ncu;

      //if( v > -73 )
	int problem = ryrgating(&localState,Ku, Ku2, Kb, CRU2[ac].cp,CRU2[ac].cjsr, &CRU2[ac].ncu, &CRU2[ac].nou, &CRU2[ac].ncb, &CRU2[ac].nob,&dt,t,i,j,k);
	  
	//else
	//ryrgating(&localState,3.2, Ku2, Kb, CRU2[ac].cp,CRU2[ac].cjsr, &CRU2[ac].ncu, &CRU2[ac].nou, &CRU2[ac].ncb, &CRU2[ac].nob,&dt,t,i,j,k);
      
      if( prevsp < 5 && (CRU2[ac].nou+CRU2[ac].nob) >= 5 )
	  CRU2[ac].nspark = 1;
      
      /*if( prevsp > 50 && CRU2[ac].ncu <= 50 )
	  CRU2[ac].nspark = 1;
	if ( problem )
	CRU2[0].nspark = problem;*/

      CRU2[ac].po=(double)(CRU2[ac].nou+CRU2[ac].nob)/(double)(nryr);
      CRU2[ac].xire=release(CRU2[ac].po,CRU2[ac].cjsr,CRU2[ac].cp)/CRU2[ac].randomi;
      if	(CRU2[ac].xire<0) CRU2[ac].xire=0;
#ifndef perm
      //if( t < 20*tperid )
	CRU[ac].xinaca=sv_ncx*svncx_lqt*ncx(cs[ac]/1000.0,v,tperid,xnai, &CRU2[ac].Ancx, dt);
	//if( t > (nbeat-3)*tperid )
	//CRU[ac].xinaca=0;
      //else
      //CRU[ac].xinaca=sv_ncx*svncx_lqt*oldncx(cs[ac]/1000.0,v,tperid,xnai, &CRU2[ac].Ancx, dt);
	
#else
      CRU[ac].xinaca=0;
#endif
      for ( int ii = 0; ii < 8; ++ii ){
	CYT[ac*8+ii].xiup=(sviup+sviup_lqt)*uptake(ci[ac*8+ii],cnsr[ac*8+ii],t);
	CYT[ac*8+ii].xileak=svileak*leak(cnsr[ac*8+ii],ci[ac*8+ii]);
      }
      double diffpi0 = (CRU2[ac].cp-ci[ac*8])/taupi/2.;
      double diffpi1 = (CRU2[ac].cp-ci[ac*8+1])/taupi/2.;
      double diffsi0 = (cs[ac]-ci[ac*8])/tausi/2.;
      double diffsi1 = (cs[ac]-ci[ac*8+1])/tausi/2.;
      double diffjn0 = (CRU2[ac].cjsr-cnsr[ac*8])/tautr/2.;
      double diffjn1 = (CRU2[ac].cjsr-cnsr[ac*8+1])/tautr/2.;

      dotcjsr=luminal(CRU2[ac].cjsr)*((-diffjn0-diffjn1)-CRU2[ac].xire*Vp*CRU2[ac].randomi/Vjsr);
      CRU2[ac].cjsrb += (1-luminal(CRU2[ac].cjsr))*((-diffjn0-diffjn1)-CRU2[ac].xire*Vp*CRU2[ac].randomi/Vjsr)*dt;     //CHANGED FOR CHECKSUM
      //CRU2[ac].cjsrb = luminal_SS(CRU2[ac].cjsr);
      
      for ( int ii = 0; ii < 8; ++ii ){
	int act = ac*8+ii;
	int north = (ii%2)?(pos(i,j,k+1)*8+ii-1):(ac*8+ii+1);
	int south = (ii%2)?(ac*8+ii-1):(pos(i,j,k-1)*8+ii+1);
	int east = ((ii/2)%2)?(pos(i,j+1,k)*8+ii-2):(ac*8+ii+2);
	int west = ((ii/2)%2)?(ac*8+ii-2):(pos(i,j-1,k)*8+ii+2);
	int top = ((ii/4)%2)?(pos(i+1,j,k)*8+ii-4):(ac*8+ii+4);
	int bottom = ((ii/4)%2)?(ac*8+ii-4):(pos(i-1,j,k)*8+ii+4);

	xicoupn[ii] = ( (cnsr[north]-cnsr[act])/(taunt*xi) +
			(cnsr[south]-cnsr[act])/(taunt*xi) +
			(cnsr[east]-cnsr[act])/(taunt*xi) +
			(cnsr[west]-cnsr[act])/(taunt*xi) +
			(cnsr[top]-cnsr[act])/(taunl*xi) +
			(cnsr[bottom]-cnsr[act])/(taunl*xi) );
	xicoupi[ii] = ( (ci[north]-ci[act])/(tauit*xi) +
			(ci[south]-ci[act])/(tauit*xi) +
			(ci[east]-ci[act])/(tauit*xi) +
			(ci[west]-ci[act])/(tauit*xi) +
			(ci[top]-ci[act])/(tauil*xi) +
			(ci[bottom]-ci[act])/(tauil*xi) );
	
	
	dotcnsr[ii]=(CYT[act].xiup-CYT[act].xileak)*Vi/Vnsr+xicoupn[ii];
	
	double buffers = ( tropf(CBU[act].catf, ci[act], dt)+
			   trops(CBU[act].cats, ci[act], dt)+
			   bucal(CBU[act].cacal, ci[act], dt)+
			   busr(CBU[act].casr, ci[act], dt)+
			   myoca(CBU[act].camyo, CBU[act].mgmyo, ci[act], dt) );

	dotci[ii]=(-CYT[act].xiup
		   +CYT[act].xileak)
		   -buffers
		   +xicoupi[ii];
      }
      dotcnsr[0] += diffjn0*Vjsr/Vnsr;
      dotcnsr[1] += diffjn1*Vjsr/Vnsr;
      dotci[0] += diffsi0*(Vs/Vi);
      dotci[1] += diffsi1*(Vs/Vi);
      dotci[0] += diffpi0*(Vp/Vi);
      dotci[1] += diffpi1*(Vp/Vi);


      for ( int ii = 0; ii < 8; ++ii ){
	int act = ac*8+ii;
	cinext[act]=ci[act]+dotci[ii]*dt;
	if (cinext[act]<0) 
	  cinext[act]=0;
	cnsrnext[act]=cnsr[act]+dotcnsr[ii]*dt;

	CBU[act].catf += tropf(CBU[act].catf,ci[act], dt)*dt;
	if( CBU[act].catf < 0 )
	  CBU[act].catf =0;
	CBU[act].cats += trops(CBU[act].cats,ci[act], dt)*dt;
	if( CBU[act].cats < 0 )
	  CBU[act].cats =0;
	CBU[act].cacal += bucal(CBU[act].cacal,ci[act], dt)*dt;
	if( CBU[act].cacal < 0 )
	  CBU[act].cacal =0;
	CBU[act].casr += busr(CBU[act].casr,ci[act], dt)*dt;
	if( CBU[act].casr < 0 )
	  CBU[act].casr =0;
	CBU[act].camyo += myoca(CBU[act].camyo,CBU[act].mgmyo,ci[act], dt)*dt;
	if( CBU[act].camyo < 0 )
	  CBU[act].camyo =0;
	CBU[act].mgmyo += myomg(CBU[act].camyo,CBU[act].mgmyo,ci[act], dt)*dt;
	if( CBU[act].mgmyo < 0 )
	  CBU[act].mgmyo =0;
	CBU[act].cadye += budye(CBU[act].cadye,ci[act], dt)*dt;
	if( CBU[act].cadye < 0 )
	  CBU[act].cadye =0;
	  
      }	
      
      
      csnext[ac] = cs[ac];
      double cpstart = CRU2[ac].cp;
      int finestep = 5;
      double dth = dt/finestep;
      for( int iii = 0; iii < finestep; ++iii ){

	double csbuffers = ( busar(SBU[ac].casar,csnext[ac], dth) + 
			     busarh(SBU[ac].casarh,csnext[ac], dth)  );

	double csdiff = (cs[pos(i,j,k+1)]+cs[pos(i,j,k-1)]-2*cs[ac])/(taust); //4.
	//csdiff += (cs[pos(i,j+1,k)]+cs[pos(i,j+1,k)]-2*cs[ac])/(taust);
	//csdiff += (cs[pos(i+1,j,k)]+cs[pos(i-1,j,k)]-2*cs[ac])/(taust);

#ifdef perm
	csdiff += (ci_basal-csnext[ac])/5;
#endif

	double dotcs = Vp/Vs*(CRU2[ac].cp-csnext[ac])/(taups)
	  - icagamma*icat1*Vp/Vs // 0
	  + CRU[ac].xinaca
	  - (cs[ac]-ci[ac*8])/tausi/2.
	  - (cs[ac]-ci[ac*8+1])/tausi/2.
	  + csdiff
	  - csbuffers
	  - bcgcur;

	double cpbuff, cpbuffh;

	cpbuff = busarj(SBU[ac].casarj,CRU2[ac].cp, dth);
	cpbuffh = busarhj(SBU[ac].casarhj,CRU2[ac].cp, dth);


	//ksaron*ci_basal*Bsar/(ksaron*ci_basal+ksaroff);
	/*double kd1 = ksaroff/ksaron;
	double kd2 = ksarhoff/ksarhon;
	double ibuff= 1/(1+ 2*Bsar*kd1/pow2(CRU2[ac].cp+kd1)
			 + 2*Bsarh*kd2/pow2(CRU2[ac].cp+kd2));
	double cpdot = ibuff*( (cs[ac]-CRU2[ac].cp)/taups + 
			 (ci[ac*8]-CRU2[ac].cp)/taupi/2. + 
			 (ci[ac*8+1]-CRU2[ac].cp)/taupi/2. + 
			 CRU2[ac].xire - icat1  ); */
	
	double cpdot = ( (csnext[ac]-CRU2[ac].cp)/taups +
			 (ci[ac*8]-cpstart)/taupi/2. +
			 (ci[ac*8+1]-cpstart)/taupi/2. +
			 CRU2[ac].xire - icat1 - cpbuff - cpbuffh );


	SBU[ac].casarj += cpbuff*dt/finestep;
	if( SBU[ac].casarj < 0 )
	  SBU[ac].casarj =0;
	SBU[ac].casarhj += cpbuffh*dt/finestep;
	if( SBU[ac].casarhj < 0 )
	  SBU[ac].casarhj =0;

	CRU2[ac].cp += cpdot*dt/finestep;

	

	SBU[ac].casar += busar(SBU[ac].casar,csnext[ac], dt/finestep)*dt/finestep;
	if( SBU[ac].casar < 0 )
	  SBU[ac].casar =0;
	SBU[ac].casarh += busarh(SBU[ac].casarh,csnext[ac], dt/finestep)*dt/finestep;
	if( SBU[ac].casarh < 0 )
	  SBU[ac].casarh =0;

	csnext[ac] += (dotcs)*dth;

      }

      if ( CRU2[ac].cp < 0 ) 
	CRU2[ac].cp=0;

      if (csnext[ac]<0){
	csnext[ac]=1e-6;
	//CRU2[0].nspark = ac;
      }
      CRU2[ac].cjsr=CRU2[ac].cjsr+dotcjsr*dt;
    
	dotcnsr[0] += diffjn0*Vjsr/Vnsr;
      dotcnsr[1] += diffjn1*Vjsr/Vnsr;

     
      csnext[ac] = cs[ac];
      int finestep = 5;
      double dth = dt/finestep;
      for( int iii = 0; iii < finestep; ++iii ){
	
	double diffpi0 = (CRU2[ac].cp-ci[ac*8])/taupi/2.;
	double diffpi1 = (CRU2[ac].cp-ci[ac*8+1])/taupi/2.;
	double diffsi0 = (csnext[ac]-ci[ac*8])/tausi/2.;
	double diffsi1 = (csnext[ac]-ci[ac*8+1])/tausi/2.;
	dotci[0] += diffsi0*(Vs/Vi)/finestep;
	dotci[1] += diffsi1*(Vs/Vi)/finestep;
	dotci[0] += diffpi0*(Vp/Vi)/finestep;
	dotci[1] += diffpi1*(Vp/Vi)/finestep;

	double csbuffers = ( busar(SBU[ac].casar,csnext[ac], dth) + 
			     busarh(SBU[ac].casarh,csnext[ac], dth)  );

	double csdiff = (cs[pos(i,j,k+1)]+cs[pos(i,j,k-1)]-2*csnext[ac])/(taust); //4.
	//csdiff += (cs[pos(i,j+1,k)]+cs[pos(i,j+1,k)]-2*cs[ac])/(taust);
	//csdiff += (cs[pos(i+1,j,k)]+cs[pos(i-1,j,k)]-2*cs[ac])/(taust);
#ifdef perm
	csdiff += (ci_basal-csnext[ac])/5;
#endif

	double dotcs = Vp/Vs*(CRU2[ac].cp-icagamma*taups*icat1-csnext[ac])/(taups)
	  + CRU[ac].xinaca
	  - (csnext[ac]-ci[ac*8])/tausi/2.
	  - (csnext[ac]-ci[ac*8+1])/tausi/2.
	  + csdiff
	  - csbuffers
	  - bcgcur;

	double cpbuff, cpbuffh;

	cpbuff = busarj(SBU[ac].casarj,CRU2[ac].cp, dth);
	cpbuffh = busarhj(SBU[ac].casarhj,CRU2[ac].cp, dth);
	double cpdot = ( (cs[ac]-CRU2[ac].cp)/taups + 
			 (ci[ac*8]-CRU2[ac].cp)/taupi/2. + 
			 (ci[ac*8+1]-CRU2[ac].cp)/taupi/2. + 
			 CRU2[ac].xire - icat1 - cpbuff - cpbuffh );


	//For reference: ksaron*ci_basal*Bsar/(ksaron*ci_basal+ksaroff);
	/*
	double kd1 = ksaroff/ksaron;
	double kd2 = ksarhoff/ksarhon;
	double ibuff= 1/(1+ 2*Bsar*kd1/pow2(CRU2[ac].cp+kd1)
			 + 2*Bsarh*kd2/pow2(CRU2[ac].cp+kd2));
	double cpdot = ibuff*( (cs[ac]-CRU2[ac].cp)/taups + 
			 (ci[ac*8]-CRU2[ac].cp)/taupi/2. + 
			 (ci[ac*8+1]-CRU2[ac].cp)/taupi/2. + 
			 CRU2[ac].xire - icat1 );
	*/


	SBU[ac].casar += busar(SBU[ac].casar,csnext[ac], dt/finestep)*dth;
	if( SBU[ac].casar < 0 )
	  SBU[ac].casar =0;
	SBU[ac].casarh += busarh(SBU[ac].casarh,csnext[ac], dt/finestep)*dth;
	if( SBU[ac].casarh < 0 )
	  SBU[ac].casarh =0;

	csnext[ac] += (dotcs)*dth;

	
	SBU[ac].casarj += cpbuff*dth;
	if( SBU[ac].casarj < 0 )
	  SBU[ac].casarj =0;
	SBU[ac].casarhj += cpbuffh*dth;
	if( SBU[ac].casarhj < 0 )
	  SBU[ac].casarhj =0;

	CRU2[ac].cp += cpdot*dth;

      }


      //update cytosol and cytosolic buffers
      for ( int ii = 0; ii < 8; ++ii ){
	int act = ac*8+ii;
	cinext[act]=ci[act]+dotci[ii]*dt;
	if (cinext[act]<0) 
	  cinext[act]=0;
	cnsrnext[act]=cnsr[act]+dotcnsr[ii]*dt;

	CBU[act].catf += tropf(CBU[act].catf,ci[act], dt)*dt;
	if( CBU[act].catf < 0 )
	  CBU[act].catf =0;
	CBU[act].cats += trops(CBU[act].cats,ci[act], dt)*dt;
	if( CBU[act].cats < 0 )
	  CBU[act].cats =0;
	CBU[act].cacal += bucal(CBU[act].cacal,ci[act], dt)*dt;
	if( CBU[act].cacal < 0 )
	  CBU[act].cacal =0;
	CBU[act].casr += busr(CBU[act].casr,ci[act], dt)*dt;
	if( CBU[act].casr < 0 )
	  CBU[act].casr =0;
	CBU[act].camyo += myoca(CBU[act].camyo,CBU[act].mgmyo,ci[act], dt)*dt;
	if( CBU[act].camyo < 0 )
	  CBU[act].camyo =0;
	CBU[act].mgmyo += myomg(CBU[act].camyo,CBU[act].mgmyo,ci[act], dt)*dt;
	if( CBU[act].mgmyo < 0 )
	  CBU[act].mgmyo =0;
	CBU[act].cadye += budye(CBU[act].cadye,ci[act], dt)*dt;
	if( CBU[act].cadye < 0 )
	  CBU[act].cadye =0;
	  
      }	

      
	
      if ( CRU2[ac].cp < 0 ) 
	CRU2[ac].cp=0;

      if (csnext[ac]<0){
	csnext[ac]=1e-6;
	//CRU2[0].nspark = ac;
      }
      CRU2[ac].cjsr=CRU2[ac].cjsr+dotcjsr*dt;
    }
	

  //Boundary conditions

#ifndef perm
  if(k==1)
    {
      for( int ii = 1; ii < 8; ii+=2 ){
	cinext[pos(i,j,0)*8+ii]=cinext[pos(i,j,1)*8+ii-1];
	cnsrnext[pos(i,j,0)*8+ii]=cnsrnext[pos(i,j,1)*8+ii-1];
      }
      csnext[pos(i,j,0)]=csnext[pos(i,j,1)];
    }
  if(k==nz-2)
    {
      for( int ii = 0; ii < 8; ii+=2 ){
	cinext[pos(i,j,nz-1)*8+ii]=cinext[pos(i,j,nz-2)*8+ii+1];
	cnsrnext[pos(i,j,nz-1)*8+ii]=cnsrnext[pos(i,j,nz-2)*8+ii+1];
      }
      csnext[pos(i,j,nz-1)]=csnext[pos(i,j,nz-2)];
    }
  if(j==1)
    {
      for( int ii = 2; ii < 8; ii+=4 ){
	cinext[pos(i,0,k)*8+ii]=cinext[pos(i,1,k)*8+ii-2];
	cinext[pos(i,0,k)*8+ii+1]=cinext[pos(i,1,k)*8+ii-1];
	cnsrnext[pos(i,0,k)*8+ii]=cnsrnext[pos(i,1,k)*8+ii-2];
	cnsrnext[pos(i,0,k)*8+ii+1]=cnsrnext[pos(i,1,k)*8+ii-1];
      }
    }
  if(j==ny-2)
    {
      for( int ii = 0; ii < 8; ii+=4 ){
	cinext[pos(i,ny-1,k)*8+ii]=cinext[pos(i,ny-2,k)*8+ii+2];
	cinext[pos(i,ny-1,k)*8+ii+1]=cinext[pos(i,ny-2,k)*8+ii+3];
	cnsrnext[pos(i,ny-1,k)*8+ii]=cnsrnext[pos(i,ny-2,k)*8+ii+2];
	cnsrnext[pos(i,ny-1,k)*8+ii+1]=cnsrnext[pos(i,ny-2,k)*8+ii+3];
      }
    }
  
  if(i==1)
    {
      for( int ii = 4; ii < 8; ++ii ){
	cinext[pos(0,j,k)*8+ii]=cinext[pos(1,j,k)*8+ii-4];
	cnsrnext[pos(0,j,k)*8+ii]=cnsrnext[pos(1,j,k)*8+ii-4];
      }
    }
  if(i==nx-2)
    {
      for( int ii = 0; ii < 4; ++ii ){
	cinext[pos(nx-1,j,k)*8+ii]=cinext[pos(nx-2,j,k)*8+ii+4];
	cnsrnext[pos(nx-1,j,k)*8+ii]=cnsrnext[pos(nx-2,j,k)*8+ii+4];
      }
    }

  /*
  1 - 0,1,2,3
  0 - 4,5,6,7
    
  
  ((ii/4)%2)?(ac*8+ii-4):(pos(i-1,j,k)*8+ii+4) - ac*8+ii;
  */

#else
  if(k==1)
    for( int ii = 1; ii < 8; ii+=2 )
      cnsrnext[pos(i,j,0)*8+ii]=cnsrnext[pos(i,j,1)*8+ii-1];
  if(k==nz-2)
    for( int ii = 0; ii < 8; ii+=2 )
      cnsrnext[pos(i,j,nz-1)*8+ii]=cnsrnext[pos(i,j,nz-2)*8+ii+1];
  if(j==1)
    {
      for( int ii = 2; ii < 8; ii+=4 ){
	cnsrnext[pos(i,0,k)*8+ii]=cnsrnext[pos(i,1,k)*8+ii-2];
	cnsrnext[pos(i,0,k)*8+ii+1]=cnsrnext[pos(i,1,k)*8+ii-1];
      }
    }
  if(j==ny-2)
    {
      for( int ii = 0; ii < 8; ii+=4 ){
	cnsrnext[pos(i,ny-1,k)*8+ii]=cnsrnext[pos(i,ny-2,k)*8+ii+2];
	cnsrnext[pos(i,ny-1,k)*8+ii+1]=cnsrnext[pos(i,ny-2,k)*8+ii+3];
      }
    }
  if(i==1)
    for( int ii = 4; ii < 8; ++ii )
      cnsrnext[pos(0,j,k)*8+ii]=cnsrnext[pos(1,j,k)*8+ii-4];
  if(i==nx-2)
    for( int ii = 0; ii < 4; ++ii )
      cnsrnext[pos(nx-1,j,k)*8+ii]=cnsrnext[pos(nx-2,j,k)*8+ii+4];
#endif
  CRU2[ac].state=localState;
    }
}

//Initializes the RNG in each thread
__global__ void	setup_kernel(unsigned long long seed, cru2 *CRU2 )  ///curandState *state)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  curand_init(seed,pos(i,j,k),0,&(CRU2[pos(i,j,k)].state)  );
}







