#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>


#define vnaca (21.0/1.0)
#define Kmcai 0.00359
#define Kmcao 1.3
#define Kmnai 12.3
#define Kmnao 87.5
#define Kda   0.0003		//mM
#ifdef sato
#define ksat  0.2
#else
#define ksat  0.27
#endif
#define eta   0.35
// #define nao   136.0

#define Farad 96.485		//	C/mmol
#define xR	8.314		//	J/mol/K
#define Temper	308       	//K
#define Cext (1*1.8)		// mM
#define a 78.0
#define b 10.0

#define nM 20     //15.0  //CHANGED FOR CHECKSUM
#define nD 30     //35.0  //CHANGED FOR CHECKSUM
#define ratedimer 5000.0

#define kdimer 850.0
#define hilldimer 23.0
#define CSQbers	460.0
#define kbers 600.0

#define Pca 11.9		//	umol/C/ms
#define gammai 0.341
#define gammao 0.341

#define pow6(x) ((x)*(x)*(x)*(x)*(x)*(x))
#define pow4(x) ((x)*(x)*(x)*(x))
#define pow3(x) ((x)*(x)*(x))
#define pow2(x) ((x)*(x))


__device__ double ncx (double cs, double v,double tperiod, double xnai, double *Ancx, double dt)		//Na_Ca exchanger
{
  //double kncxoff = 4.4e-5;//9.4e-5;
  //double kncxon = 1e-2  /   (1+pow4(0.7/(cs*1000)));// 3e-2 
 double Inaca;
 double t1;
 double t2;
 double t3;
 double Ka;
 double za=v*Farad/xR/Temper;
 //Ka=1.0/(1.0+pow3(Kda/cs));
 Ka = *Ancx;
#ifdef slowncxchange
 Ka = 0.025*12.;
 Ka = *Ancx*5;
#endif
 t1=Kmcai*pow3(xnao)*(1.0+pow3(xnai/Kmnai));
 t2=pow3(Kmnao)*cs*(1.0+cs/Kmcai);		  //i'm not sure what this is (check hund/Rudy 2004)
 //t2=pow3(Kmnao)*cs+pow3(Kmnai)*Cext*(1.0+cs/Kmcai); //expression from mahajan
 t3=(Kmcao+Cext)*pow3(xnai)+cs*pow3(xnao);
 Inaca=Ka*vnaca*(exp(eta*za)*pow3(xnai)*Cext-exp((eta-1.0)*za)*pow3(xnao)*cs)/((t1+t2+t3)*(1.0+ksat*exp((eta-1.0)*za)));
 double Ancxdot;
 //Ancxdot = (1.0/(1.0+pow3(0.0003/cs))-*Ancx)/20000.;
 Ancxdot = (1.0/(1.0+pow3(0.0003/cs))-*Ancx)/150.;
 //Ancxdot = (1.0/(1.0+pow3(0.0009/cs))-*Ancx)/2000.;
   
 *Ancx += Ancxdot*dt;
 return (Inaca);
}
 
__device__ double oldncx (double cs, double v,double tperiod, double xnai, double *Ancx, double dt)		//Na_Ca exchanger
{
  double kncxoff = 0.00005;
  double kncxon = 2.53e-3*0.05;
 double Inaca;
 double t1;
 double t2;
 double t3;
 double Ka;
 double za=v*Farad/xR/Temper;
 Ka=1.0/(1.0+pow3(Kda/cs));
 //Ka = *Ancx;
 t1=Kmcai*pow3(xnao)*(1.0+pow3(xnai/Kmnai));
 // t2=pow3(Kmnao)*cs*(1.0+cs/Kmcai);		  //i'm not sure what this is (check hund/Rudy 2004)
 t2=pow3(Kmnao)*cs+pow3(Kmnai)*Cext*(1.0+cs/Kmcai); //expression from mahajan
 t3=(Kmcao+Cext)*pow3(xnai)+cs*pow3(xnao);
 Inaca=Ka*vnaca*(exp(eta*za)*pow3(xnai)*Cext-exp((eta-1.0)*za)*pow3(xnao)*cs)/((t1+t2+t3)*(1.0+ksat*exp((eta-1.0)*za)));
 double Ancxdot = (kncxon * pow2(cs*1000) *( 1-*Ancx) - *Ancx*kncxoff);
 if (  kncxon * pow2(cs*1000)*dt > 0.95)
   Ancxdot = 0.95/dt *( 1-*Ancx) - *Ancx*kncxoff;
   
 *Ancx += Ancxdot*dt;
 return (Inaca);
}

__device__ double ncx_noallo (double cs, double v,double tperiod, double xnai)		//Na_Ca exchanger
{
 double Inaca;
 double t1;
 double t2;
 double t3;
 double Ka;
 double za=v*Farad/xR/Temper;
 Ka = 1.;
 if ( v < -75 ) 
   Ka=1.0/(1.0+pow3(Kda/cs));
 t1=Kmcai*pow3(xnao)*(1.0+pow3(xnai/Kmnai));
 // t2=pow3(Kmnao)*cs*(1.0+cs/Kmcai);		  //i'm not sure what this is (check hund/Rudy 2004)
 t2=pow3(Kmnao)*cs+pow3(Kmnai)*Cext*(1.0+cs/Kmcai); //expression from mahajan
 t3=(Kmcao+Cext)*pow3(xnai)+cs*pow3(xnao);
 Inaca=Ka*vnaca*(exp(eta*za)*pow3(xnai)*Cext-exp((eta-1.0)*za)*pow3(xnao)*cs)/((t1+t2+t3)*(1.0+ksat*exp((eta-1.0)*za)));
	
 return (Inaca);
}

__device__ double currentdsi( double cs, double ci, double tausi_v)		//diffusion from cs to ci
{double Idsi;
 Idsi=(cs-ci)/tausi_v;
 return(Idsi);
}

__device__ double uptake(double ci, double cnsr, double t)			//uptake
{double Iup;
 double vup=0.3;    //0.3 for T=400ms
 double Ki=0.123;
 double Knsr=1700.0;			//1700 for T=400ms
 double HH=1.787;
 double upfactor=1.0;			//factor of SERCA increasing
 Iup=upfactor*vup*(pow(ci/Ki,HH)-pow(cnsr/Knsr,HH))/(1.0+pow(ci/Ki,HH)+pow(cnsr/Knsr,HH));
 return(Iup);
}

__device__ double leak(double cnsr, double ci)			//leak from nsr
{double Ileak;
 double gleak=0.00001035;
 double Kjsr=500.0;
 
 Ileak=gleak*(cnsr-ci)*pow2(cnsr)/(pow2(cnsr)+pow2(Kjsr));
 return(Ileak);
 }
 


__device__ double currenttr(double cnsr, double cjsr, double taure)	//refilling from Nsr to Jsr
{
 
 double Itr;
 Itr=(cnsr-cjsr)/taure;
 return (Itr);
}




__device__ double luminal(double cjsr)			//luminal buffer
{
double beta;	
double roo2;
double mono;
double ene;
double enedot;

 
	 roo2=ratedimer/(1.0+pow(kdimer/cjsr,hilldimer));
	 mono=(-1.0+sqrt(1.0+8.0*CSQbers*roo2))/(4.0*roo2*CSQbers);
	 ene=mono*nM+(1.0-mono)*nD;
	 enedot=(nM-nD)*(-mono + 1.0/(4.0*CSQbers*mono*roo2 + 1.0))*(hilldimer/cjsr)*(1.0 - roo2/ratedimer);
	 beta=1.0/(1.0 + (CSQbers*kbers*ene + CSQbers*enedot*cjsr*(cjsr+kbers))/pow2((kbers+cjsr)));   //CHANGED FOR CHECKSUM
	 return(beta);
	 
} 

__device__ double luminal_SS(double cjsr)			//luminal buffer
{
double B;	
double roo2;
double mono;
double ene;

 
	 roo2=ratedimer/(1.0+pow(kdimer/cjsr,hilldimer));
	 mono=(-1.0+sqrt(1.0+8.0*CSQbers*roo2))/(4.0*roo2*CSQbers);
	 ene=mono*nM+(1.0-mono)*nD;

	 B = CSQbers*ene*cjsr/(kbers+cjsr);

	 return(B);
	 
} 






__device__ double currentdps(double cp, double cs, double tauptemp)		//diffusion from the proximal to cs
{double Idps;
 Idps=(cp-cs)/tauptemp;
 return (Idps);
}

__device__ double couplingNsr (double ccc0, double ccc1, double ccc2, double ccc3, double ccc4, double ccc5, double ccc6, double taul_l, double taul_r, double taut_u, double taut_d)
{ double Icoup;
    Icoup=(ccc1-ccc0)/taul_l+(ccc2-ccc0)/taul_r+(ccc3-ccc0)/taut_d+(ccc4-ccc0)/taut_u+(ccc5-ccc0)/taut_d+(ccc6-ccc0)/taut_u;
	return(Icoup);
}

//coupling effect from neighbours
__device__ double couplingI (double ccc0, double ccc1, double ccc2, double ccc3, double ccc4, double ccc5, double ccc6, double taul, double taut)
  { double Icoup;
    Icoup=(ccc2+ccc1-2.0*ccc0)/taul+(ccc4+ccc3-2.0*ccc0)/taut+(ccc6+ccc5-2.0*ccc0)/taut;
	return(Icoup);
  }


__device__ double release( double po, double cjsr, double cp)	//SR release current
{
  double Jmax = (0.0147*svjmax);
 double Ir;
 Ir=1.0*Jmax*po*(cjsr-cp)/Vp;
 return (Ir);
}


__device__ double lcccurrent(double v, double cp)		//Ica
{
 double za=v*Farad/xR/Temper;
 double ica; 
	
 if (fabs(za)<0.001) 
 {
   ica=2.0*Pca*Farad*gammai*(cp/1000.0*exp(2.0*za)-Cext);
 }
 else 
 {
   ica=4.0*Pca*za*Farad*gammai*(cp/1000.0*exp(2.0*za)-Cext)/(exp(2.0*za)-1.0);
 }
 if (ica > 0.0)
   ica=0.0;
 return (ica);
}


										//RyR gating
__device__ int ryrgating (curandState *state, double Ku, double Ku2, double Kb, double cp, double cjsr, int * ncu, int * nou, int * ncb, int * nob, double *dt, double t, int i, int j, int k)
 {

   int ret = 0;

   curandState localState=*state;
   double roo2;
   double aub;
   double abu;

   double pku, pkb;
   double ryrpref = 4.5*1.5;//4. * (1. + stable/4); //2. / 4&Ku=3&1.65*3kmin
   double cygatek = Ku; //18. / ( 1 + 1.65*stable );
   double ryrhill = 2;//1 + 1/( 1+ 0.111*stable );
   
   pku = ryrpref*Ku2   * 1/(1+pow2(5000/cjsr)) * 1/(1+pow(cygatek/cp,ryrhill));
   pkb = ryrpref/36.*Kb * 1/(1+pow2(5000/cjsr)) * 1/(1+pow(cygatek/cp,ryrhill));

   //pku = ryrpref*Ku2   * 1/(1+pow2(5000/cjsr)) * pow(cp/cygatek,ryrhill);
   //pkb = ryrpref/36.*Kb * 1/(1+pow2(5000/cjsr)) * pow(cp/cygatek,ryrhill);


   //alternatively 30000

   //getting time to peak to work, it seems to me that the problem is in ryrhill.  Removing that should give me the right load-time to peak trend. Do more advanced runs with many loads at each property (pref, gatek, hill).

   //pku=Ku*0.00038*pow2(cp);
   //pkb=   0.00005*pow2(cp);




   double pkuminus=1./taucu;
   double pkbminus=1./taucb;
   //pkuminus=1./(taucu * ( 1 + 1.5*stable ));
   //pkbminus=1./(taucb * ( 1 + 1.5*stable ));


   double puu;
   double pkuh;
   double lamplus;
   double pkum;
   double lamminus;
   double pau;
   double lamau;
   double pbu;
   double lambu;
   double pcb;
   double lamcb;
   double pcub;
   double lamcub;
   double pkbh;
   double lamplbs;
   double pkbm;
   double lamminbs;
 
   double u1;
   double u2;
   double re;

   int n_ou_cu;
   int n_cu_ou;
   int n_ou_ob;
   int n_ob_ou;
   int n_cu_cb;
   int n_cb_cu;
   int n_ob_cb;
   int n_cb_ob;
   int kk;
 
 
   if (pku*(*dt) > 1.0) pku = 1.0/(*dt);
   if (pkb*(*dt) > 1.0) pkb = 1.0/(*dt);
   if (pkuminus*(*dt) > 1.0) pkuminus = 1.0/(*dt);
   if (pkbminus*(*dt) > 1.0) pkbminus = 1.0/(*dt);
 
 
   roo2=ratedimer/(1.0+pow(kdimer/(cjsr),hilldimer));
   aub=(-1.0+sqrt(1.0+8.0*CSQbers*roo2))/(4.0*roo2*CSQbers)/taub;
   abu=1.0/tauu;
   
   //__________________________ going from OU >> CU
   pkuh = pkuminus*(*dt);
   lamplus = exp(-(*nou)*pkuh);
   n_ou_cu=-1;
   if ((*nou) <=1 || pkuh < 0.2 || ((*nou) <=5 && pkuh < 0.3))
     {//***********
       kk = 0;
       puu = 1.0;
       while(puu >= lamplus && kk < 195)	  //generates poisson number = fraction of closed RyR's that open
	 {
	   kk++;
	   re=curand_uniform_double(&localState);
	   puu=puu*(re);
	 }
       n_ou_cu = kk-1;
     }
   else
     {//***********
       kk = 0;
     while(n_ou_cu < 0)
       {
	 //next is really a gaussian
	 u1=curand_uniform_double(&localState);
	 u2=curand_uniform_double(&localState);
	 n_ou_cu = floor((*nou)*pkuh +sqrt((*nou)*pkuh*(1.0-pkuh))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))+curand(&localState)%2;
	 kk++;
	 if( kk > 200){
	   n_ou_cu = 0;
	   ret = 100000*i+1000*j+10*k+1;
	 }
       }
       }//**********
   if(n_ou_cu > nryr) {n_ou_cu = nryr;}
   //____________________
   //_____________________going from CU --> OU
   pkum = pku*(*dt);
   lamminus = exp(-(*ncu)*pkum);
   n_cu_ou = -1;
   if((*ncu) <= 1 ||pkum < 0.2 || ((*ncu) <= 5 && pkum < 0.3))	//checks if we use gaussian or poisson approx
     {//***********
       kk = 0;
       puu = 1.0;
       while(puu >= lamminus && kk < 195)						//generates poisson number = fraction of closed RyR's that open
	 {
	   kk++;
	   re=curand_uniform_double(&localState);
	   puu=puu*re;
	 }
       n_cu_ou = kk-1;
     }
   else
     {//***********
       kk = 0;
     while(n_cu_ou < 0)
       {             //next is really a gaussian
	 u1=curand_uniform_double(&localState);
	 u2=curand_uniform_double(&localState);
	 n_cu_ou = floor((*ncu)*pkum +sqrt((*ncu)*pkum*(1.0-pkum))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))+curand(&localState)%2;
	 kk++;
	 if( kk > 200){
	 n_cu_ou = 0;
	   ret = 100000*i+1000*j+10*k+2;
	 }
       }
       }//**********
   if(n_cu_ou > nryr) {n_cu_ou = nryr;}
   //____________________
			
		
   //_____________________going from OU --> OB
   if( pkb < 1e-16 )
     pau=0;
   else
     pau = aub*(*dt);
   lamau = exp(-(*nou)*pau);
   n_ou_ob = -1;
   if((*nou) <= 1 || pau < 0.2 || (*nou <= 5 && pau < 0.3))	//checks if we use gaussian or poisson approx
     {//***********
       kk = 0;
       puu = 1.0;
       while(puu >= lamau && kk < 195)						//generates poisson number = fraction of open RyR's that close
	 {
	   kk++;
	   re=curand_uniform_double(&localState);
	   puu=puu*re;
	 }
       n_ou_ob = kk-1;
     }
   else
     {//***********
       kk = 0;
     while(n_ou_ob < 0)
       {
	 //next is really a gaussian
	 u1=curand_uniform_double(&localState);
	 u2=curand_uniform_double(&localState);
	 n_ou_ob = floor(*nou*pau +sqrt(*nou*pau*(1.0-pau))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))+curand(&localState)%2;
	 kk++;
	 if( kk > 200){
	 n_ou_ob = 0;
	   ret = 100000*i+1000*j+10*k+3;
	 }
       }
       }//***********
   if(n_ou_ob > nryr) n_ou_ob = nryr;
			
   //______________________
   //_____________________going from OB ---> OU
   if( pkb < 1e-16 )
     pbu = abu*(*dt)*36;
   else
     pbu = abu*(*dt)*(pku/pkb);
   lambu = exp(-(*nob)*pbu);
   n_ob_ou = -1;
			
   if((*nob) <= 1 || pbu < 0.2 || (*nob <= 5 && pbu < 0.3))	//checks if we use gaussian or poisson approx
     {//***********
       kk = 0;
       puu = 1.0;
       while(puu >= lambu && kk < 195)						//generates poisson number = fraction of open RyR's that close
	 {
	   kk++;
	   re=curand_uniform_double(&localState);
	   puu=puu*re;
	 }
       n_ob_ou = kk-1;
				
     }
			
   else
     {//***********
       kk = 0;
     while(n_ob_ou < 0)
       {				//next is really a gaussian
	 u1=curand_uniform_double(&localState);
	 u2=curand_uniform_double(&localState);
	 n_ob_ou = floor(*nob*pbu +sqrt(*nob*pbu*(1.0-pbu))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))+curand(&localState)%2;
	 kk++;
	 if( kk > 200){
	 n_ob_ou = 0;
	   ret = 100000*i+1000*j+10*k+4;
	 }
       }
       }//***********
   if(n_ob_ou > nryr) n_ob_ou = nryr;
   //______________________
			
			
   //_____________________going from CB---->CU
   pcb = abu*(*dt);
   lamcb = exp(-(*ncb)*pcb);
   n_cb_cu = -1;
   if((*ncb) <= 1 || pcb < 0.2 	|| (pcb < 0.3 && (*ncb) <= 5))	//checks if we use gaussian or poisson approx
     {//***********
       kk = 0;
       puu = 1.0;
       while(puu >= lamcb && kk < 195)						//generates poisson number = fraction of open RyR's that close
	 {
	   kk++;
	   re=curand_uniform_double(&localState);
	   puu=puu*re;
	 }
       n_cb_cu = kk-1;
     }
   else
     {//***********
       kk = 0;
     while(n_cb_cu < 0)
       {				//next is really a gaussian
	 u1=curand_uniform_double(&localState);
	 u2=curand_uniform_double(&localState);
	 n_cb_cu = floor((*ncb)*pcb +sqrt((*ncb)*pcb*(1.0-pcb))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))+curand(&localState)%2;
	 kk++;
	 if( kk > 200){
	 n_cb_cu = 0;
	   ret = 100000*i+1000*j+10*k+5;
	 }
       }
       }//***********
   if(n_cb_cu > nryr) {n_cb_cu = nryr;}
   //______________________
   //_____________________going from CU---->CB
   pcub = aub*(*dt);
   lamcub = exp(-*ncu*pcub);
   n_cu_cb = -1;
   if(*ncu <= 1 || pcub < 0.2 || (*ncu <= 5 && pcub < 0.3))	//checks if we use gaussian or poisson approx
     {//***********
       kk = 0;
       puu = 1.0;
       while(puu >= lamcub && kk < 195)						//generates poisson number = fraction of open RyR's that close
	 {
	   kk++;
	   re=curand_uniform_double(&localState);
	   puu=puu*re;
	 }
       n_cu_cb = kk-1;
     }
   else
     {//***********
       kk = 0;
     while(n_cu_cb < 0)
       {				//next is really a gaussian
	 u1=curand_uniform_double(&localState);
	 u2=curand_uniform_double(&localState);
	 n_cu_cb = floor(*ncu*pcub +sqrt(*ncu*pcub*(1.0-pcub))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))+curand(&localState)%2;
	 kk++;
	 if( kk > 200){
	 n_cu_cb = 0;
	   ret = 100000*i+1000*j+10*k+6;
	 }
       }
       }//***********
   if(n_cu_cb > nryr) {n_cu_cb = nryr;}
   //______________________

		
   //_____________________going from OB --> CB
   pkbh = pkbminus*(*dt);
   lamplbs = exp(-(*nob)*pkbh);
   n_ob_cb = -1;
   if((*nob) <= 1 || pkbh < 0.2 || ((*nob) <= 5 && pkbh < 0.3))	//checks if we use gaussian or poisson approx
     {//***********
       kk = 0;
       puu = 1.0;
       while(puu >= lamplbs && kk < 195)						//generates poisson number = fraction of closed RyR's that open
	 {
	   kk++;
	   re=curand_uniform_double(&localState);
	   puu=puu*re;
	 }
       n_ob_cb = kk-1;
     }
   else
     {//***********
       kk = 0;
     while(n_ob_cb < 0)
       {            //next is really a gaussian
	 u1=curand_uniform_double(&localState);
	 u2=curand_uniform_double(&localState);
	 n_ob_cb = floor((*nob)*pkbh +sqrt((*nob)*pkbh*(1.0-pkbh))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))+curand(&localState)%2;
	 kk++;
	 if( kk > 200){
	 n_ob_cb = 0;
	 ret = 100000*i+1000*j+10*k+7;
	 }
       }
       }//**********
   //if(n_ob_cb > nn) {n_ob_cb = nn;}
   //____________________
   //_____________________going from CB --> OB
   pkbm = pkb*(*dt);
   lamminbs = exp(-(*ncb)*pkbm);
   n_cb_ob = -1;
   if((*ncb) <= 1 || pkbm < 0.2 || ((*ncb) <= 5 && pkbm < 0.3))	//checks if we use gaussian or poisson approx
     {//***********
       kk = 0;
       puu = 1.0;
       while(puu >= lamminbs && kk < 195)						//generates poisson number = fraction of closed RyR's that open
	 {
	   kk++;
	   re=curand_uniform_double(&localState);
	   puu=puu*re;
	 }
       n_cb_ob = kk-1;
     }
   else
     {//***********
       kk = 0;
     while(n_cb_ob < 0)
       {
	 //next is really a gaussian
	 u1=curand_uniform_double(&localState);
	 u2=curand_uniform_double(&localState);
	 n_cb_ob = floor((*ncb)*pkbm +sqrt((*ncb)*pkbm*(1.0-pkbm))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))+curand(&localState)%2;
	 kk++;
	 if( kk > 200){
	 n_cb_ob = 0;
	   ret = 100000*i+1000*j+10*k+8;
	 }
       }
     }//**********
   if(n_cb_ob > nryr) {n_cb_ob = nryr;}
   //____________________

			
   if(n_ou_ob  +  n_ou_cu > *nou)
     {
       if(n_ou_cu >= n_ou_ob) n_ou_cu = 0;
       else  n_ou_ob = 0;
       if (n_ou_ob > *nou) n_ou_ob = 0;
       else if(n_ou_cu > *nou) n_ou_cu = 0;
     }
			
   if(n_ob_ou  +  n_ob_cb > *nob)
     { 
       if(n_ob_ou >= n_ob_cb) n_ob_ou = 0;
       else  n_ob_cb = 0;
       if (n_ob_cb > *nob) n_ob_cb = 0;
       else if(n_ob_ou > *nob) n_ob_ou = 0;
     }
			
   if(n_cu_ou  +  n_cu_cb > *ncu ) 
     {
       if(n_cu_cb >= n_cu_ou) n_cu_cb = 0;
       else  n_cu_ou = 0;
       if (n_cu_ou > *ncu) n_cu_ou = 0;
       else if(n_cu_cb > *ncu) n_cu_cb = 0;
     }
			
			
   *nou += -n_ou_ob  -  n_ou_cu    +n_ob_ou  + n_cu_ou;
   if(*nou<0)				
     (*nou)=0;
   if(*nou>nryr)		
     *nou=nryr;	
		
   *nob += -n_ob_ou  -  n_ob_cb    +n_ou_ob  + n_cb_ob;
   if(*nob<0)				
     *nob=0;
   if(*nob>nryr)	
     *nob=nryr;
			
   *ncu += -n_cu_ou  -  n_cu_cb    +n_ou_cu  + n_cb_cu;
   if(*ncu<0)	
     *ncu=0;
   if(*ncu>nryr)
     *ncu=nryr;

   *ncb=nryr-*nou-*nob-*ncu;
	
   if(*ncb<0)	
     *ncb=0;
   if(*ncb>nryr)	
     *ncb=nryr;

   *state=localState;
   return ret;
 }
			
			
			

	
			
