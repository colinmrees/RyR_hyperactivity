  //0         ! Open
  //1	      ! CAR
  //2	      ! ODR
  //3	      ! CDR
  //4	      ! OAI
  //5	      ! CAI
  //6	      ! ODI
  //7         ! CDI
  //8         Unused//Does not exist


#define pow2(x) ((x)*(x))

__device__ int lcc_markov(curandState *state,int i, double cpx,double v, double dt, double r1, double s1t, double s1pref, double cat, double cpt)
{
  curandState localState=*state;
  //int jj;



  double dv5 = 5;//-12.6;
  double dvk = 8;//6;

  double fv5 = -22.8;
  double fvk = 9.1;//6.1;

  double alphac = 0.22; //still want to reduce this to get the right ica, cat
  double betac = 4; //3
  
#ifdef iso2
  betac=2;
  dv5 = 0;//-12.6;
  fv5 = -28;//-36; -23; -28
  fvk = 8.5;
#endif

  double dinf = 1.0/(1.0+exp(-(v-dv5)/dvk));
  double taudin = 1.0/((1.0-exp(-(v-dv5)/dvk))/(0.035*(v-dv5))*dinf);
  if( (v < 0.0001) && (v > -0.0001) )
      taudin = 0.035*dvk/dinf;
  //if( v < -80 )
  //dinf = 0;
  double finf = //1.0/(1.0+exp((v-fv5)/fvk));
    1.-1.0/(1.0+exp(-(v-fv5)/fvk))/(1.+exp((v-60)/12.));
  double taufin = (0.02-0.007*exp(-pow2(0.0337*(v+10.5))));//(0.0197*exp(-pow2(0.0337*(v+10.5)))+0.02); //14.5
    //1/((10+4954*exp(v/15.6)-450)/(1+exp(-(v+40)/4.))+450);
  
  //if( !((i/8)%2) )
  //taufin = 1./( 22+250*exp(-pow2(v+40)/700)+exp((v+140)/35) );
  //else
  //taufin = 1./( 5+10*exp(-pow2(v+60)/700)+0.1*exp((v+140)/35) );
    
  

  double alphad = dinf*taudin;
  double betad = (1-dinf)*taudin;
  
  double alphaf = (finf)*taufin;
  double betaf = (1-finf)*taufin;
  
  double alphafca = 0.012/2.;
  double betafca = .0012/2.*cpx*cpx;
  betafca = 0.175/(1+pow2(25/cpx));
  
  double ragg=curand_uniform_double(&localState);
  *state=localState;
  double rig = ragg/dt;
  
  if ( (i%2) )
    if ( rig < alphac )
      return i-1;
    else
      rig-=alphac;
  else
    if ( rig < betac )
      return i+1;
    else
      rig-=betac;
  
  if ( ((i/2)%2) )
    if ( rig < alphad )
      return i-2;
    else
      rig-=alphad;
  else
    if ( rig < betad )
      return i+2;
    else
      rig-=betad;
  
  
  if ( ((i/4)%2) )
    if ( rig < alphaf )
      return i-4;
    else
      rig-=alphaf;
  else
    if ( rig < betaf )
      return i+4;
    else
      rig-=betaf;
  
  
  if ( ((i/8)%2) )
    if ( rig < alphafca )
      return i-8;
    else
      rig-=alphafca;
  else
    if ( rig < betafca )
      return i+8;
    else
      rig-=betafca;
  
  return(i);

  /*
  switch (i){
    case 7:
      if ( rig < betad )
	jj = 6;
      else if ( rig < betad+betaf )
	jj = 5;
      else if ( rig < betad+betaf+betaca )
	jj = 3;
      else
	jj = i;
      break;
    case 6:
      if ( rig < alphad )
	jj = 7;
      else if ( rig < alphad+betaf )
	jj = 4;
      else if ( rig < alphad+betaf+betaca )
	jj = 2;
      else
	jj = i;
      break;
    case 5:
      if ( rig < betad )
	jj = 4;
      else if ( rig < betad+alphaf )
	jj = 7;
      else if ( rig < betad+alphaf+betaca )
	jj = 1;
      else
	jj = i;
      break;
    case 4:
      if ( rig < alphad )
	jj = 5;
      else if ( rig < alphad+alphaf )
	jj = 6;
      else if ( rig < alphad+alphaf+betaca )
	jj = 0;
      else
	jj = i;
      break;
    case 3:
      if ( rig < betad )
	jj = 2;
      else if ( rig < betad+betaf )
	jj = 1;
      else if ( rig < betad+betaf+alphaca )
	jj = 7;
      else
	jj = i;
      break;
    case 2:
      if ( rig < alphad )
	jj = 3;
      else if ( rig < alphad+betaf )
	jj = 0;
      else if ( rig < alphad+betaf+alphaca )
	jj = 6;
      else
	jj = i;
      break;
    case 1:
      if ( rig < betad )
	jj = 0;
      else if ( rig < betad+alphaf )
	jj = 3;
      else if ( rig < betad+alphaf+alphaca )
	jj = 5;
      else
	jj = i;
      break;
    case 0:
      if ( rig < alphad )
	jj = 1;
      else if ( rig < alphad+alphaf )
	jj = 2;
      else if ( rig < alphad+alphaf+alphaca )
	jj = 4;
      else
	jj = i;
      break;
    }     
  
  return(jj);
  */
}



