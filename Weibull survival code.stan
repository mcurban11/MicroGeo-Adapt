// after https://rpubs.com/kaz_yos/bayes_surv1
data {
    int<lower=1> Nobs;               //number of Dead - uncensored
    int<lower=1> Ncen;         //number of UnDead - censored
    int<lower=1> M_bg;               //number of locations
    int<lower=1> K;               //number of Trials
    int<lower=1> O;               //number of Trial-Pools
    //int<lower=1> P;               //number of night/day categories
    
    int trials_alive[Ncen]; //vector of alive trials
    int trials_dead[Nobs]; //vector of dead trials
    int tp_alive[Ncen]; //vector of alive trials
    int tp_dead[Nobs]; //vector of dead trials
    //int nt_alive[Ncen]; //vector of alive trials
    //int nt_dead[Nobs]; //vector of dead trials
    
    vector<lower=0>[Ncen] ycen;  //vector of alive event times
    vector<lower=0>[Nobs] yobs;   //vector of dead event times   
    
    matrix[Nobs, M_bg] Xobs_bg;  //matrix of dead locations
    matrix[Ncen, M_bg] Xcen_bg;  //matrix of alive locations
    //vector[Ncen] night_alive;       //matrix of alive night
    //vector[Nobs] night_dead;        //matrix of dead night 
        }

transformed data { 
   real<lower=0> tau_al; 
   tau_al = 10;
 }  

parameters { 
   real alpha_raw; 
   vector[M_bg] beta_bg;
   //real trialb;
   //real nightb;
   vector[K] tr; //trial intercepts
   real<lower=0> sigma_tr; //trial sd  
   vector[O] tp; //trial-pool intercepts
   real<lower=0> sigma_tp; //trial-pool sd  
   //vector[P] nt; //night intercepts
   //real<lower=0> sigma_nt; //night sd 
 } 
  
transformed parameters { 
    
  real alpha; // now vector size of factors
  vector[Nobs] yhatobs;
  vector[Ncen] yhatcen;

   alpha = exp(tau_al * alpha_raw);      
   
     yhatobs = exp(-(Xobs_bg * beta_bg  + tp[tp_dead]+ tr[trials_dead] )./(alpha));// nightb*night_dead+ nt[nt_dead]
     yhatcen = exp(-(Xcen_bg * beta_bg + tp[tp_alive]+ tr[trials_alive] )./(alpha));//nightb*night_alive+ nt[nt_alive]
}
  
model { 
   tr~normal(0,sigma_tr);
   sigma_tr~uniform(0,10);
   tp~normal(0,sigma_tp);
   sigma_tp~uniform(0,10);
   beta_bg ~ normal(-7,5); 
   alpha_raw ~ normal(0,5);  
   //nt~normal(0,sigma_nt);
   //sigma_nt~uniform(0,10);
   //trialb ~ normal(0,5);
   //nightb ~ normal(0,5);

   yobs ~ weibull(alpha, yhatobs);
   target += weibull_lccdf(ycen | alpha, yhatcen);
   }

  
generated quantities { 
     vector[Nobs + Ncen] log_lik;
    for (i in 1:Nobs) {
        log_lik[i] = weibull_lpdf(yobs[i] | alpha, yhatobs[i]);
        }
    for (i in 1:Ncen) {    
        log_lik[Nobs+i] = weibull_lccdf(ycen[i] | alpha, yhatcen[i]);
        }
}
