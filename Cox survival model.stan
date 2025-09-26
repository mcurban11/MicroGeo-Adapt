data {
    int<lower=1> N;               //individuals
    int<lower=1> J;               //number of locations
    int<lower=1> K;               //number of Trials
    int<lower=1> L;               //number of Pools
    int<lower=1> O;               //number of Trial-Pools
    int<lower=1> N_dead;          //number of Dead - uncensored
    int<lower=1> N_alive;         //number of UnDead - censored
    int<lower=0> N_covs;          //number of covariates = classes - 1
    //int trials_alive[N_alive]; //vector of alive trials
    //int trials_dead[N_dead]; //vector of dead trials
    int tp_alive[N_alive]; //vector of alive trials
    int tp_dead[N_dead]; //vector of dead trials
    
    matrix[N_alive,J] X_alive;       //matrix of alive locations
    matrix[N_dead,J] X_dead;        //matrix of dead locations
    //vector[N_alive] night_alive;       //matrix of alive night
    //vector[N_dead] night_dead;        //matrix of dead night
    vector<lower=0>[N_alive] times_alive; //vector of alive event times
    vector<lower=0>[N_dead] times_dead;   //vector of dead event times
    }
  
parameters {
      vector[J] betas;      
      //real nightb;
      real intercept;
      //vector[K] tr; //trial intercepts
      //real<lower=0> sigma_tr; //trial sd       
      vector[O] tp; //trial intercepts
      real<lower=0> sigma_tp; //trial sd       
    }
  
model {
    betas ~ normal(0,2); 
    //nightb ~ normal(0,2);
    intercept ~ normal(-4,2);  
    //tr~normal(0,sigma_tr);
    //sigma_tr~uniform(0,10);
    tp~normal(0,sigma_tp);
    sigma_tp~uniform(0,10);
    
    target += exponential_lpdf(times_dead | exp(intercept+tp[tp_dead]+X_dead*betas )); //+night_dead*nightb+tr[trials_dead]
    target += exponential_lccdf(times_alive | exp(intercept+tp[tp_alive]+X_alive*betas ));  //+ night_alive*nightb+tr[trials_alive]
    }
    
generated quantities {
    //matrix[N_dead,N_covs] times_dead_sampled;
    real yhat[N_dead+N_alive];
    real log_lik[N_dead+N_alive];
      for(i in 1:N_dead) {
                            yhat[i] = exponential_rng(exp(intercept+tp[tp_dead[i]]+X_dead[i]*betas ));//+ night_dead[i]*nightb+tr[trials_dead[i]]
        log_lik[i] = exponential_lpdf(times_dead[i] | exp(intercept+tp[tp_dead[i]]+X_dead[i]*betas ));//+ night_dead[i]*nightb+tr[trials_dead[i]]
        }
        for(i in 1:N_alive) {
                              yhat[N_dead + i] = exponential_rng(exp(intercept+tp[tp_alive[i]]+X_alive[i]*betas ));//+ night_alive[i]*nightb+tr[trials_alive[i]]
        log_lik[N_dead + i] = exponential_lccdf(times_alive[i] | exp(intercept+tp[tp_alive[i]]+X_alive[i]*betas ));// + night_alive[i]*nightb+tr[trials_alive[i]]
        }
    }
