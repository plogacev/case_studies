// Note: Tutorial on HMMs in Stan: https://luisdamiano.github.io/stancon18/hmm_stan_tutorial.pdf

functions {
  /**
   * Forward algorithm for a 2-state HMM
   * Computes the log marginal likelihood log p(y | params)
   *
   * States:
   *   state 1 = stockout  (emission: sales = 0 with prob 1)
   *   state 2 = regular   (emission: Negative Binomial)
   *
   * Arguments:
   *   sales_qty            - observed sales counts (length n_days)
   *   sales_mu, sales_phi  - Negative Binomial parameters for regular sales
   *   lp_transition        - 2x2 log transition matrix for hidden states
   *   lp_emission_stockout - vector[n_days], log p(sales_qty_t | stockout)
   *   lp_emission_regular  - vector[n_days], log p(sales_qty_t | regular)
   *
   * Returns:
   *   log marginal likelihood of the sequence under the HMM
   */
  vector hmm_marginal_loglik_path(vector log_alpha_prev, matrix lp_transition, array[] real lp_emission_t)
  {
        int STATE_STOCKOUT = 1;
        int STATE_REGULAR = 2;
    
        vector[2] state_t;
        vector[2] log_alpha_next;

        for (state_next in 1:2) {
            for (state_prev in 1:2) {
                state_t[state_prev] = log_alpha_prev[state_prev] + lp_transition[state_prev, state_next]; // state_prev → state_next
            }

            // add emission likelihood
            log_alpha_next[state_next] = log_sum_exp(state_t) + lp_emission_t[state_next];
        }

        return log_alpha_next;
  }
  
  // prior for initial hidden state; to-do: can probably be derived, at least approximately, from other parameters 
  vector init_delta() {
        vector[2] log_prior_init = [ log(0.05), log(0.95) ]';
        return log_prior_init;
  }
  
  real hmm_marginal_loglik(
        array[] int sales_qty,
        real sales_mu,
        real sales_phi,
        matrix lp_transition,
        vector lp_emission_stockout,
        vector lp_emission_regular
  ) {
    int n_days = size(sales_qty);         // total time points

    int STATE_STOCKOUT = 1;
    int STATE_REGULAR = 2;
  
    vector[2] log_prior_init = init_delta();

    // Initialization: init both states with initial probabilities
    vector[2] log_alpha;
    log_alpha[STATE_STOCKOUT] = log_prior_init[STATE_STOCKOUT] + lp_emission_stockout[1];
    log_alpha[STATE_REGULAR]  =  log_prior_init[STATE_REGULAR] + lp_emission_regular[1];

    // Forward pass: for each time step, propagate alpha forward
    for (t in 2:n_days)
    {
          // get log-forward probs for time t from those for t-1
          log_alpha = hmm_marginal_loglik_path(log_alpha, lp_transition, {lp_emission_stockout[t], lp_emission_regular[t]} );
    }

    // return sum over both possible final states
    return log_sum_exp(log_alpha);
  }
}

data {
  int<lower=1> n_days;        // Number of days
  array[n_days] int<lower=0> sales_qty;

}

parameters {
  real<lower=0> sales_mu;      // Mean of Negative Binomial for regular sales
  real<lower=0> sales_phi;     // Dispersion (shape) of Negative Binomial

  real<lower=0, upper=1> p_SS; // P(stockout → stockout): persistence of stockouts
  real<lower=0, upper=1> p_RR; // P(regular → regular): persistence of regular sales
}

transformed parameters {
  // ----- transition matrix -----
  matrix[2,2] lp_transition; // rows: previous state; columns: current state
  lp_transition[1,1] = log(p_SS);     // [1,1] = P(stockout → stockout)
  lp_transition[1,2] = log(1 - p_SS); // [1,2] = P(stockout → regular)
  lp_transition[2,1] = log(1 - p_RR); // [2,1] = P(regular → stockout)
  lp_transition[2,2] = log(p_RR);     // [2,2] = P(regular → regular)

  // ----- emission log-likelihoods -----
  vector[n_days] lp_emission_stockout;
  vector[n_days] lp_emission_regular;

  for (t in 1:n_days) {
    // For stockout state: if sales > 0, probability = 0 → log prob = -Inf
    lp_emission_stockout[t] = (sales_qty[t] == 0) ? 0 : negative_infinity();

    // For regular state: use NB density
    lp_emission_regular[t] = neg_binomial_2_lpmf(sales_qty[t] | sales_mu, sales_phi);
  }
}

model {
  // Priors: regular sales
  sales_mu ~ normal(4, 2);  // negative binomial mean: should be around typical sales if no stockout
  sales_phi ~ gamma(2, 0.1); // negative binomial dispersion: broad gamma prior for flexibility

  // Priors: state persistence 
  p_SS ~ beta(5, 1);      // stockout self-transition: moderately sticky
  p_RR ~ beta(10, 1);     // regular self-transition: strongly sticky

  // Likelihood: Marginalize hidden states using the forward algorithm
  target += hmm_marginal_loglik(
    sales_qty,
    sales_mu,
    sales_phi,
    lp_transition,
    lp_emission_stockout,
    lp_emission_regular
  );
}

generated quantities {
  array[n_days] int viterbi_path;
  {
    // State indexing
    int STATE_STOCKOUT = 1;
    int STATE_REGULAR  = 2;

    // Backpointers and log-probs
    array[n_days, 2] int back_ptr;
    array[n_days, 2] real log_delta;

    // Initial state log-probs
    log_delta[1, STATE_STOCKOUT] = log(0.05) + lp_emission_stockout[1];
    log_delta[1, STATE_REGULAR]  = log(0.95) + lp_emission_regular[1];

    // Viterbi forward pass
    for (t in 2:n_days) {
        for (k in 1:2) {
            real best_logp = negative_infinity();
            int best_prev = -1;
            for (j in 1:2) {
                real logp = log_delta[t - 1, j] + lp_transition[j, k];
                if (logp > best_logp) {
                    best_logp = logp;
                    best_prev = j;
                }
            }
            log_delta[t, k] = best_logp + (k == STATE_STOCKOUT ? lp_emission_stockout[t] : lp_emission_regular[t]);
            back_ptr[t, k] = best_prev;
        }
    }

    // Backtrace
    {
      int best_final_state = (log_delta[n_days, STATE_STOCKOUT] > log_delta[n_days, STATE_REGULAR])
                             ? STATE_STOCKOUT : STATE_REGULAR;
      viterbi_path[n_days] = best_final_state;

      for (t in 1:(n_days - 1)) {
        int next_t = n_days - t + 1;
        int cur_t = n_days - t;
        viterbi_path[cur_t] = back_ptr[next_t, viterbi_path[next_t]];
      }
    }
  }
}


