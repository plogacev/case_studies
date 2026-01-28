// TMB Local Level Model with Fourier Seasonality
// State-space model for time series with exogenous price effect
//
// State equation:   alpha_{t+1} = alpha_t + eta_t,  eta_t ~ N(0, sigma_eta^2)
// Observation:      y_t = alpha_t + beta*x_t + seasonal_t + eps_t,  eps_t ~ N(0, sigma_eps^2)
// Seasonality:      seasonal_t = sum_{k=1}^{K} [gamma_k*sin(2*pi*k*t/period) + delta_k*cos(2*pi*k*t/period)]

#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(y);           // log quantity (response)
  DATA_VECTOR(x);           // log price (exogenous)
  DATA_IVECTOR(t_idx);      // time index for Fourier terms (0-indexed)
  DATA_INTEGER(n_harmonics); // number of Fourier harmonics
  DATA_SCALAR(period);      // seasonal period (7 for weekly)
  DATA_IVECTOR(obs_idx);    // indices of non-NA observations

  // Parameters
  PARAMETER(log_sigma_eta);    // log SD of state innovation
  PARAMETER(log_sigma_eps);    // log SD of observation error
  PARAMETER(beta);             // price elasticity coefficient
  PARAMETER(alpha0);           // initial state
  PARAMETER_VECTOR(gamma);     // Fourier sine coefficients
  PARAMETER_VECTOR(delta);     // Fourier cosine coefficients

  // Transform parameters
  Type sigma_eta = exp(log_sigma_eta);
  Type sigma_eps = exp(log_sigma_eps);

  int n = y.size();
  int n_obs = obs_idx.size();

  Type nll = Type(0.0);
  Type pi = Type(3.14159265358979323846);

  // State vector (random effects could be used, but we use Kalman filter approach)
  vector<Type> alpha(n);
  alpha(0) = alpha0;

  // State innovations - penalize large deviations
  for(int t = 1; t < n; t++) {
    // For efficiency, we integrate out states using Kalman filter logic
    // Here we use a simpler approach: states as implicit random effects
    alpha(t) = alpha(0); // Will be updated via filtering
  }

  // Kalman filter for local level model
  Type a = alpha0;  // filtered state mean
  Type P = Type(1000.0);  // filtered state variance (diffuse initialization)

  int obs_ptr = 0;

  for(int t = 0; t < n; t++) {
    // Prediction step
    Type a_pred = a;
    Type P_pred = P + sigma_eta * sigma_eta;

    // Compute seasonal component
    Type seasonal = Type(0.0);
    for(int k = 0; k < n_harmonics; k++) {
      Type freq = Type(2.0) * pi * Type(k + 1) * Type(t_idx(t)) / period;
      seasonal += gamma(k) * sin(freq) + delta(k) * cos(freq);
    }

    // Check if this time point has an observation
    if(obs_ptr < n_obs && obs_idx(obs_ptr) == t) {
      // Observation equation: y_t = alpha_t + beta*x_t + seasonal_t + eps_t
      Type y_pred = a_pred + beta * x(t) + seasonal;
      Type F = P_pred + sigma_eps * sigma_eps;  // prediction variance
      Type v = y(t) - y_pred;  // prediction error

      // Update negative log-likelihood
      nll += Type(0.5) * (log(F) + v * v / F);

      // Update step
      Type K = P_pred / F;  // Kalman gain
      a = a_pred + K * v;
      P = P_pred * (Type(1.0) - K);

      obs_ptr++;
    } else {
      // No observation at this time point, just propagate
      a = a_pred;
      P = P_pred;
    }
  }

  // Report quantities
  ADREPORT(beta);
  ADREPORT(sigma_eta);
  ADREPORT(sigma_eps);

  return nll;
}
