// generated with brms 2.19.0
data {
  int<lower=1> N;  // total number of observations
  vector[N] Y;  // response variable
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
  // data for the CAR structure
  int<lower=1> Nloc;
  int<lower=1> Jloc[N];
  int<lower=0> Nedges;
  int<lower=1> edges1[Nedges];
  int<lower=1> edges2[Nedges];
  // data for group-level effects of ID 1
  int<lower=1> N_1;  // number of grouping levels
  int<lower=1> M_1;  // number of coefficients per level
  int<lower=1> J_1[N];  // grouping indicator per observation
  // group-level predictor values
  vector[N] Z_1_1;
  vector[N] Z_1_2;
  vector[N] Z_1_3;
  vector[N] Z_1_4;
  // data for group-level effects of ID 2
  int<lower=1> N_2;  // number of grouping levels
  int<lower=1> M_2;  // number of coefficients per level
  int<lower=1> J_2[N];  // grouping indicator per observation
  // group-level predictor values
  vector[N] Z_2_1;
  vector[N] Z_2_2;
  vector[N] Z_2_3;
  vector[N] Z_2_4;
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
  int Kc = K - 1;
  matrix[N, Kc] Xc;  // centered version of X without an intercept
  vector[Kc] means_X;  // column means of X before centering
  for (i in 2:K) {
    means_X[i - 1] = mean(X[, i]);
    Xc[, i - 1] = X[, i] - means_X[i - 1];
  }
}
parameters {
  vector[Kc] b;  // population-level effects
  real Intercept;  // temporary intercept for centered predictors
  vector<lower=0> [N_2] sdcar;  // SD of the CAR structure
  // parameters for the ICAR structure
  vector[N] zcar;
  real<lower=0> sigma;  // dispersion parameter
  vector<lower=0>[M_1] sd_1;  // group-level standard deviations
  vector[N_1] z_1[M_1];  // standardized group-level effects
  vector<lower=0>[M_2] sd_2;  // group-level standard deviations
  vector[N_2] z_2[M_2];  // standardized group-level effects
}
transformed parameters {
  // scaled parameters for the ICAR structure
  vector[N] rcar;
  vector[N_1] r_1_1;  // actual group-level effects
  vector[N_1] r_1_2;  // actual group-level effects
  vector[N_1] r_1_3;  // actual group-level effects
  vector[N_1] r_1_4;  // actual group-level effects
  vector[N_2] r_2_1;  // actual group-level effects
  vector[N_2] r_2_2;  // actual group-level effects
  vector[N_2] r_2_3;  // actual group-level effects
  vector[N_2] r_2_4;  // actual group-level effects
  real lprior = 0;  // prior contributions to the log posterior
  // compute scaled parameters for the ICAR structure
  for (n in 1:N) {
    rcar[n] = zcar[n] .* sdcar[J_2[n]];
  }
  r_1_1 = (sd_1[1] * (z_1[1]));
  r_1_2 = (sd_1[2] * (z_1[2]));
  r_1_3 = (sd_1[3] * (z_1[3]));
  r_1_4 = (sd_1[4] * (z_1[4]));
  r_2_1 = (sd_2[1] * (z_2[1]));
  r_2_2 = (sd_2[2] * (z_2[2]));
  r_2_3 = (sd_2[3] * (z_2[3]));
  r_2_4 = (sd_2[4] * (z_2[4]));
  lprior += student_t_lpdf(Intercept | 3, 4.7, 3.1);
  lprior += student_t_lpdf(sdcar | 3, 0, 3.1)
    - 1 * student_t_lccdf(0 | 3, 0, 3.1);
  lprior += student_t_lpdf(sigma | 3, 0, 3.1)
    - 1 * student_t_lccdf(0 | 3, 0, 3.1);
  lprior += student_t_lpdf(sd_1 | 3, 0, 3.1)
    - 4 * student_t_lccdf(0 | 3, 0, 3.1);
  lprior += student_t_lpdf(sd_2 | 3, 0, 3.1)
    - 4 * student_t_lccdf(0 | 3, 0, 3.1);
}
model {
  // likelihood including constants
  if (!prior_only) {
    // initialize linear predictor term
    vector[N] mu = rep_vector(0.0, N);
    mu += Intercept + Xc * b;
    for (n in 1:N) {
      // add more terms to the linear predictor
      mu[n] += rcar[n] + r_1_1[J_1[n]] * Z_1_1[n] + r_1_2[J_1[n]] * Z_1_2[n] + r_1_3[J_1[n]] * Z_1_3[n] + r_1_4[J_1[n]] * Z_1_4[n] + r_2_1[J_2[n]] * Z_2_1[n] + r_2_2[J_2[n]] * Z_2_2[n] + r_2_3[J_2[n]] * Z_2_3[n] + r_2_4[J_2[n]] * Z_2_4[n];
    }
    target += normal_lpdf(Y | mu, sigma);
  }
  // priors including constants
  target += lprior;
  // improper prior on the spatial CAR component
  target += -0.5 * dot_self(zcar[edges1] - zcar[edges2]);
  // soft sum-to-zero constraint
  target += normal_lpdf(sum(zcar) | 0, 0.001 * Nloc);
  target += std_normal_lpdf(z_1[1]);
  target += std_normal_lpdf(z_1[2]);
  target += std_normal_lpdf(z_1[3]);
  target += std_normal_lpdf(z_1[4]);
  target += std_normal_lpdf(z_2[1]);
  target += std_normal_lpdf(z_2[2]);
  target += std_normal_lpdf(z_2[3]);
  target += std_normal_lpdf(z_2[4]);
}
generated quantities {
  // actual population-level intercept
  real b_Intercept = Intercept - dot_product(means_X, b);
  
  // Log likelihood for observed data
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(Y[n] | b_Intercept + Xc[n] * b + rcar[n] +
                                      r_1_1[J_1[n]] * Z_1_1[n] +
                                      r_1_2[J_1[n]] * Z_1_2[n] +
                                      r_1_3[J_1[n]] * Z_1_3[n] +
                                      r_1_4[J_1[n]] * Z_1_4[n] +
                                      r_2_1[J_2[n]] * Z_2_1[n] +
                                      r_2_2[J_2[n]] * Z_2_2[n] +
                                      r_2_3[J_2[n]] * Z_2_3[n] +
                                      r_2_4[J_2[n]] * Z_2_4[n],
                                      sigma);
  }
}
