data {
    int<lower=2> K; // num topics
    int<lower=2> V; // num words
    int<lower=1> M; // num docs
    int<lower=1> N; // total word instances
    int<lower=1,upper=V> w[N];      // word n
    int<lower=1,upper=M> doc[N];    // doc ID for word n
    vector<lower=0>[V] beta;        // word prior
    int<lower=0,upper=1> class[M];  // class, 0-1, for each document
} 
parameters {
    vector[K] mu;               // topic mean
    corr_matrix[K] Omega;       // correlation matrix
    vector<lower=0>[K] sigma;   // scales
    vector[K] eta[M];           // logit topic dist for topic k
    simplex[V] phi[K];          // word dist for topic k
    
} 
transformed parameters {
    simplex[K] theta[M];
    for (m in 1:M) 
        theta[m] = softmax(eta[m]);
    cov_matrix[K] Sigma;            // covariance matrix
    for (m in 1:K)
        Sigma[m,m] = sigma[m] * sigma[m] * Omega[m,m];
    for (m in 1:(K-1)) {
        for (n in 1:(K-1)) {
            Sigma[m, n] = sigma[m] * sigma[n] * Omega[m, n];
            Sigma[n, m] = Sigma[m, n]
        }
    }
}
model {
    mu ~ normal(0, 5);      // vectorized, diffuse
    Omega ~ lkj_corr(2.0);  // regularize to unit correlation
    sigma ~ cauchy(0, 5);   // half-Cauchy due to constraint
    
    for (m in 1:M)
        eta[m] ~ multi_normal(mu, Sigma);
    for (k in 1:K)
        phi[k] ~ dirichlet(beta); // prior
    for (n in 1:N) {
        real gamma[K];
        for (k in 1:K)
            gamma[k] = log(theta[doc[n], k]) + log(phi[k, w[n]]);
        target += log_sum_exp(gamma); // likelihood;
    }
}

