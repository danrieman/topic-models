data {
    int<lower=2> K; // num topics
    int<lower=2> V; // num words
    int<lower=1> M; // num docs
    int<lower=1> N; // total word instances
    int<lower=1,upper=V> w[N];      // word n
    int<lower=1,upper=M> doc[N];    // doc ID for word n
    vector<lower=0>[V] beta;        // word prior
    vector[K] mu;           // topic mean - replaces alpha from lda
    cov_matrix[K] Sigma;    // topic covariance - replaces alpha from lda     
} 
parameters {
    simplex[V] phi[K];  // word dist for topic k 
    vector[K] eta[M];   // topic dist for doc m
} 
transformed parameters {
    simplex[K] theta[M];
    for (m in 1:M) 
        theta[m] = softmax(eta[m]);
}
model {
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

