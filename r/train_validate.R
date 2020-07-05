library(dplyr)
library(stm)

# Load test/train assignments
load('localdata/train_uids.RData')
load('localdata/test_uids.RData')

# Load processed documents
load('localdata/tp_res.RData')
load('localdata/tp_res_tw.RData')
load('localdata/tp_res_fb.RData')


# Define training boolean vectors
istrain <- tp_res$meta$user_id %in% train_uids
istrain_tw <- tp_res_tw$meta$user_id %in% train_uids
istrain_fb <- tp_res_fb$meta$user_id %in% train_uids


# Vocab
vocab <- tp_res$vocab
vocab_tw <- tp_res_tw$vocab
vocab_fb <- tp_res_fb$vocab

# Training documents
docs <- tp_res$documents[istrain]
docs_tw <- tp_res_tw$documents[istrain_tw]
docs_fb <- tp_res_fb$documents[istrain_fb]

# Training metadata
meta <- lapply(tp_res$meta, function(x)x[istrain])
meta_tw <- lapply(tp_res_tw$meta, function(x)x[istrain_tw])
meta_fb <- lapply(tp_res_fb$meta, function(x)x[istrain_fb])

# Test documents
docs_t <- tp_res$documents[!istrain]
docs_t_tw <- tp_res_tw$documents[!istrain_tw]
docs_t_fb <- tp_res_fb$documents[!istrain_fb]

# Test metadata
meta_t <- lapply(tp_res$meta, function(x)x[!istrain])
meta_t_tw <- lapply(tp_res_tw$meta, function(x)x[!istrain_tw])
meta_t_fb <- lapply(tp_res_fb$meta, function(x)x[!istrain_fb])


# Prep training data
pd_res_lwr200 <- prepDocuments(docs, vocab, data.frame(meta), lower.thresh=200)
save(pd_res_lwr200, file='localdata/test_train/pd_res_lwr200.RData')
# Remaining: 

pd_res_lwr200_tw <- prepDocuments(docs_tw, vocab_tw, data.frame(meta_tw), lower.thresh=200)
save(pd_res_lwr200_tw, file='localdata/test_train/pd_res_lwr200_tw.RData')
# Remaining: 592929 documents, 3134 terms, 2610102 tokens

pd_res_lwr200_fb <- prepDocuments(docs_fb, vocab_fb, data.frame(meta_fb), lower.thresh=200)
save(pd_res_lwr200_fb, file='localdata/test_train/pd_res_lwr200_fb.RData')
# Remaining: 487337 documents, 3711 terms, 3271792 documents


# Prep test data
pd_res_lwr200_t <- prepDocuments(docs_t, vocab, data.frame(meta_t), lower.thresh=200)
save(pd_res_lwr200_t, file='localdata/test_train/pd_res_lwr200_t.RData')
# Remaining: 481321 documents, 3050 terms, 2490350 tokens

pd_res_lwr200_t_tw <- prepDocuments(docs_t_tw, vocab_tw, data.frame(meta_t_tw), lower.thresh=200)
save(pd_res_lwr200_t_tw, file='localdata/test_train/pd_res_lwr200_t_tw.RData')
# Remaining: 264024 documents, 1443 terms, 1001693 tokens

pd_res_lwr200_t_fb <- prepDocuments(docs_t_fb, vocab_fb, data.frame(meta_t_fb), lower.thresh=200)
save(pd_res_lwr200_t_fb, file='localdata/test_train/pd_res_lwr200_t_fb.RData')
# Remaining: 207038 documents, 1608 terms, 1126904 tokens


######################################
### Run K = 50 for tw, fb, and all ###
######################################

# Run on tw
# Finished after: Completing Iteration 31 (approx. per word bound = -6.903, relative change = 1.565e-04)
K <- 50
t0_tw <- Sys.time()
stm_k50_tw <- stm(
    documents=pd_res_lwr200_tw$documents, 
    vocab=pd_res_lwr200_tw$vocab,
    K=K,
    prevalence= ~ age + gender,
    data=data.frame(pd_res_lwr200_tw$meta),
    init.type='LDA',
    seed=42, 
    max.em.its=2000, # setting to 0 returns initialization
    verbose=TRUE,
    emtol=0.00001 # 0.00001 is default
)
t_tot_tw <- Sys.time() - t0_tw
cat('\n', t_tot_tw, '\n')
save(stm_k50_tw, file='localdata/stm_k50_tw.RData')

# Run on fb
# Finished after: Completing Iteration 128 (approx. per word bound = -7.319, relative change = 1.015e-05)
K <- 50
t0_fb <- Sys.time()
stm_k50_fb <- stm(
    documents=pd_res_lwr200_fb$documents, 
    vocab=pd_res_lwr200_fb$vocab,
    K=K,
    prevalence= ~ age + gender,
    data=pd_res_lwr200_fb$meta,
    init.type='LDA',
    seed=42, 
    max.em.its=2000, # setting to 0 returns initialization
    verbose=TRUE,
    emtol=0.00001 # 0.00001 is default
)
t_tot_fb <- Sys.time() - t0_fb
cat('\n', t_tot_fb, '\n')
save(stm_k50_fb, file='localdata/stm_k50_fb.RData')

# Run on all
K <- 50
t0 <- Sys.time()
stm_k50 <- stm(
    documents=pd_res_lwr200$documents, 
    vocab=pd_res_lwr200$vocab,
    K=K,
    prevalence= ~ age + gender + is_fb,
    content= ~ is_fb,
    data=pd_res_lwr200$meta, 
    init.type='LDA',
    seed=42, 
    max.em.its=2000, # setting to 0 returns initialization
    verbose=TRUE,
    emtol=0.00001 # 0.00001 is default
)
t_tot <- Sys.time() - t0
cat('\n', t_tot, '\n')
save(stm_k50, file='localdata/stm_k50.RData')


#######################################
### Run K = 100 for tw, fb, and all ###
#######################################

# Run on tw
load('localdata/test_train/pd_res_lwr200_tw.RData')
K <- 100
t0_tw <- Sys.time()
stm_k100_tw <- stm(
    documents=pd_res_lwr200_tw$documents, 
    vocab=pd_res_lwr200_tw$vocab,
    K=K,
    prevalence= ~ age + gender,
    data=data.frame(pd_res_lwr200_tw$meta),
    init.type='LDA',
    seed=42, 
    max.em.its=2000, # setting to 0 returns initialization
    verbose=TRUE,
    emtol=0.00001 # 0.00001 is default
)
t_tot_tw <- Sys.time() - t0_tw
cat('\n', t_tot_tw, '\n')
save(stm_k100_tw, file='localdata/stm_k100_tw.RData')

# Run on fb
load('localdata/test_train/pd_res_lwr200_fb.RData')
K <- 100
t0_fb <- Sys.time()
stm_k100_fb <- stm(
    documents=pd_res_lwr200_fb$documents, 
    vocab=pd_res_lwr200_fb$vocab,
    K=K,
    prevalence= ~ age + gender,
    data=pd_res_lwr200_fb$meta,
    init.type='LDA',
    seed=42, 
    max.em.its=2000, # setting to 0 returns initialization
    verbose=TRUE,
    emtol=0.00001 # 0.00001 is default
)
t_tot_fb <- Sys.time() - t0_fb
cat('\n', t_tot_fb, '\n')
save(stm_k100_fb, file='localdata/stm_k100_fb.RData')

# Run on all
load('localdata/test_train/pd_res_lwr200.RData')
K <- 100
t0 <- Sys.time()
stm_k100 <- stm(
    documents=pd_res_lwr200$documents, 
    vocab=pd_res_lwr200$vocab,
    K=K,
    prevalence= ~ age + gender + is_fb,
    content= ~ is_fb,
    data=pd_res_lwr200$meta, 
    init.type='LDA',
    seed=42, 
    max.em.its=2000, # setting to 0 returns initialization
    verbose=TRUE,
    emtol=0.00001 # 0.00001 is default
)
t_tot <- Sys.time() - t0
cat('\n', t_tot, '\n')
save(stm_k100, file='localdata/stm_k100.RData')


####################################
### Try running stm::alignCorpus ###
####################################

load('localdata/test_train/pd_res_lwr200_tw.RData')
align_corp_t_tw <- alignCorpus(
    new=list(
        documents=docs_t_tw,
        vocab=vocab_tw,
        meta=data.frame(meta_t_tw)
    ), 
    old.vocab=pd_res_lwr200_tw$vocab
)
# Removing 11547 Documents with No Words
# Your new corpus now has 269313 documents, 3035 non-zero terms of 3134 total terms in the original set.
# 1016817 terms from the new data did not match.
# This means the new data contained 96.8% of the old terms
# and the old data contained 0.3% of the unique terms in the new data.
# You have retained 1199003 tokens of the 2381519 tokens you started with (50.3%).

load('localdata/test_train/pd_res_lwr200_fb.RData')
align_corp_t_fb <- alignCorpus(
    new=list(
        documents=docs_t_fb,
        vocab=vocab_fb,
        meta=data.frame(meta_t_fb)
    ), 
    old.vocab=pd_res_lwr200_fb$vocab
)
# Removing 6988 Documents with No Words
# Your new corpus now has 212056 documents, 3702 non-zero terms of 3711 total terms in the original set.
# 559308 terms from the new data did not match.
# This means the new data contained 99.8% of the old terms
# and the old data contained 0.7% of the unique terms in the new data.
# You have retained 1487826 tokens of the 2438427 tokens you started with (61.0%).

load('localdata/test_train/pd_res_lwr200.RData')
align_corp_t <- alignCorpus(
    new=list(
        documents=docs_t,
        vocab=vocab,
        meta=data.frame(meta_t)
    ), 
    old.vocab=pd_res_lwr200$vocab
)
# Removing 12819 Documents with No Words
# Your new corpus now has 487085 documents, 6296 non-zero terms of 6393 total terms in the original set.
# 1390357 terms from the new data did not match.
# This means the new data contained 98.5% of the old terms
# and the old data contained 0.5% of the unique terms in the new data.
# You have retained 3014378 tokens of the 4819946 tokens you started with (62.5%).


###########################
### Load trained models ###
###########################

load('localdata/stm_k50_tw.RData')
load('localdata/stm_k100_tw.RData')

load('localdata/stm_k50_fb.RData')
load('localdata/stm_k100_fb.RData')

load('localdata/stm_k50.RData')
load('localdata/stm_k100.RData')


############################
### Test fitNewDocuments ###
############################

fnd_k50_tw_t <- fitNewDocuments(model=stm_k50_tw, documents=align_corp_t_tw$documents, 
                                newData=align_corp_t_tw$meta, returnPosterior=TRUE)

#############################
### Test optimizeDocument ###
#############################

calc.mu <- function(model, df, model_formula) {
    X <- model.matrix(model_formula, df)
    mu <- t(X %*% model$mu$gamma)
    return(mu)
}


library(stm); library(dplyr)
load('localdata/stm_k50_tw.RData')
load('localdata/test_train/pd_res_lwr200_tw.RData')
doc_idx <- 1
test_doc <- pd_res_lwr200_tw$documents[[doc_idx]]
test_mu <- calc.mu(stm_k50_tw, pd_res_lwr200_tw$meta[doc_idx,], ~ age + gender)

test_od <- optimizeDocument(
    document=test_doc, 
    eta=stm_k50_tw$eta, 
    mu=test_mu, 
    beta=exp(stm_k50_tw$beta[[1]][[1]]), 
    sigma=stm_k50_tw$sigma#, 
    # sigmainv=stm_k50_tw$invsigma
)

test_lnorm <- stm:::logisticnormalcpp(
    eta=stm_k50_tw$eta, 
    mu=test_mu, 
    siginv=stm_k50_tw$invsigma, 
    beta=exp(stm_k50_tw$beta[[1]][[1]]), 
    doc=test_doc, 
    sigmaentropy=NULL,
    method='BFGS', 
    control=list(maxit=5), # normally 500
    hpbcpp=FALSE
)



### testing

logitnormal.mvt.p <- function(x, mu, sigma, sigmainv) {
    # x, K by 1
    # mu, (k-1) by 1
    # sigma, (k-1) by (k-1)
    # sigmainv, (k-1) by (k-1)
    k <- nrow(x)
    Z <- det(2*pi*sigma)^.5 * prod(x)
    expo <- 0.5 * log(matrix(x[-m,1]/x[m],nrow=1)) %*% sigmainv %*% log(matrix(x[-m,1]/m[m], ncol=1))
    return(exp(expo)/Z)
}


