# Title     : TODO
# Objective : TODO
# Created by: rieman
# Created on: 2020-07-12

library(stm)
library(dplyr)

load("../localdata/test_train/pd_res_lwr200.RData")
load("../localdata/test_train/align_tp_res_t.RData")
load("../localdata/2020-07-05/stm_domain_100.RData")

corp = pd_res_lwr200
model <- stm_domain_100

rm(stm_domain_100)
rm(pd_res_lwr200)

feats <- model$theta %>%
    data.frame %>%
    (function(x) {colnames(x) <- paste0("Topic_", c(1:ncol(x))); return(x)}) %>%
    cbind(corp$meta)

theta_t <- fitNewDocuments(
    model=model,
    documents=align_tp_res_t$documents,
    newData=align_tp_res_t$meta,
    contentPrior="Covariate",
    betaIndex=model.matrix(~is_fb, align_tp_res_t$meta)
)

feats_t <- theta_t$theta %>%
    data.frame %>%
    (function(x) {colnames(x) <- paste0("Topic_", c(1:ncol(x))); return(x)}) %>%
    cbind(align_tp_res_t$meta)

write.csv(feats, file="../localdata/features/training_domain_100.csv", row.names=FALSE)
write.csv(feats_t, file="../localdata/features/test_domain_100.csv", row.names=FALSE)
