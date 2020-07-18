# Title     : TODO
# Objective : TODO
# Created by: rieman
# Created on: 2020-07-12

library(stm)
library(dplyr)

load("../localdata/test_train/pd_res_lwr200.RData")
load("../localdata/test_train/align_tp_res_t.RData")
load("../localdata/2020-07-05/stm_no_25.RData")

corp = pd_res_lwr200
model <- stm_no_25

rm(stm_no_25)
rm(pd_res_lwr200)

feats <- model$theta %>%
    data.frame %>%
    (function(x) {colnames(x) <- paste0("Topic_", c(1:ncol(x))); return(x)}) %>%
    cbind(corp$meta)

theta_t <- fitNewDocuments(
    model=model,
    documents=align_tp_res_t$documents,
    newData=align_tp_res_t$meta
)

feats_t <- theta_t$theta %>%
    data.frame %>%
    (function(x) {colnames(x) <- paste0("Topic_", c(1:ncol(x))); return(x)}) %>%
    cbind(align_tp_res_t$meta)

write.csv(feats, file="../localdata/features/training_no_25.csv", row.names=FALSE)
write.csv(feats_t, file="../localdata/features/test_no_25.csv", row.names=FALSE)
