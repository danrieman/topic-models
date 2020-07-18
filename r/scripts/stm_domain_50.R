library(stm)

load("../localdata/test_train/pd_res_lwr200.RData")

corp = pd_res_lwr200

stm_domain_50 = stm(
    documents=corp$documents,
    vocab=corp$vocab,
    data=corp$meta[c("user_id", "is_fb")],
    content=~is_fb,
    init.type="LDA",
    K=50,
    seed=42,
    max.em.its=2000,
    verbose=TRUE,
    emtol=0.00001
)

save(stm_domain_50, file="../localdata/2020-07-05/stm_domain_50.RData")
