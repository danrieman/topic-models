library(dplyr)
library(stm)

load('localdata/tp_res.RData')

meta_df <- data.frame(tp_res$meta)

# keep only gender %in% c(1,2)
meta_df <- meta_df %>% filter(gender %in% c(1,2))

uni_uid_gen <- unique(meta_df[,c('user_id','gender')])

uid_gen1 <- sort(uni_uid_gen$user_id[uni_uid_gen$gender==1])
uid_gen2 <- sort(uni_uid_gen$user_id[uni_uid_gen$gender==2])

set.seed(42)
train_gen1 <- sort(sample(uid_gen1, round(0.7*length(uid_gen1), 0), replace=FALSE))
train_gen2 <- sort(sample(uid_gen2, round(0.7*length(uid_gen2), 0), replace=FALSE))

test_gen1 <- uid_gen1[!uid_gen1 %in% train_gen1]
test_gen2 <- uid_gen2[!uid_gen2 %in% train_gen2]

train_uids <- sort(c(train_gen1, train_gen2))
test_uids <- sort(c(test_gen1, test_gen2))

save(train_uids, file='localdata/train_uids.RData')
save(test_uids, file='localdata/test_uids.RData')
