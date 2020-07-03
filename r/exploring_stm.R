#!/usr/bin/env R
library(DBI)
library(RMySQL)
library(dplyr)
library(stm)


#####################
### Set Constants ###
#####################

# # Database
# DEF_fb_db <- 'empathy_qualtrics'
# DEF_twt_db <- 'empathy_qualtrics'

# # Messages
# DEF_msgs_tbl <- list(
#     twt='statuses_tw',
#     fb='statuses_fin'
# )
# DEF_msgs_colname <- list(
#     twt='message',
#     fb='message'
# )
# DEF_msgs_key <- list(
#     twt='user_id', # copy of fb id
#     fb='user_id'
# )

# # Labels
# DEF_lab_tbl <- list(
#     twt='',
#     fb=''
# )
# DEF_lab_tgts <- list(
#     twt=list(gender='gender', age='age'),
#     fb=list(gender='gender', age='age')
# )
# DEF_lab_key <- list(
#     twt='user_id',
#     fb='user_id'
# )


###########################
### Connect to Database ###
###########################

db_eng <- dbConnect(MySQL())


########################
### Define functions ###
########################

# get.msgs <- function(conn, db, tbl, where='', key_col='user_id', msg_col='message') {
#     ### This function selects all (key_col, message) records from db.tbl
#     # Define basic select
#     sql <- paste0('SELECT ', key_col, ' AS id, ', msg_col, ' AS message FROM ', db, '.', tbl)
#     # Add on optional where
#     if(nchar(where) > 0) sql <- paste0(sql, ' WHERE ', where)
#     # Get then return records
#     recs <- dbGetQuery(conn, sql)
#     return(recs)
# }

# get.tgts <- function(conn, db, tbl, where='', key_col='user_id', targets=list()) {
#     ### This function selects all (key_col, target_1, ..., target_n) records from db.tbl
#     select_args <- ''
#     for(i in names(targets)) {
#         tmp <- paste0(targets[[i]], ' AS ', i)
#         if(nchar(select_args)==0) {
#             select_args <- tmp
#         } else {
#             select_args <- paste0(select_args, ', ', tmp)
#         }
#     }
#     if(nchar(select_args) > 0) select_args <- paste0(', ', select_args)
#     sql <- paste0('SELECT ', key_col, ' AS id', select_args, ' FROM ', db, '.', tbl)
#     if(nchar(where)>0) sql <- paste0(sql, ' WHERE ', where)
#     recs <- dbGetQuery(conn, sql)
#     return(recs)
# }




statuses_fb <- dbGetQuery(db_eng, 'SELECT * FROM empathy_qualtrics.statuses_fin')
statuses_tw <- dbGetQuery(db_eng, 'SELECT * FROM empathy_qualtrics.statuses_tw')

survey <- dbGetQuery(db_eng, 'SELECT * FROM empathy_qualtrics.survey_combined_final')

uni_uid_tw <- sort(unique(statuses_tw$user_id))
uni_uid_fb <- sort(unique(statuses_fb$user_id))
uni_uid_intersection <- uni_uid_tw[uni_uid_tw %in% uni_uid_fb]

msgs_tw <- statuses_tw %>% filter(user_id %in% uni_uid_intersection)
msgs_fb <- statuses_fb %>% filter(user_id %in% uni_uid_intersection)


set.seed(42)
sample_idx_tw <- sample.int(nrow(msgs_tw), 200, replace=FALSE)
sample_idx_fb <- sample.int(nrow(msgs_fb), 200, replace=FALSE)

samp_tp_tw <- textProcessor(msgs_tw$message[sample_idx_tw], metadata=msgs_tw[sample_idx_tw,])
samp_tp_fb <- textProcessor(msgs_fb$message[sample_idx_fb], metadata=msgs_fb[sample_idx_fb,])

dat_docs <- c(msgs_tw$message, msgs_fb$message)
dat_meta <- data.frame(
                user_id=c(msgs_tw$user_id, msgs_fb$user_id),
                is_fb=c(rep(0, nrow(msgs_tw)), rep(1, nrow(msgs_fb)))) %>% 
            left_join(survey %>% select(user_id, age=age_fixed, gender), by='user_id')

missing_meta <- apply(dat_meta, 1, function(x)sum(is.na(x))) > 0
dat_docs <- dat_docs[!missing_meta]
dat_meta <- dat_meta[!missing_meta,]

mallet_stopwords <- readLines('/home/rieman/mallet/mallet-2.0.8RC3/stoplists/en.txt')

tp_res_tw <- textProcessor(dat_docs[dat_meta$is_fb==0], 
                           metadata=dat_meta %>% filter(is_fb==0) %>% as.data.frame, 
                           customstopwords=mallet_stopwords, lowercase=TRUE, removenumbers=FALSE, 
                           removepunctuation=FALSE, stem=FALSE, wordLengths=c(2,Inf), striphtml=TRUE, 
                           removestopwords=FALSE)
tp_res_fb <- textProcessor(dat_docs[dat_meta$is_fb==1], 
                           metadata=dat_meta %>% filter(is_fb==1) %>% as.data.frame,
                           customstopwords=mallet_stopwords, lowercase=TRUE, removenumbers=FALSE, 
                           removepunctuation=FALSE, stem=FALSE, wordLengths=c(2,Inf), striphtml=TRUE, 
                           removestopwords=FALSE)
tp_res <- textProcessor(dat_docs, metadata=dat_meta, customstopwords=mallet_stopwords, 
                        lowercase=TRUE, removenumbers=FALSE, removepunctuation=FALSE, 
                        stem=FALSE, wordLengths=c(2,Inf), striphtml=TRUE, removestopwords=FALSE)

# save(tp_res_tw, file='/sandata/rieman/stm/data/tp_res_tw.RData')
# save(tp_res_fb, file='/sandata/rieman/stm/data/tp_res_fb.RData')
# save(tp_res, file='/sandata/rieman/stm/data/tp_res.RData')

# load('/sandata/rieman/stm/data/tp_res_tw.RData')
# load('/sandata/rieman/stm/data/tp_res_fb.RData')
# load('/sandata/rieman/stm/data/tp_res.RData')

pd_res_lwr200_tw <- prepDocuments(tp_res_tw$documents, tp_res_tw$vocab, tp_res_tw$meta, lower.thresh=200)
# left with 4466 words
pd_res_lwr200_fb <- prepDocuments(tp_res_fb$documents, tp_res_fb$vocab, tp_res_fb$meta, lower.thresh=200)
# left with 5045 words
pd_res_lwr200 <- prepDocuments(tp_res$documents, tp_res$vocab, tp_res$meta, lower.thresh=200)
# left with 8794 words

save(pd_res_lwr200_tw, file='/sandata/rieman/stm/data/pd_res_lwr200_tw.RData')
save(pd_res_lwr200_fb, file='/sandata/rieman/stm/data/pd_res_lwr200_fb.RData')
save(pd_res_lwr200, file='/sandata/rieman/stm/data/pd_res_lwr200.RData')


docs_tw <- pd_res_lwr200_tw$documents
vocab_tw <- pd_res_lwr200_tw$vocab
meta_tw <- pd_res_lwr200_tw$meta

docs_fb <- pd_res_lwr200_fb$documents
vocab_fb <- pd_res_lwr200_fb$vocab
meta_fb <- pd_res_lwr200_fb$meta

docs <- pd_res_lwr200$documents
vocab <- pd_res_lwr200$vocab
meta <- pd_res_lwr200$meta


##############
### Sample ###
##############

# Run a sample to see what objects are returned in the output
set.seed(42)
samp_idx <- sample.int(length(docs), 10000, replace=FALSE)
samp_docs <- docs[samp_idx]
samp_meta <- meta[samp_idx]

names(samp_docs) <- as.character(c(1:length(samp_docs)))

K <- 50
t0 <- Sys.time()
samp_stm_k50 <- stm(
    documents=samp_docs, 
    vocab=vocab,
    prevalence=as.matrix(data.frame(age=samp_meta$age, gender=samp_meta$gender)),
    content=as.matrix(data.frame(age=samp_meta$age, gender=samp_meta$gender)),
    init.type='LDA',
    seed=42, 
    max.em.its=2000, # setting to 0 returns initialization
    verbose=TRUE,
    emtol=0.00001 # 0.00001 is default
)
t_tot <- Sys.time() - t0
cat('\n', t_tot, '\n')


######################################
### Run K = 50 for tw, fb, and all ###
######################################

# Run on tw
# Finished after: Completing Iteration 68 (approx. per word bound = -7.161, relative change = 1.301e-05)
K <- 50
t0_tw <- Sys.time()
stm_k50_tw <- stm(
    documents=docs_tw, 
    vocab=vocab_tw,
    K=K,
    prevalence= ~ age + I(as.integer(gender==2)),
    # content= ~ age + I(as.integer(gender==2)), # can only contain one variable - saving for is_fb
    data=meta_tw,
    init.type='LDA',
    seed=42, 
    max.em.its=2000, # setting to 0 returns initialization
    verbose=TRUE,
    emtol=0.00001 # 0.00001 is default
)
t_tot_tw <- Sys.time() - t0_tw
cat('\n', t_tot_tw, '\n')
save(stm_k50_tw, file='/sandata/rieman/stm/data/stm_k50_tw_alldata.RData')

# Run on fb
# Finished after: Completing Iteration 112 (approx. per word bound = -7.517, relative change = 1.099e-05)
K <- 50
t0_fb <- Sys.time()
stm_k50_fb <- stm(
    documents=docs_fb, 
    vocab=vocab_fb,
    K=K,
    prevalence= ~ age + I(as.integer(gender==2)),
    # content= ~ age + I(as.integer(gender==2)), # can only contain one variable - saving for is_fb
    data=meta_fb,
    init.type='LDA',
    seed=42, 
    max.em.its=2000, # setting to 0 returns initialization
    verbose=TRUE,
    emtol=0.00001 # 0.00001 is default
)
t_tot_fb <- Sys.time() - t0_fb
cat('\n', t_tot_fb, '\n')
save(stm_k50_fb, file='/sandata/rieman/stm/data/stm_k50_fb_alldata.RData')

# Run on all
# Finished after: Completing Iteration 69 (approx. per word bound = -7.769, relative change = 1.098e-05)
K <- 50
t0 <- Sys.time()
stm_k50 <- stm(
    documents=docs, 
    vocab=vocab,
    K=K,
    prevalence= ~ age + I(as.integer(gender==2)) + is_fb,
    content= ~ is_fb,
    data=meta, 
    init.type='LDA',
    seed=42, 
    max.em.its=2000, # setting to 0 returns initialization
    verbose=TRUE,
    emtol=0.00001 # 0.00001 is default
)
t_tot <- Sys.time() - t0
cat('\n', t_tot, '\n')
save(stm_k50, file='/sandata/rieman/stm/data/stm_k50_alldata.RData')


#################################
### Analyze Resulting objects ###
#################################


##########################################
### Redo analysis with test/train sets ###
##########################################

# see train_validate.R
