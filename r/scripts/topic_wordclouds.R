# Title     : TODO
# Objective : TODO
# Created by: rieman
# Created on: 2020-07-26
library(RColorBrewer)
library(wordcloud)
library(stm)


make.wordcloud <- function(model, topic) {
    set.seed(42)
    p <- cloud(model, topic=topic, random.order=FALSE, max.words=50, colors=brewer.pal(8, "Dark2"))
    return(p)
}

make.domain.wordcloud <- function(model, topic, domain) {
    set.seed(42)
    words <- model$vocab
    freq <- exp(model$beta$logbeta[[domain]])[topic,]
    p <- wordcloud(
        words=words,
        freq=freq,
        colors=brewer.pal(8, "Dark2"),
        max.words=100,
        random.order=FALSE
    )
    return(p)
}


make.tw.wordcloud <- function(model, topic) make.domain.wordcloud(model, topic, 1)


make.fb.wordcloud <- function(model, topic) make.domain.wordcloud(model, topic, 2)


stm.wordclouds <- function(model) {
    loc <- paste0(c("../localdata/wordclouds/stm/", model$settings$dim$K, "/Topic_"), collapse="")
    for(i in 1:model$settings$dim$K) {
        png(filename=paste0(c(loc, i, ".png"), collapse=""))
        make.wordcloud(model, i)
        dev.off()

        png(filename=paste0(c(loc, i, "_Twitter.png"), collapse=""))
        make.tw.wordcloud(model, i)
        dev.off()

        png(filename=paste0(c(loc, i, "_Facebook.png"), collapse=""))
        make.fb.wordcloud(model, i)
        dev.off()
    }
}


ctm.wordclouds <- function(model) {
    loc <- paste0(c("../localdata/wordclouds/ctm/", model$settings$dim$K, "/Topic_"), collapse="")
    for(i in 1:model$settings$dim$K) {
        png(filename=paste0(c(loc, i, ".png"), collapse=""))
        make.wordcloud(model, i)
        dev.off()
    }
}


load("../localdata/2020-07-05/stm_domain_10.RData")
stm.wordclouds(stm_domain_10)
rm(stm_domain_10)

load("../localdata/2020-07-05/stm_domain_25.RData")
stm.wordclouds(stm_domain_25)
rm(stm_domain_25)

load("../localdata/2020-07-05/stm_domain_50.RData")
stm.wordclouds(stm_domain_50)
rm(stm_domain_50)

load("../localdata/2020-07-05/stm_domain_100.RData")
stm.wordclouds(stm_domain_100)
rm(stm_domain_100)

load("../localdata/2020-07-05/stm_no_10.RData")
ctm.wordclouds(stm_no_10)
rm(stm_no_10)

load("../localdata/2020-07-05/stm_no_25.RData")
ctm.wordclouds(stm_no_25)
rm(stm_no_25)

load("../localdata/2020-07-05/stm_no_50.RData")
ctm.wordclouds(stm_no_50)
rm(stm_no_50)

load("../localdata/2020-07-05/stm_no_100.RData")
ctm.wordclouds(stm_no_100)
rm(stm_no_100)
