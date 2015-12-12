rm(list = ls())

library('ggplot2')
library('reshape2')
library('plyr')
library('extrafont')

# setwd("~/Google Drive/Classes/Stats 608/Project/Code")

true_loglik = -5196.976212643527

results = read.csv('./intermediate_data/singlerun_results.csv')
results$iter = 1:nrow(results)
results = reshape(results, direction='long',
    varying=c("logliks_em", "times_em",
              "logliks_da", "times_da",
              "logliks_sa", "times_sa"),
    idvar='iter', timevar='method', sep='_')
results$method[results$method == 'em'] = 'EM Algorithm'
results$method[results$method == 'da'] = 'Deterministic Anneal'
results$method[results$method == 'sa'] = 'Simulated Anneal'


sa = read.csv('./intermediate_data/sa_singlerun.csv')
sa$iter = 1:nrow(sa)
sa = reshape(sa, direction='long',
    varying=c("logliks_curr", "logliks_best"),
    idvar='iter', timevar='type', sep='_')
sa$type[sa$type == 'curr'] = 'Current'
sa$type[sa$type == 'best'] = 'Best'

sa_950 = read.csv('./intermediate_data/sa_t950.csv')
sa_975 = read.csv('./intermediate_data/sa_t975.csv')
sa_992 = read.csv('./intermediate_data/sa_t992.csv')
sa_999 = read.csv('./intermediate_data/sa_t999.csv')
sa_950$run = as.factor(sa_950$run)
sa_975$run = as.factor(sa_975$run)
sa_992$run = as.factor(sa_992$run)
sa_999$run = as.factor(sa_999$run)

sa_950_mean = ddply(sa_950, .(iter), summarize,  loglik=mean(loglik_best))
sa_975_mean = ddply(sa_975, .(iter), summarize,  loglik=mean(loglik_best))
sa_992_mean = ddply(sa_992, .(iter), summarize,  loglik=mean(loglik_best))
sa_999_mean = ddply(sa_999, .(iter), summarize,  loglik=mean(loglik_best))
sa_950_mean$alpha = '0.950'
sa_975_mean$alpha = '0.975'
sa_992_mean$alpha = '0.992'
sa_999_mean$alpha = '0.999'

sa_means = rbind(sa_950_mean, sa_975_mean, sa_992_mean, sa_999_mean)

gplot_byiter = ggplot(results, aes(x=iter, y=logliks, color=method)) +
    geom_line() +
    geom_hline(yintercept=true_loglik) +
    coord_cartesian(ylim=c(-6350, -4850)) +
    xlab('Iterations') + ylab('Log Likelihood') +
    theme_bw() +
    theme(legend.position=c(1,0), legend.justification=c(1,0),
          legend.title=element_blank(),
          legend.background=element_rect(color="lightgrey"))

gplot_bytime = ggplot(results, aes(x=times, y=logliks, color=method)) +
    geom_line() +
    geom_hline(yintercept=true_loglik) +
    coord_cartesian(ylim=c(-6350, -4850)) + xlim(NA, 0.45) +
    xlab('Time') + ylab('Log Likelihood') +
    theme_bw() +
    theme(legend.position=c(1,0), legend.justification=c(1,0),
          legend.title=element_blank(),
          legend.background=element_rect(color="lightgrey"))

gplot_singlesa = ggplot(sa, aes(x=iter, y=logliks, color=type)) +
    geom_line() +
    geom_hline(yintercept=true_loglik) +
    coord_cartesian(ylim=c(-6350, -4850)) +
    xlab('Iterations') + ylab('Log Likelihood') +
    theme_bw() +
    theme(legend.position=c(1,0), legend.justification=c(1,0),
          legend.title=element_blank(),
          legend.background=element_rect(color="lightgrey"))

gplot_sa950 = ggplot(sa_950[sa_950$run %in% 0:9,],
                     aes(x=iter, y=loglik_best)) +
    geom_line(aes(color=run)) +
    geom_hline(yintercept=true_loglik) +
    coord_cartesian(ylim=c(-6350, -4850)) +
    xlab('Iterations') + ylab('Log Likelihood') +
    theme_bw() +
    theme(legend.position="none")

gplot_sa975 = ggplot(sa_975[sa_975$run %in% 0:9,],
                     aes(x=iter, y=loglik_best)) +
    geom_line(aes(color=run)) +
    geom_hline(yintercept=true_loglik) +
    coord_cartesian(ylim=c(-6350, -4850)) +
    xlab('Iterations') + ylab('Log Likelihood') +
    theme_bw() +
    theme(legend.position="none")

gplot_sa992 = ggplot(sa_992[sa_992$run %in% 0:9,],
                     aes(x=iter, y=loglik_best)) +
    geom_line(aes(color=run)) +
    geom_hline(yintercept=true_loglik) +
    coord_cartesian(ylim=c(-6350, -4850)) +
    xlab('Iterations') + ylab('Log Likelihood') +
    theme_bw() +
    theme(legend.position="none")

gplot_sa999 = ggplot(sa_999[sa_999$run %in% 0:9,],
                     aes(x=iter, y=loglik_best)) +
    geom_line(aes(color=run)) +
    geom_hline(yintercept=true_loglik) +
    coord_cartesian(ylim=c(-6350, -4850)) +
    xlab('Iterations') + ylab('Log Likelihood') +
    theme_bw() +
    theme(legend.position="none")

gplot_sameans = ggplot(sa_means, aes(x=iter, y=loglik, color=alpha)) +
    geom_line() +
    geom_hline(yintercept=true_loglik) +
    coord_cartesian(ylim=c(-5650, -4950)) +
    xlab('Iterations') + ylab('Log Likelihood') +
    theme_bw() +
    theme(legend.position=c(1,0), legend.justification=c(1,0),
          legend.title=element_blank(),
          legend.background=element_rect(color="lightgrey"))



pdf('./figures/results_byiter.pdf', height=5, width=5, family='CM Roman')
print(gplot_byiter)
dev.off()

pdf('./figures/results_bytime.pdf', height=5, width=5, family='CM Roman')
print(gplot_bytime)
dev.off()

pdf('./figures/sa_singlerun.pdf', height=5, width=5, family='CM Roman')
print(gplot_singlesa)
dev.off()


pdf('./figures/sa950.pdf', height=5, width=5, family='CM Roman')
print(gplot_sa950)
dev.off()
pdf('./figures/sa975.pdf', height=5, width=5, family='CM Roman')
print(gplot_sa975)
dev.off()
pdf('./figures/sa992.pdf', height=5, width=5, family='CM Roman')
print(gplot_sa992)
dev.off()
pdf('./figures/sa999.pdf', height=5, width=5, family='CM Roman')
print(gplot_sa999)
dev.off()


pdf('./figures/sameans_byalpha.pdf', height=5, width=5, family='CM Roman')
print(gplot_sameans)
dev.off()
