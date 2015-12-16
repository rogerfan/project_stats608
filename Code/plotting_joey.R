library(ggplot2)
library(plyr)
library(extrafont)

# load all required data sets
dat_em = read.csv("./intermediate_data/em_2000.csv")
dat_da = read.csv("./intermediate_data/da_2000.csv")
dat_sa = read.csv("./intermediate_data/sa_2000.csv")
dat_sa = dat_sa[,-3]
names(dat_sa) = names(dat_em)
dat_em$typ = as.factor("EM Algorithm")
dat_da$typ = as.factor("Deterministic Anneal")
dat_sa$typ = as.factor("Stochastic EM")
dat_em$size = as.factor(2000)
dat_da$size = as.factor(2000)
dat_sa$size = as.factor(2000)
dat_em = dat_em[!is.na(dat_em$loglik_curr),]
dat_da = dat_da[!is.na(dat_da$loglik_curr),]
dat_sa = dat_sa[!is.na(dat_sa$loglik_curr),]
dat = rbind(dat_em, dat_da, dat_sa)

dat_em = read.csv("./intermediate_data/em_1000.csv")
dat_da = read.csv("./intermediate_data/da_1000.csv")
dat_sa = read.csv("./intermediate_data/sa_1000.csv")
dat_sa = dat_sa[,-3]
names(dat_sa) = names(dat_em)
dat_em$typ = as.factor("EM Algorithm")
dat_da$typ = as.factor("Deterministic Anneal")
dat_sa$typ = as.factor("Stochastic EM")
dat_em$size = as.factor(1000)
dat_da$size = as.factor(1000)
dat_sa$size = as.factor(1000)
dat_em = dat_em[!is.na(dat_em$loglik_curr),]
dat_da = dat_da[!is.na(dat_da$loglik_curr),]
dat_sa = dat_sa[!is.na(dat_sa$loglik_curr),]
dat = rbind(dat, dat_em, dat_da, dat_sa)

dat_em = read.csv("./intermediate_data/em_500.csv")
dat_da = read.csv("./intermediate_data/da_500.csv")
dat_sa = read.csv("./intermediate_data/sa_500.csv")
dat_sa = dat_sa[,-3]
names(dat_sa) = names(dat_em)
dat_em$typ = as.factor("EM Algorithm")
dat_da$typ = as.factor("Deterministic Anneal")
dat_sa$typ = as.factor("Stochastic EM")
dat_em$size = as.factor(500)
dat_da$size = as.factor(500)
dat_sa$size = as.factor(500)
dat_em = dat_em[!is.na(dat_em$loglik_curr),]
dat_da = dat_da[!is.na(dat_da$loglik_curr),]
dat_sa = dat_sa[!is.na(dat_sa$loglik_curr),]
dat = rbind(dat, dat_em, dat_da, dat_sa)


pw_avg_lik = ddply(dat,
                    c("typ","iter","size"),
                    summarise,
                    avg_lik = mean(loglik_curr))

pw2000 = subset(pw_avg_lik, size==2000)
pw1000 = subset(pw_avg_lik, size==1000)
pw500 = subset(pw_avg_lik, size==500)

# plotting
gplot_2000 = ggplot(pw2000, aes(x=iter, y=avg_lik, color=typ)) +
   geom_line() +
   coord_cartesian(ylim=c(-6500, -4850)) + xlim(NA, 200) +
   xlab('Iterations') + ylab('Log Likelihood') +
   theme_bw() +
   theme(legend.position=c(1,0), legend.justification=c(1,0),
         legend.title=element_blank(),
         legend.background=element_rect(color="lightgrey"))

gplot_1000 = ggplot(pw1000, aes(x=iter, y=avg_lik)) +
   geom_line(aes(colour=typ)) +
   coord_cartesian(ylim=c(-3500, -2400)) + xlim(NA, 200) +
   xlab('Iterations') + ylab('Log Likelihood') +
   theme_bw() +
   theme(legend.position=c(1,0), legend.justification=c(1,0),
         legend.title=element_blank(),
         legend.background=element_rect(color="lightgrey"))

gplot_500 = ggplot(pw500, aes(x=iter, y=avg_lik)) +
   geom_line(aes(colour=typ)) +
   coord_cartesian(ylim=c(-1800, -1150)) + xlim(NA, 200) +
   xlab('Iterations') + ylab('Log Likelihood') +
   theme_bw() +
   theme(legend.position=c(1,0), legend.justification=c(1,0),
         legend.title=element_blank(),
         legend.background=element_rect(color="lightgrey"))


pdf('figures/avg_lik_2000.pdf', height=4.5, width=5.5, family='CM Roman')
print(gplot_2000)
dev.off()

# pdf('figures/avg_lik_1000.pdf', height=4.5, width=5.5, family='CM Roman')
# print(gplot_1000)
# dev.off()

pdf('figures/avg_lik_500.pdf', height=4.5, width=5.5, family='CM Roman')
print(gplot_500)
dev.off()


# code for comparisons of EM
mix_df = read.csv("./intermediate_data/em_mix.csv")
mu_df = read.csv("./intermediate_data/em_mean.csv")
mix_df = subset(mix_df, iter < 300)
mu_df = subset(mu_df, iter < 300)
names(mix_df) = names(mu_df) = c("run", "iter", "loglik")
fn = function(vect) {
   return(vect - max(vect))
}

mix_df_plot = ddply(mix_df, "run", transform,
                dist_fc = fn(loglik))
mu_df_plot = ddply(mu_df, "run", transform,
                    dist_fc = fn(loglik))


gplot_mix = ggplot(mix_df_plot, aes(x=iter, y=-dist_fc)) +
   geom_line(aes(color=as.factor(run))) +
   xlab('Iteration') + ylab('Distance from max loglik') +
   theme_bw() +
   theme(legend.position=c(1,1), legend.justification=c(1,1),
         legend.title=element_blank(),
         legend.background=element_rect(color="lightgrey")) +
   scale_color_discrete(name="",
                        breaks=c(0,1),
                        labels=c("Unbalanced",
                                 "Balanced"))

gplot_overlap = ggplot(mu_df_plot, aes(x=iter, y=-dist_fc)) +
   geom_line(aes(color=as.factor(run))) +
   xlab('Iteration') + ylab('Distance from max loglik') +
   theme_bw() +
   theme(legend.position=c(1,1), legend.justification=c(1,1),
         legend.title=element_blank(),
         legend.background=element_rect(color="lightgrey")) +
   scale_color_discrete(name="",
                        breaks=c(0,1),
                        labels=c("More Overlap",
                                 "Less Overlap"))


pdf('figures/em_imbalance.pdf', height=4.5, width=5.5, family='CM Roman')
print(gplot_mix)
dev.off()

pdf('figures/em_overlap.pdf', height=4.5, width=5.5, family='CM Roman')
print(gplot_overlap)
dev.off()
