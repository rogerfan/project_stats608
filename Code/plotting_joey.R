# load all required data sets
dat_em <- read.csv("./intermediate_data/em_2000.csv")
dat_da<- read.csv("./intermediate_data/da_2000.csv")
dat_sa<- read.csv("./intermediate_data/sa_2000.csv")
dat_em$typ <- as.factor("em")
dat_da$typ <- as.factor("da")
dat_sa$typ <- as.factor("sa")
dat_em$size <- as.factor(2000)
dat_da$size <- as.factor(2000)
dat_sa$size <- as.factor(2000)
dat <- rbind(dat_em, dat_da, dat_sa[,-4])

dat_em <- read.csv("./intermediate_data/em_1000.csv")
dat_da<- read.csv("./intermediate_data/da_1000.csv")
dat_sa<- read.csv("./intermediate_data/sa_1000.csv")
dat_em$typ <- as.factor("em")
dat_da$typ <- as.factor("da")
dat_sa$typ <- as.factor("sa")
dat_em$size <- as.factor(1000)
dat_da$size <- as.factor(1000)
dat_sa$size <- as.factor(1000)

dat <- rbind(dat, dat_em, dat_da, dat_sa[,-4])

dat_em <- read.csv("./intermediate_data/em_500.csv")
dat_da<- read.csv("./intermediate_data/da_500.csv")
dat_sa<- read.csv("./intermediate_data/sa_500.csv")
dat_em$typ <- as.factor("em")
dat_da$typ <- as.factor("da")
dat_sa$typ <- as.factor("sa")
dat_em$size <- as.factor(500)
dat_da$size <- as.factor(500)
dat_sa$size <- as.factor(500)

dat <- rbind(dat, dat_em, dat_da, dat_sa[,-4])

library(ggplot2); library(plyr);library('extrafont')

pw_avg_lik <- ddply(dat, 
                    c("typ","iter","size"),
                    summarise, 
                    avg_lik = mean(loglik_curr))

pw2000 <- subset(pw_avg_lik, size==2000)
pw1000 <- subset(pw_avg_lik, size==1000)
pw500 <- subset(pw_avg_lik, size==500)

# plotting
gplot_2000 = ggplot(pw2000, aes(x=iter, y=avg_lik)) +
   geom_line(aes(colour=typ)) +
   coord_cartesian(ylim=c(-6500, -4850)) + xlim(NA, 200) +
   xlab('Iteartion Number') + ylab('Log Likelihood') +
   ggtitle('N=2000') +
   theme_bw() +
   theme(legend.position=c(1,0), legend.justification=c(1,0),
         legend.title=element_blank(),
         legend.background=element_rect(color="lightgrey"))

gplot_1000 = ggplot(pw1000, aes(x=iter, y=avg_lik)) +
   geom_line(aes(colour=typ)) +
   #coord_cartesian(ylim=c(-3500, -2400)) + xlim(NA, 200) +
   xlab('Iteartion Number') + ylab('Log Likelihood') +
   ggtitle('N=1000') +
   theme_bw() +
   theme(legend.position=c(1,0), legend.justification=c(1,0),
         legend.title=element_blank(),
         legend.background=element_rect(color="lightgrey"))

gplot_500 = ggplot(pw500, aes(x=iter, y=avg_lik)) +
   geom_line(aes(colour=typ)) +
   #coord_cartesian(ylim=c(-1800, -1200)) + xlim(NA, 200) +
   xlab('Iteartion Number') + ylab('Log Likelihood') +
   ggtitle('N=500') +
   theme_bw() +
   theme(legend.position=c(1,0), legend.justification=c(1,0),
         legend.title=element_blank(),
         legend.background=element_rect(color="lightgrey"))

setwd("project_stat608/include/")

pdf('avg_lik_2000.pdf', height=5, width=5, family='CM Roman')
print(gplot_2000)
dev.off()

pdf('avg_lik_1000.pdf', height=5, width=5, family='CM Roman')
print(gplot_1000)
dev.off()

pdf('avg_lik_500.pdf', height=5, width=5, family='CM Roman')
print(gplot_500)
dev.off()


# code for comparisons of EM

setwd("~/Box Sync/stat608/project_stats608/data")
mix_df <- read.csv("em_mix.csv")
mu_df <- read.csv("em_mean.csv")
mix_df <- subset(mix_df, iter < 300)
mu_df <- subset(mu_df, iter < 300)
names(mix_df) = names(mu_df) <- c("run", "iter", "loglik")
fn <- function(vect) {
   return(vect - max(vect))
}

mix_df_plot <- ddply(mix_df, "run", transform,
                dist_fc = fn(loglik))
mu_df_plot <- ddply(mu_df, "run", transform,
                    dist_fc = fn(loglik))


gplot_mix = ggplot(mix_df_plot, aes(x=iter, y=-dist_fc)) +
   geom_line(aes(colour=as.factor(run))) +
   xlab('Iteartion Number') + ylab('Distance from max(Loglik)') +
   ggtitle('Effect of Imbalance in Mixing Coefficients') +
   theme_bw() +
   theme(legend.position=c(1,.75), legend.justification=c(1,0),
         legend.title=element_blank(),
         legend.background=element_rect(color="lightgrey")) +
   scale_color_discrete(name="",
                        breaks=c(0,1),
                        labels=c("Unbalanced",
                                 "Balanced"))

gplot_overlap = ggplot(mu_df_plot, aes(x=iter, y=-dist_fc)) +
   geom_line(aes(colour=as.factor(run))) +
   xlab('Iteartion Number') + ylab('Distance from max(Loglik)') +
   ggtitle('Effect of Overlap') +
   theme_bw() +
   theme(legend.position=c(1,.75), legend.justification=c(1,0),
         legend.title=element_blank(),
         legend.background=element_rect(color="lightgrey")) +
   scale_color_discrete(name="",
                        breaks=c(0,1),
                        labels=c("More Overlap",
                                 "Less Overlap"))


setwd("project_stat608/include/")

pdf('imbalance.pdf', height=5, width=5, family='CM Roman')
print(gplot_mix)
dev.off()

pdf('imbalance.pdf', height=5, width=5, family='CM Roman')
print(gplot_overlap)
dev.off()




