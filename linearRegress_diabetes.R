library(rstanarm)
library(corrplot)
library(ggplot2)

options(mc.cores = parallel::detectCores())
set.seed(123)

#' load data
db = read.csv("http://www.rob-mcculloch.org/data/diabetes.csv")

x <- db[,c(2:11)]
y <- db$y

#' Plot correlation structure
corrplot(cor(db[, c(1,2:11)]))

#' Generalized linear modeling with optional prior distributions 
#' for the coefficients, intercept, and auxiliary parameters.
#' run the model
model <- stan_glm(y ~ age + sex + bmi + map + tc + ldl+ hdl + tch + ltg + glu, 
                  chains=4,iter=2000, warmup=250,data=db)
# warm up = you neglect the first 250
# chain = 4: you run 4 times independently this alg and put together in the end,
# in this way you paralelize your work, you do more iterations and you have better result
model

plot(model, prob = 0.5)
plot(model, prob = 0.5, pars = "beta")
plot(model, prob = 0.5, "hist", pars="sigma")


pplot<-plot(model, "areas", prob = 0.95, prob_outer = 1)
pplot+ geom_vline(xintercept = 0)

prior_summary(model)


round(coef(model), 2)

round(posterior_interval(model, prob = 0.95), 2)


launch_shinystan(model, ppd = FALSE)

y_rep <- posterior_predict(model)
dim(y_rep)
## 7000  442

pp_check(model, plotfun = "stat", stat = "mean")
pp_check(model, plotfun = "stat_2d", stat = c("mean", "sd"))

# for alpha:
mean(posteriors$`(Intercept)`) # posterior mean of alpha
bayestestR::map_estimate(posteriors$`(Intercept)`)

ggplot(posteriors,aes(x = `(Intercept)`)) + 
  geom_density(fill="orange")  +   # posterior density of alpha
  geom_vline(xintercept = mean(posteriors$`(Intercept)`),size=1,col=1)+
  geom_vline(xintercept = median(posteriors$`(Intercept)`),size=1,col=2)+
  geom_vline(xintercept = bayestestR::map_estimate(posteriors$`(Intercept)`),size=1,col=3)

## we pick bmi, ltg and map becasue they were the most important
## variables in CART and RF model prediciton

# for age
# highest density regions (equivalent of confidence intervals in the bayesian setting):
Hdi_age <- bayestestR::hdi(posteriors$age, ci=0.9) # we set confidence interval to 90%
Hdi_age$CI_low    # lower bound
Hdi_age$CI_high   # upper bound

ggplot(posteriors,aes(x = age)) +
  geom_density(fill="orange") +
  geom_vline(xintercept = Hdi_age$CI_low,size=1)+
  geom_vline(xintercept = Hdi_age$CI_high,size=1)

range(posteriors$age) # from min to max

# bmi
# highest density regions (equivalent of confidence intervals in the bayesian setting):
Hdi_bmi <- bayestestR::hdi(posteriors$bmi, ci=0.9) # we set confidence interval to 90%
Hdi_bmi$CI_low    # lower bound
Hdi_bmi$CI_high   # upper bound

ggplot(posteriors,aes(x = bmi)) +
  geom_density(fill="blue") +
  geom_vline(xintercept = Hdi_bmi$CI_low,size=1)+
  geom_vline(xintercept = Hdi_bmi$CI_high,size=1)

range(posteriors$bmi) # from min to max

# ltg
# highest density regions (equivalent of confidence intervals in the bayesian setting):
Hdi_ltg <- bayestestR::hdi(posteriors$ltg, ci=0.9) # we set confidence interval to 90%
Hdi_ltg$CI_low    # lower bound
Hdi_ltg$CI_high   # upper bound

ggplot(posteriors,aes(x = ltg)) +
  geom_density(fill="red") +
  geom_vline(xintercept = Hdi_ltg$CI_low,size=1)+
  geom_vline(xintercept = Hdi_ltg$CI_high,size=1)

range(posteriors$ltg) # from min to max

# for map
# highest density regions (equivalent of confidence intervals in the bayesian setting):
Hdi_map <- bayestestR::hdi(posteriors$map, ci=0.9) # we set confidence interval to 90%
Hdi_map$CI_low    # lower bound
Hdi_map$CI_high   # upper bound

ggplot(posteriors,aes(x = map)) +
  geom_density(fill="orange") +
  geom_vline(xintercept = Hdi_map$CI_low,size=1)+
  geom_vline(xintercept = Hdi_map$CI_high,size=1)

range(posteriors$map) # from min to max

#' Leave-one-out cross-validation
(loo1 <- loo(model, save_psis = TRUE))

#' Comparison to a baseline model
#' Compute baseline result without covariates.
post0 <- update(model, formula = y ~ 1, QR = FALSE, data=db)
#' Compare to baseline
(loo0 <- loo(model))
loo_compare(loo0, loo1)

#' Other predictive performance measurements
#' 
# Predicted probabilities
linpred <- posterior_linpred(model)
preds <- posterior_linpred(model, transform=TRUE)
pred <- colMeans(preds)
