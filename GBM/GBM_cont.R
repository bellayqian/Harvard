library(survey)
library(polycor)
library(gbm)
setwd("~/Desktop/有事/Harvard/1Fall 2023/MIT 6.7900/Project")
set.seed(42)
F.aac.iter <- function(i,data,ps.model,ps.num,rep,criterion) {
  # i: number of iterations (trees)
  # data: dataset containing the treatment and the covariates
  # ps.model: the boosting model to estimate p(T_iX_i)
  # ps.num: the estimated p(T_i)
  # rep: number of replications in bootstrap
  # criterion: the correlation metric used as the stopping criterion
  GBM.fitted = predict(ps.model, newdata = data, n.trees = floor(i), type = 'response')
  ps.den = dnorm((data$T-GBM.fitted)/sd(data$T-GBM.fitted),0,1)
  wt = ps.num/ps.den
  aac_iter = rep(NA, rep) 
  for (i in 1:rep){
    bo = sample(1:dim(data)[1],replace = TRUE,prob = wt)
    newsample = data[bo,]
    j.drop = match(c("T"),names(data))
    j.drop = j.drop[!is.na(j.drop)]
    x = newsample[,-j.drop]
    if(criterion == "spearman"| criterion == "kendall"){
      ac = apply(x, MARGIN = 2, FUN = cor, y = newsample$T, method = criterion)
    } else if (criterion == "distance"){
      ac = apply(x, MARGIN = 2, FUN = dcor, y = newsample$T) 
    } else if (criterion == "pearson"){
      ac = matrix(NA,dim(x)[2],1)
      for (j in 1:dim(x)[2]){
        ac[j] = ifelse (!is.factor(x[,j]), 
                        cor(newsample$T, x[,j],method=criterion),
                        polyserial(newsample$T, x[,j])) 
      }
    } else print("The criterion is not correctly specified") 
    aac_iter[i] = mean(abs(1/2*log((1+ac)/(1-ac))),na.rm = TRUE) 
    }
  aac = mean(aac_iter) 
  return(aac)
}

# Find the optimal number of trees using Pearson/polyserial correlation 
# train_data <- read.csv("./Data/price_final.csv") # Do not change this line
# all_indices <- 1:10000
# train_indices <- sample(all_indices, 8000)
# test_indices <- setdiff(all_indices, train_indices)
# train <- train_data[train_indices, ]
# test <- train_data[test_indices, ]
train <- read.csv("./Data/train_set_full.csv") # Do not change this line
test <- read.csv("./Data/test_set_full.csv") # Do not change this line
summary(train)
dataset_type <- "train" # "test" "val"

x <- data.frame(train$cost, train$time, train$emotion, train$price_sensitive)
mydata <- data.frame(T=train$price, X=x)

model.num <- lm(T~1, data = mydata)
ps.num <- dnorm((mydata$T-model.num$fitted)/(summary(model.num))$sigma, 0, 1)
model.den <- gbm(T~., data = mydata, shrinkage = 0.1, interaction.depth = 1, 
                 n.trees = 20000, verbose = TRUE)
opt <- optimize(F.aac.iter,interval = c(1,20000), data = mydata, ps.model = model.den,
                 ps.num = ps.num, rep=50, criterion= "pearson")
# saveRDS(opt, file = paste0("./Data/", dataset_type, "_opt.rds"))
saveRDS(opt, file = paste0("./Data/", dataset_type, "_opt_new.rds"))

best.aac.iter <- opt$minimum
best.aac <- opt$objective
# saveRDS(model.den, file = paste0("./Data/", dataset_type, "_model.den.rds"))
saveRDS(model.den, file = paste0("./Data/", dataset_type, "_model.den_new.rds"))

# Calculate the inverse probability weights
# model.den$fitted <- predict(model.den, newdata = mydata, n.trees = floor(best.aac.iter), type = "response")
# ps.den <- dnorm((mydata$T-model.den$fitted)/sd(mydata$T-model.den$fitted),0,1)
# weight.gbm <- ps.num/ps.den
# saveRDS(model.den, file = paste0("./Data/", dataset_type, "_model.den.rds"))

x <- data.frame(test$cost, test$time, test$emotion, test$price_sensitive)
testdata <- data.frame(T=test$price, X=x)
new_names <- c("T", "X.train.cost","X.train.time","X.train.emotion","X.train.price_sensitive")
colnames(testdata) <- new_names

## Test Dataset prediction 
propensity_scores <- predict(model.den, newdata = testdata, n.trees = floor(best.aac.iter), type = "response")
testdata$ps <- propensity_scores
new_names2 <- c("T", "X.test.cost","X.test.time","X.test.emotion","X.test.price_sensitive","GBM_predicted")
colnames(testdata) <- new_names2
# saveRDS(testdata, file = paste0("./Data/testdata.rds"))
# testdata <- readRDS("./Data/testdata.rds")
saveRDS(testdata, file = paste0("./Data/testdata_new.rds"))
# testdata <- readRDS("./Data/testdata_new.rds")
write.csv(testdata$GBM_predicted, "./Data/test_predict_price.csv", row.names = FALSE)

mse <- mean((testdata$GBM_predicted - testdata$T)^2)
rmse <- sqrt(mse)

## Weight
iptw_weights <- 1 / propensity_scores
outcome.df <- data.frame(outcome = test$outcome, price = test$price, predict_price = testdata$GBM_predicted, 
                         iptw_weights)
design.b <- svydesign(ids = ~1, weights = ~iptw_weights, data = outcome.df)
# fit <- svyglm(outcome~price, family = quasibinomial(),design = design.b)

traindf$weights <- get.weights(ps.dat2, stop.method = "ks.mean")
design.ps <- svydesign(ids=~1, weights = ~weights, data = traindf)
with_weight_data <- design.b$variables

treated_outcome <- sum(with_weight_data$Outcome * with_weight_data$weights * with_weight_data$T_train) / sum(with_weight_data$weights * with_weight_data$T_train)
control_outcome <- sum(with_weight_data$Outcome * with_weight_data$weights * (1 - with_weight_data$T_train)) / sum(with_weight_data$weights * (1 - with_weight_data$T_train))
ate <- treated_outcome - control_outcome
ate#obtained ate here

summary(model.den$fitted)
summary(train$price)
train$predict_price <- model.den$fitted
model.den$




# # Outcome analysis using survey package
# outcome.df <- data.frame(train$outcome, train$price, weight.gbm)
# saveRDS(outcome.df, file = paste0("./Data/", dataset_type, "_outcome.df.rds"))
# design.b <- svydesign(ids = ~1, weights = ~weight.gbm, data = outcome.df)
# fit <- svyglm(train$outcome~train$price, family = quasibinomial(),design = design.b)
# summary(fit)






