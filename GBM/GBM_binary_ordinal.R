library(twang)
library(gbm)
library(survey)
setwd("C:/Users/Min/Desktop/machine learning project")
# dat <- read.csv("merged_twin_pairs_data_not_normalized.csv")
# n <- nrow(dat)
# sampled_indices <- sample(n, size = 1000)
# sampled_twin <- dat[sampled_indices, ]
# 
# ps.dat <- ps(T~pldel + birattnd + brstate + stoccfipb+mager8+ormoth+mrace+meduc6+dmar+mplbir+mpre5+adequacy +orfath+frace+birmon+gestat10+csex+anemia+cardiac+lung+ diabetes + herpes+hydra+hemo+chyper+phyper+eclamp+incervix+pre4000+preterm+renal+rh+uterine+othermr+tobacco+alcohol+cigar6+drink5+crace+data_year+nprevistq+dfageq+feduc6+dlivord_min+dtotord_min+bord_0+bord_1, 
#              data = sampled_twin,
#              # n.trees=5000,
#              # interaction.depth=2,
#              # shrinkage=0.01,
#              estimand = "ATE",
#              # stop.method=c("es.mean","ks.max"),
#              verbose=TRUE)
# plot(ps.dat)
# table <- bal.table(ps.dat)
#write.csv(table, "table n=1000.csv")
#try to conver to integer


df <- read.csv("merged_outcome_twin_unscaled.csv")

# Splitting features and treatment 
X <- df[, !(names(df) %in% c("tobacco", "Outcome"))]  # Covariates
T <- df$tobacco  # binary Treatment Variable
Y <- df$Outcome

# Splitting data into training and testing sets
set.seed(42)  # for reproducibility
library(caTools)
split <- sample.split(T, SplitRatio = 0.8)
X_train <- subset(X, split == TRUE)
T_train <- subset(T, split == TRUE)
X_test <- subset(X, split == FALSE)
T_test <- subset(T, split == FALSE)
Y_train <- subset(Y, split == TRUE)
Y_test <- subset(Y, split == TRUE)

traindf <- data.frame(T_train, X_train)

#chisq.test(table(dat$Outcome , dat$tobacco)) # the treatment and outcome are associated

ps.dat <- ps(T_train~pldel + birattnd + brstate + stoccfipb+mager8+ormoth+mrace+meduc6+dmar+mplbir+mpre5+adequacy +orfath+frace+birmon+gestat10+csex+anemia+cardiac+lung+ diabetes + herpes+hydra+hemo+chyper+phyper+eclamp+incervix+pre4000+preterm+renal+rh+uterine+othermr+alcohol+cigar6+drink5+crace+data_year+nprevistq+dfageq+feduc6+dlivord_min+dtotord_min + bord_0 + bord_1, 
             data = traindf,
             n.trees=5000,
             # interaction.depth=2,
             shrinkage=0.015,
             estimand = "ATE",
             stop.method=c("es.mean","ks.mean"),
             verbose=TRUE)
summary(ps.dat)
table <- bal.table(ps.dat)
table
write.csv(table, "new data ntrees 5000 shrinkage 0.015.csv")
plot(ps.dat, plots = 2)
#ks.mean.ATE    6791  50285  174.8305 50252.89 0.6798719 0.07395168 0.2315864       NA 0.03200941 3308
#refit gs using optimal n.trees
ps.dat2 <- ps(T_train~pldel + birattnd + brstate + stoccfipb+mager8+ormoth+mrace+meduc6+dmar+mplbir+mpre5+adequacy +orfath+frace+birmon+gestat10+csex+anemia+cardiac+lung+ diabetes + herpes+hydra+hemo+chyper+phyper+eclamp+incervix+pre4000+preterm+renal+rh+uterine+othermr+alcohol+cigar6+drink5+crace+data_year+nprevistq+dfageq+feduc6+dlivord_min+dtotord_min + bord_0 + bord_1, 
             data = traindf,
             n.trees=3308,
             # interaction.depth=2,
             shrinkage=0.015,
             estimand = "ATE",
             stop.method=c("ks.mean"),
             verbose=TRUE)
summary(ps.dat2)
save(ps.dat2, file = "ps.dat2.RData")
#START HERE IF LOADING RDATA
load("ps.dat2.RData")
gbm_model <- ps.dat2$gbm.obj
#we have obtained an optimal GBM using ks.mean and is saved in gbm_model
testdf <- data.frame(T_test, X_test)
propensity_scores <- predict(gbm_model, newdata = testdf, type = "response")
#obtained propensity score 
testdf$ps <- propensity_scores

#Calculate mse here
test_predictions <- predict(gbm_model, newdata = testdf, type = "response")
mse <- mean((T_test - test_predictions)^2)
print(paste("Test Mean Squared Error:", mse)) #"Test Mean Squared Error: 0.00574815105408259"

# following code no use here
# dat$weights <- get.weights(ps.dat, stop.method = "ks.mean")
# design.ps <- svydesign(ids=~1, weights = ~weights, data = dat)
# with_weight_data <- design.ps$variables
# with_weight_data$ps <- NA
# with_weight_data$ps[with_weight_data$tobacco == 1] <- 1/with_weight_data$weights[with_weight_data$tobacco ==1]
# with_weight_data$ps[with_weight_data$tobacco == 0] <- 1 - 1/with_weight_data$weights[with_weight_data$tobacco ==0]

testdf$predtreat <- ifelse(testdf$ps > 0.5,1,0)#predict treatment if ps <0.5, treatment = 0; if ps >0.5, treatment =1

save(testdf, file = "testdf.RData")
write.csv(testdf, "testdf.csv")

#Can start from here by loading Rdata
load("testdf.ps.RData")
testdf <- read.csv("testdf.csv")


confusion_mtx <- table(testdf$predtreat, testdf$T_test)
confusion_mtx
print(confusion_mtx)#x axis is predtreat, y axis is tobacco
#true negative is 12565, true positive is 1616, false positive is 82, false negative is 6

traindf$weights <- get.weights(ps.dat2, stop.method = "ks.mean")
design.ps <- svydesign(ids=~1, weights = ~weights, data = traindf)
with_weight_data <- design.ps$variables
with_weight_data$Outcome <- Y_train

treated_outcome <- sum(with_weight_data$Outcome * with_weight_data$weights * with_weight_data$T_train) / sum(with_weight_data$weights * with_weight_data$T_train)
control_outcome <- sum(with_weight_data$Outcome * with_weight_data$weights * (1 - with_weight_data$T_train)) / sum(with_weight_data$weights * (1 - with_weight_data$T_train))
ate <- treated_outcome - control_outcome
ate#obtained ate here

#draw the confusion matrix here
print(confusion_mtx)
confusion_mtx[1,2]
confusion_matrix <- confusion_mtx
rownames(confusion_matrix) <- c("Actual 0", "Actual 1")
colnames(confusion_matrix) <- c("Predicted 0", "Predicted 1")

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Create a heatmap using ggplot2
library(ggplot2)

# Convert the confusion matrix to a data frame with appropriate column names
confusion_df <- as.data.frame(as.table(confusion_matrix))

# Plot the heatmap
ggplot(data = confusion_df, aes(x = Var1, y = Var2, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +  # Add text labels
  scale_fill_gradient(low = "white", high = "blue") +
  theme_minimal() +
  labs(title = paste("Confusion Matrix\nAccuracy:", scales::percent(accuracy)),
       x = "Actual",
       y = "Predicted",
       fill = "Frequency")



#line of thought:
#propensity score is based on prediction of treatment, and are applied to observations based on how normal it is to the specific treatment
#if not like a typical observation under the treatment, ps is low, then weight is low. 
#first train test split, use train to fit ps(), then find optimal n.tree, refit ps() using train ds
#extract the gbm.obj from ps.dat, which is the gbm model. 

# Assuming you have a dataframe `df`


mse <- mean((T_test - test_predictions)^2)
print(paste("Test Mean Squared Error:", mse))
#obtain the propensity score, then categorize <0.5 treatment =0, >0.5 treatment =1, plot confusion matrix
#propensity score for ordinal treatment will result in three values add up to 1
#select the highest ordinal values, first corresponds to 0, second to 1, third to 2(level), get predicted treament
#construct confusion matrix: if match with the actual treatment, is true positive/.... generate confusionn matrix



##################################################################################
#Ordinal coding
library(twang)
library(survey)
library(caret)
price_full <- read.csv("price_final.csv")
quantiles <- quantile(price_full$price, probs = c(1/3, 2/3))
price_full$price_category <- cut(price_full$price, 
                                 breaks = c(-Inf, quantiles[1], quantiles[2], Inf),
                                 labels = c("low", "medium", "high"),
                                 include.lowest = TRUE)
table(price_full$price_category)
splitIndex <- createDataPartition(y = price_full$outcome , p = 0.8, list = FALSE)
price_train <- price_full[splitIndex, ]
price_test <- price_full[-splitIndex, ]
# Separate price into three categories: low medium high

# write.csv(price_train, "./Data/price_final.csv", row.names = FALSE)
X_train <- price_train[, c("cost", "time", "emotion", "price_sensitive")] # Covariates
T_train <- price_train$price_category  # ordinary Treatment Variable
Y_train <- price_train$outcome

mnps.price <- mnps(price_category~cost+time+emotion+price_sensitive,
                   data = price_train,
                   n.trees=3000,
                   # interaction.depth=2,
                   shrinkage=0.005,
                   estimand = "ATE",
                   stop.method=c("es.mean","ks.mean"),
                   verbose=TRUE)
summary(mnps.price) #stop at 1434 ks.mean 
#actually each of the three glm has different n.trees optimal
table <- bal.table(mnps.price)
table
write.csv(table, "ordinal data ntrees 3000 shrinkage 0.005.csv")
plot(mnps.price, plots = 1)
#refit  using optimal n.trees
mnps.price2 <- mnps(price_category~cost+time+emotion+price_sensitive,
                   data = price_train,
                   n.trees=1434,
                   # interaction.depth=2,
                   shrinkage=0.005,
                   estimand = "ATE",
                   stop.method=c("ks.mean"),
                   verbose=TRUE)
summary(mnps.price2)
save(mnps.price2, file = "mnps.price2.RData")
#START HERE IF LOADING RDATA
load("mnps.price2.RData")
#have one gbm for each treatment?
gbm_model_low <- mnps.price2$psList$low$gbm.obj
gbm_model_medium <- mnps.price2$psList$medium$gbm.obj
gbm_model_high <- mnps.price2$psList$high$gbm.obj

testdf <- price_test


propensity_scores_low <- predict(gbm_model_low, newdata = testdf, type = "response")
propensity_scores_medium <- predict(gbm_model_medium, newdata = testdf, type = "response")
propensity_scores_high <- predict(gbm_model_high, newdata = testdf, type = "response")

#obtained propensity score 
testdf$ps_low <- propensity_scores_low
testdf$ps_medium <- propensity_scores_medium
testdf$ps_high <- propensity_scores_high

#find the predicted treatment using propensity score that is highest

library(dplyr)
testdf <- testdf %>%
  mutate(predicted_treatment = case_when(
    ps_low > ps_medium & ps_low > ps_high ~ "low",
    ps_medium > ps_low & ps_medium > ps_high ~ "medium",
    ps_high > ps_low & ps_high > ps_medium ~ "high",
    TRUE ~ "tie"  # Handle the case where two or more columns have the same maximum value
  ))
#table(testdf$predicted_treatment)["tie"] #no tie occured



save(testdf, file = "testdf.RData")
write.csv(testdf, "testdf.csv")

#Can start from here by loading Rdata
load("testdf.ps.RData")
testdf <- read.csv("testdf.csv")


confusion_mtx <- table(testdf$predicted_treatment, testdf$price_category)
order_levels <- c("low", "medium", "high")
confusion_mtx <- confusion_mtx[order_levels, order_levels]
confusion_mtx #horizontal axis is predicted, verticle axis is actual 
print(confusion_mtx)#x axis is predtreat, y axis is tobacco

#ate maybe later
traindf$weights <- get.weights(ps.dat2, stop.method = "ks.mean")
design.ps <- svydesign(ids=~1, weights = ~weights, data = traindf)
with_weight_data <- design.ps$variables
with_weight_data$Outcome <- Y_train

treated_outcome <- sum(with_weight_data$Outcome * with_weight_data$weights * with_weight_data$T_train) / sum(with_weight_data$weights * with_weight_data$T_train)
control_outcome <- sum(with_weight_data$Outcome * with_weight_data$weights * (1 - with_weight_data$T_train)) / sum(with_weight_data$weights * (1 - with_weight_data$T_train))
ate <- treated_outcome - control_outcome
ate#obtained ate here

#draw the confusion matrix here
confusion_matrix <- confusion_mtx
print(confusion_matrix)
rownames(confusion_matrix) <- c("Actual low", "Actual medium","Actual high")
colnames(confusion_matrix) <- c("Predicted low", "Predicted medium", "Predicted high")

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Create a heatmap using ggplot2
library(ggplot2)

# Convert the confusion matrix to a data frame with appropriate column names
confusion_df <- as.data.frame(as.table(confusion_matrix))

# Plot the heatmap
ggplot(data = confusion_df, aes(x = Var1, y = Var2, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +  # Add text labels
  scale_fill_gradient(low = "white", high = "blue") +
  theme_minimal() +
  labs(title = paste("Confusion Matrix\nAccuracy:", scales::percent(accuracy)),
       x = "Actual",
       y = "Predicted",
       fill = "Frequency")








