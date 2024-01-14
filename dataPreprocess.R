library(twang)
# data(AOD)
# summary(AOD)
setwd("~/Desktop/有事/Harvard/1Fall 2023/MIT 6.7900/Project")
set.seed(67900)

######## Binary Data ######## 
# twin_scale <- read.csv("./merged_twin_pairs_data_scale.csv")
# twin <- read.csv("./merged_twin_pairs_data.csv")
twin <- read.csv("./merged_outcome_twin_unscaled.csv")
twin_0 <- subset(twin, Outcome == 0)
twin_1 <- subset(twin, Outcome == 1)

# Sample 5000 observations from each group
sample_0 <- twin_0[sample(nrow(twin_0), 5000), ]
sample_1 <- twin_1[sample(nrow(twin_1), 5000), ]
# Combine the samples
balanced_twin <- rbind(sample_0, sample_1)

write.csv(balanced_twin, "./balanced_twin.csv", row.names = FALSE)

n <- nrow(twin)
sampled_indices <- sample(n, size = 1000)
sampled_twin <- twin[sampled_indices, ]

# summary(twin)

########  Multi-valued Dataset ######## 
price_train <- read.csv("./Data/price_final.csv") # Do not change this line
price_val <- read.csv("./Data/data/Demand/0/val.csv")
price_test <- read.csv("./Data/data/Demand/1/train.csv")
# summary(price_val)
# summary(price_test)

old_names <- colnames(price_val)
new_names <- c("noise_price", "noise_demand", "cost", "time", "emotion", 
               "price", "mu0", "mut", "structural", "outcome","price_sensitive")
colnames(price_val) <- new_names
colnames(price_test) <- new_names

price_indices <- sample(nrow(price_test), size = 2000)
price_test <- price_test[price_indices, ]

write.csv(price_val, "./Data/price_val.csv", row.names = FALSE)
write.csv(price_test, "./Data/price_test.csv", row.names = FALSE)
