rm(list = ls())

library(data.table)
library(Matrix)
library(xgboost)
require(randomForest)
library(caret)
library(plyr)
require(ggplot2)
library(pROC)
library(stringr)
library(Metrics)
library(kernlab)
library(mlbench)
library(MASS)
library(caTools)
library(dplyr)
library(FeatureHashing)
library(gbm)          # basic implementation
library(h2o)          # a java-based platform

train <- fread("trainset2410[rpart+encoding].csv")

train$tota_1stSF <- NULL
train <- train[-c(1299),]
train <- train[-c(1325),]
train <- train[-c(524),]
train <- train[-c(1325),]
train <- train[-c(1325),]
train <- train[-c(610),]

set.seed(123)

sample = sample.split(train$OverallQual, SplitRatio = 0.8)
training = subset(train, sample == TRUE)
testing  = subset(train, sample == FALSE)

# train GBM model
gbm.fit.final <- gbm(
  formula = SalePrice ~ .,
  distribution = "gaussian",
  data = training,
  n.trees = 3234,
  interaction.depth = 5,
  shrinkage = 0.01,
  cv.folds = 5,
  bag.fraction = 0.67,
  n.minobsinnode = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  
# RMSE of GBM model
prediction <- predict(gbm.fit.final, testing, type="response")
model_output <- cbind(testing, prediction)

model_output$log_prediction <- log(model_output$prediction)
model_output$log_SalePrice <- log(model_output$SalePrice)



#Test with RMSE
#0.141 # 0.1098 drop tota_1stSF, delete 1299,1350, 610..
rmse(model_output$log_SalePrice,model_output$log_prediction)
test <- fread("modified_test2410.csv")
test1 <- fread("test.csv")


# Prediction on test set
prediction <- predict(gbm.fit.final, test, type="response")
prediction <- as.data.frame(prediction)
prediction <- cbind(test1$Id, prediction$prediction)
colnames(prediction) <- c("Id", "SalePrice")

write.csv(prediction, "submission8.csv", row.names = FALSE)

