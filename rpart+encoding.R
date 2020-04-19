rm(list = ls())

library(data.table)
library(Matrix)
library(xgboost)
require(randomForest)
require(caret)
require(dplyr)
require(ggplot2)
library(pROC)
library(stringr)
library(Metrics)
library(kernlab)
library(mlbench)
library(MASS)
library(caTools)
# library(DMwR) KNN immutation to handle missing values. Is it better than rpart?
library(rpart)
library(car)
library(e1071)
library(Boruta)


###Clean the Data

setwd("/Users/jessica/Desktop/Kaggle/house-prices-advanced-regression-techniques")
train <- fread("train.csv")
test_pred <- fread("test.csv")

colnames(train)[colnames(train) =="1stFlrSF"] <- "FlrSF1st"
colnames(train)[colnames(train) =="2ndFlrSF"] <- "FlrSF2nd"
colnames(train)[colnames(train) =="3SsnPorch"] <- "SsnPorch3"

train = train %>% mutate_if(is.character, as.factor)

train$Utilities<- NULL # 1400+ Pave No big use
train$RoofMatl<- NULL # 1400+ Pave No big use
train$MiscFeature <- NULL # 1400+ Pave No big use
train$Alley <- NULL # too many missing values
train$PoolQC <- NULL # too many missing values
train$Fence <- NULL # too many missing values, should l impute it or delete it

summary(train)
which(apply(train, 2, function(x) any(is.na(x)))) # to see if there are any missing values
colSums(is.na(train))

# LotFrontage, MasVnrType, MasVnrArea, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 
# BsmtFinType2, Electrical, FireplaceQu, GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond 
# and Fence have missing values
class_mod <- rpart(Electrical ~ . - SalePrice, data = train[!is.na(train$Electrical), ], method = "class", na.action = na.omit)
Electrical_pred <- predict(class_mod, train[is.na(train$Electrical),])
train$Electrical[is.na(train$Electrical)] <- "SBrkr"


anova_mod <- rpart(LotFrontage ~ . - SalePrice, data=train[!is.na(train$LotFrontage), ], method="anova", na.action=na.omit)  # since rad is a factor
LotFrontage_pred <- predict(anova_mod, train[is.na(train$LotFrontage),])
ptratio_pred <- predict(anova_mod, train[is.na(train$LotFrontage), ])
anova_mod <- rpart(LotFrontage ~ . - SalePrice, data=train[!is.na(train$LotFrontage), ], method="anova", na.action=na.omit)  # since ptratio is numeric
train$LotFrontage[is.na(train$LotFrontage)] <- LotFrontage_pred
train$LotFrontage <- round(train$LotFrontage, digit = 2)

class_mod <- rpart(MasVnrType ~ . - SalePrice, data = train[!is.na(train$MasVnrType), ], method = "class", na.action = na.omit)
MasVnrType_pred <- predict(class_mod, train[is.na(train$MasVnrType),])
a <- colnames(MasVnrType_pred)[max.col(MasVnrType_pred, ties.method = "first")]
a <- as.data.frame(a)
MasVnrType_pred <- cbind(MasVnrType_pred, a)
train$MasVnrType[is.na(train$MasVnrType)] <- MasVnrType_pred$a

class_mod <- rpart(gar_attach ~ . - SalePrice, data = train[!is.na(train$gar_attach), ], method = "class", na.action = na.omit)
MasVnrType_pred <- predict(class_mod, train[is.na(train$gar_attach),])
a <- colnames(MasVnrType_pred)[max.col(MasVnrType_pred, ties.method = "first")]
a <- as.data.frame(a)
MasVnrType_pred <- cbind(MasVnrType_pred, a)
train$gar_attach[is.na(train$gar_attach)] <- MasVnrType_pred$a

class_mod <- rpart(fire ~ . - SalePrice, data = train[!is.na(train$fire), ], method = "class", na.action = na.omit)
MasVnrType_pred <- predict(class_mod, train[is.na(train$fire),])
a <- colnames(MasVnrType_pred)[max.col(MasVnrType_pred, ties.method = "first")]
a <- as.data.frame(a)
MasVnrType_pred <- cbind(MasVnrType_pred, a)
train$fire[is.na(train$fire)] <- MasVnrType_pred$a

anova_mod <- rpart(MasVnrArea ~ . - SalePrice, data = train[!is.na(train$MasVnrArea), ], method = "anova", na.action = na.omit)
MasVnrArea_pred <- predict(anova_mod, train[is.na(train$MasVnrArea),])
train$MasVnrArea[is.na(train$MasVnrArea)] <- MasVnrArea_pred
train$MasVnrArea <- round(train$MasVnrArea, digits = 2)

class_mod <- rpart(BsmtQual ~ . - SalePrice, data = train[!is.na(train$BsmtQual), ], method = "class", na.action = na.omit)
BsmtQual_pred <- predict(class_mod, train[is.na(train$BsmtQual),])
a <- colnames(BsmtQual_pred)[max.col(BsmtQual_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtQual_pred <- cbind(BsmtQual_pred, a)
train$BsmtQual[is.na(train$BsmtQual)] <- BsmtQual_pred$a

class_mod <- rpart(BsmtCond ~ . - SalePrice, data = train[!is.na(train$BsmtCond), ], method = "class", na.action = na.omit)
BsmtCond_pred <- predict(class_mod, train[is.na(train$BsmtCond), ])
a <- colnames(BsmtCond_pred)[max.col(BsmtCond_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtCond_pred <- cbind(BsmtCond_pred, a)
train$BsmtCond[is.na(train$BsmtCond)] <- BsmtCond_pred$a

class_mod <- rpart(BsmtExposure ~ . - SalePrice, data = train[!is.na(train$BsmtExposure), ], method = "class", na.action = na.omit)
BsmtExp_pred <- predict(class_mod, train[is.na(train$BsmtExposure), ])
a <- colnames(BsmtExp_pred)[max.col(BsmtExp_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtExp_pred <- cbind(BsmtExp_pred, a)
train$BsmtExposure[is.na(train$BsmtExposure)] <- BsmtExp_pred$a

class_mod <- rpart(BsmtFinType1 ~ . - SalePrice, data = train[!is.na(train$BsmtFinType1), ], method = "class", na.action = na.omit)
BsmtExp_pred <- predict(class_mod, train[is.na(train$BsmtFinType1), ])
a <- colnames(BsmtExp_pred)[max.col(BsmtExp_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtExp_pred <- cbind(BsmtExp_pred, a)
train$BsmtFinType1[is.na(train$BsmtFinType1)] <- BsmtExp_pred$a

class_mod <- rpart(BsmtFinType2 ~ . - SalePrice, data = train[!is.na(train$BsmtFinType2), ], method = "class", na.action = na.omit)
BsmtExp_pred <- predict(class_mod, train[is.na(train$BsmtFinType2), ])
a <- colnames(BsmtExp_pred)[max.col(BsmtExp_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtExp_pred <- cbind(BsmtExp_pred, a)
train$BsmtFinType2[is.na(train$BsmtFinType2)] <- BsmtExp_pred$a

class_mod <- rpart(FireplaceQu ~ . - SalePrice, data = train[!is.na(train$FireplaceQu), ], method = "class", na.action = na.omit)
BsmtExp_pred <- predict(class_mod, train[is.na(train$FireplaceQu), ])
a <- colnames(BsmtExp_pred)[max.col(BsmtExp_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtExp_pred <- cbind(BsmtExp_pred, a)
train$FireplaceQu[is.na(train$FireplaceQu)] <- BsmtExp_pred$a

class_mod <- rpart(GarageType ~ . - SalePrice, data = train[!is.na(train$GarageType), ], method = "class", na.action = na.omit)
BsmtExp_pred <- predict(class_mod, train[is.na(train$GarageType), ])
a <- colnames(BsmtExp_pred)[max.col(BsmtExp_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtExp_pred <- cbind(BsmtExp_pred, a)
train$GarageType[is.na(train$GarageType)] <- BsmtExp_pred$a

anova_mod <- rpart(GarageYrBlt ~ . - SalePrice, data = train[!is.na(train$GarageYrBlt), ], method = "anova", na.action = na.omit)
MasVnrArea_pred <- predict(anova_mod, train[is.na(train$GarageYrBlt),])
train$GarageYrBlt[is.na(train$GarageYrBlt)] <- MasVnrArea_pred
train$GarageYrBlt <- round(train$GarageYrBlt, digits = 0)

# GarageFinish, GarageQual, GarageCond are all missing for 81 rows in the database
class_mod <- rpart(GarageFinish ~ . - SalePrice, data = train[!is.na(train$GarageFinish), ], method = "class", na.action = na.omit)
BsmtExp_pred <- predict(class_mod, train[is.na(train$GarageFinish), ])
a <- colnames(BsmtExp_pred)[max.col(BsmtExp_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtExp_pred <- cbind(BsmtExp_pred, a)
train$GarageFinish[is.na(train$GarageFinish)] <- BsmtExp_pred$a

class_mod <- rpart(GarageQual ~ . - SalePrice, data = train[!is.na(train$GarageQual), ], method = "class", na.action = na.omit)
BsmtExp_pred <- predict(class_mod, train[is.na(train$GarageQual), ])
a <- colnames(BsmtExp_pred)[max.col(BsmtExp_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtExp_pred <- cbind(BsmtExp_pred, a)
train$GarageQual[is.na(train$GarageQual)] <- BsmtExp_pred$a

class_mod <- rpart(GarageCond ~ . - SalePrice, data = train[!is.na(train$GarageCond), ], method = "class", na.action = na.omit)
BsmtExp_pred <- predict(class_mod, train[is.na(train$GarageCond), ])
a <- colnames(BsmtExp_pred)[max.col(BsmtExp_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtExp_pred <- cbind(BsmtExp_pred, a)
train$GarageCond[is.na(train$GarageCond)] <- BsmtExp_pred$a

summary(train)


table(train$LotShape)

LotShape <- summarise(group_by(train, LotShape), 
                      mean(SalePrice, na.rm=T))
train$regshape[train$LotShape == "Reg"] <- 1
train$regshape[train$LotShape != "Reg"] <- 0


# table(train$LandContour)
LandContour <- summarize(group_by(train, LandContour), mean(SalePrice, na.rm=T))
LandContour

train$flat[train$LandContour == "Lvl"] <- 1
train$flat[train$LandContour == "Bnk"] <- 0
train$flat[train$LandContour == "Low"] <- 2
train$flat[train$LandContour == "HLS"] <- 3

train$gentle_slope[train$LandSlope == "Gtl"] <- 1
train$gentle_slope[train$LandSlope != "Gtl"] <- 0

# lotconfig <- summarize(group_by(train, LotConfig), mean(SalePrice, na.rm=T))
# lotconfig
train$culdesac_fr3[train$LotConfig %in% c("CulDSac", "FR3")] <- 1
train$culdesac_fr3[!train$LotConfig %in% c("CulDSac", "FR3")] <- 0

nbhdprice <- summarize(group_by(train, Neighborhood), mean(SalePrice, na.rm=T))

#nbhdprice[order(nbhdprice$`mean(SalePrice, na.rm = T)`),]

nbhdprice_lo <- filter(nbhdprice, nbhdprice$`mean(SalePrice, na.rm = T)` < 140000)
nbhdprice_med <- filter(nbhdprice, nbhdprice$`mean(SalePrice, na.rm = T)` < 180000 &
                          nbhdprice$`mean(SalePrice, na.rm = T)` >= 140000 )
nbhdprice_hi <- filter(nbhdprice, nbhdprice$`mean(SalePrice, na.rm = T)` >= 180000 & nbhdprice$`mean(SalePrice, na.rm = T)` < 300000)
nbhdprice_suphi <- filter(nbhdprice, nbhdprice$`mean(SalePrice, na.rm = T)` >= 300000)

train$nbhd_price_level[train$Neighborhood %in% nbhdprice_lo$Neighborhood] <- 1
train$nbhd_price_level[train$Neighborhood %in% nbhdprice_med$Neighborhood] <- 2
train$nbhd_price_level[train$Neighborhood %in% nbhdprice_hi$Neighborhood] <- 3
train$nbhd_price_level[train$Neighborhood %in% nbhdprice_suphi$Neighborhood] <- 4


cond1 <- summarize(group_by(train, Condition1), mean(SalePrice, na.rm=T))


train$pos_features_1[train$Condition1 %in% c("PosA", "PosN", "RRNn")] <- 2
train$pos_features_1[!train$Condition1 %in% c("PosA", "PosN", "RRNn", "Feedr", "RRAe", "Artery")] <- 1
train$pos_features_1[train$Condition1 %in% c("Feedr", "RRAe", "Artery")] <- 0

# cond2 <- summarize(group_by(train, Condition2), mean(SalePrice, na.rm=T))

train$pos_features_2[!train$Condition1 %in% c("Norm", "RRAe", "PosA", "PosN")] <- 0
train$pos_features_2[train$Condition1 %in% c("Norm", "RRAe")] <- 1
train$pos_features_2[train$Condition1 %in% c("PosA", "PosN")] <- 2


# summarize(group_by(train, BldgType), mean(SalePrice, na.rm=T))

train$twnhs_end_or_1fam[train$BldgType %in% c("1Fam", "TwnhsE")] <- 1
train$twnhs_end_or_1fam[!train$BldgType %in% c("1Fam", "TwnhsE")] <- 0

housestyle_price <- summarize(group_by(train, HouseStyle), mean(SalePrice, na.rm=T))

housestyle_lo <- filter(housestyle_price, housestyle_price$`mean(SalePrice, na.rm = T)` < 150000)
housestyle_med <- filter(housestyle_price, housestyle_price$`mean(SalePrice, na.rm = T)` < 200000 &
                           housestyle_price$`mean(SalePrice, na.rm = T)` >= 150000 )
housestyle_hi <- filter(housestyle_price, housestyle_price$`mean(SalePrice, na.rm = T)` >= 200000)

train$house_style_level[train$HouseStyle %in% housestyle_lo$HouseStyle] <- 1
train$house_style_level[train$HouseStyle %in% housestyle_med$HouseStyle] <- 2
train$house_style_level[train$HouseStyle %in% housestyle_hi$HouseStyle] <- 3


# roofstyle_price <- summarize(group_by(train, RoofStyle), mean(SalePrice, na.rm=T))
# Could be more catalogies?
train$roof_hip_shed[train$RoofStyle %in% c("Hip", "Shed", "Flat")] <- 1
train$roof_hip_shed[!train$RoofStyle %in% c("Hip", "Shed", "Flat")] <- 0

price <- summarize(group_by(train, Exterior1st), mean(SalePrice, na.rm=T))

matl_lo_1 <- filter(price, price$`mean(SalePrice, na.rm = T)` < 140000)
matl_med_1<- filter(price, price$`mean(SalePrice, na.rm = T)` < 200000 &
                      price$`mean(SalePrice, na.rm = T)` >= 140000 )
matl_hi_1 <- filter(price, price$`mean(SalePrice, na.rm = T)` >= 200000)

train$exterior_1[train$Exterior1st %in% matl_lo_1$Exterior1st] <- 1
train$exterior_1[train$Exterior1st %in% matl_med_1$Exterior1st] <- 2
train$exterior_1[train$Exterior1st %in% matl_hi_1$Exterior1st] <- 3


price <- summarize(group_by(train, Exterior2nd), mean(SalePrice, na.rm=T))

matl_lo <- filter(price, price$`mean(SalePrice, na.rm = T)` < 140000)
matl_med <- filter(price, price$`mean(SalePrice, na.rm = T)` < 190000 &
                     price$`mean(SalePrice, na.rm = T)` >= 140000 )
matl_hi <- filter(price, price$`mean(SalePrice, na.rm = T)` >= 190000 & price$`mean(SalePrice, na.rm = T)`<300000)
matl_suphi <- filter(price, price$`mean(SalePrice, na.rm = T)` > 300000)

train$exterior_2[train$Exterior2nd %in% matl_lo$Exterior2nd] <- 1
train$exterior_2[train$Exterior2nd %in% matl_med$Exterior2nd] <- 2
train$exterior_2[train$Exterior2nd %in% matl_hi$Exterior2nd] <- 3
train$exterior_2[train$Exterior2nd %in% matl_suphi$Exterior2nd] <- 4


price <- summarize(group_by(train, MasVnrType),
                   mean(SalePrice, na.rm=T))
train$exterior_mason_1[train$MasVnrType %in% c("Stone", "BrkFace") | is.na(train$MasVnrType)] <- 1
train$exterior_mason_1[!train$MasVnrType %in% c("Stone", "BrkFace") & !is.na(train$MasVnrType)] <- 0


price <- summarize(group_by(train, ExterQual),
                   mean(SalePrice, na.rm=T))

train$exterior_cond[train$ExterQual == "Ex"] <- 4
train$exterior_cond[train$ExterQual == "Gd"] <- 3
train$exterior_cond[train$ExterQual == "TA"] <- 2
train$exterior_cond[train$ExterQual == "Fa"] <- 1


# price <- summarize(group_by(train, ExterCond), mean(SalePrice, na.rm=T))

train$exterior_cond2[train$ExterCond == "Ex"] <- 5
train$exterior_cond2[train$ExterCond == "Gd"] <- 4
train$exterior_cond2[train$ExterCond == "TA"] <- 3
train$exterior_cond2[train$ExterCond == "Fa"] <- 2
train$exterior_cond2[train$ExterCond == "Po"] <- 1

# price <- summarize(group_by(train, Foundation), mean(SalePrice, na.rm=T))

train$found_concrete[train$Foundation == "PConc"] <- 2
train$found_concrete[train$Foundation == "Slab"] <- 0
train$found_concrete[!train$Foundation %in% c("PConc", "Slab")] <- 1


# price <- summarize(group_by(train, BsmtQual), mean(SalePrice, na.rm=T))

train$bsmt_cond1[train$BsmtQual == "Ex"] <- 5
train$bsmt_cond1[train$BsmtQual == "Gd"] <- 4
train$bsmt_cond1[train$BsmtQual == "TA"] <- 3
train$bsmt_cond1[train$BsmtQual == "Fa"] <- 2
train$bsmt_cond1[is.na(train$BsmtQual)] <- 1


# price <- summarize(group_by(train, BsmtCond), mean(SalePrice, na.rm=T))

train$bsmt_cond2[train$BsmtCond == "Gd"] <- 5
train$bsmt_cond2[train$BsmtCond == "TA"] <- 4
train$bsmt_cond2[train$BsmtCond == "Fa"] <- 3
train$bsmt_cond2[is.na(train$BsmtCond)] <- 2
train$bsmt_cond2[train$BsmtCond == "Po"] <- 1


# price <- summarize(group_by(train, BsmtExposure), mean(SalePrice, na.rm=T))

train$bsmt_exp[train$BsmtExposure == "Gd"] <- 5
train$bsmt_exp[train$BsmtExposure == "Av"] <- 4
train$bsmt_exp[train$BsmtExposure == "Mn"] <- 3
train$bsmt_exp[train$BsmtExposure == "No"] <- 2
train$bsmt_exp[is.na(train$BsmtExposure)] <- 1


# price <- summarize(group_by(train, BsmtFinType1), mean(SalePrice, na.rm=T))

train$bsmt_fin1[train$BsmtFinType1 == "GLQ"] <- 5
train$bsmt_fin1[train$BsmtFinType1 == "Unf"] <- 4
train$bsmt_fin1[train$BsmtFinType1 == "ALQ"] <- 3
train$bsmt_fin1[train$BsmtFinType1 %in% c("BLQ", "Rec", "LwQ")] <- 2
train$bsmt_fin1[is.na(train$BsmtFinType1)] <- 1



# price <- summarize(group_by(train, BsmtFinType2), mean(SalePrice, na.rm=T))

train$bsmt_fin2[train$BsmtFinType2 == "ALQ"] <- 6
train$bsmt_fin2[train$BsmtFinType2 == "Unf"] <- 5
train$bsmt_fin2[train$BsmtFinType2 == "GLQ"] <- 4
train$bsmt_fin2[train$BsmtFinType2 %in% c("Rec", "LwQ")] <- 3
train$bsmt_fin2[train$BsmtFinType2 == "BLQ"] <- 2
train$bsmt_fin2[is.na(train$BsmtFinType2)] <- 1

# price <- summarize(group_by(train, Heating), mean(SalePrice, na.rm=T))


train$gasheat[train$Heating %in% c("GasA", "GasW")] <- 2
train$gasheat[train$Heating %in% c("Wall", "OthW")] <- 1
train$gasheat[!train$Heating %in% c("GasA", "GasW", "Wall", "Othw")] <- 0


# price <- summarize(group_by(train, HeatingQC), mean(SalePrice, na.rm=T))

train$heatqual[train$HeatingQC == "Ex"] <- 5
train$heatqual[train$HeatingQC == "Gd"] <- 4
train$heatqual[train$HeatingQC == "TA"] <- 3
train$heatqual[train$HeatingQC == "Fa"] <- 2
train$heatqual[train$HeatingQC == "Po"] <- 1


# Only two potential values
train$air[train$CentralAir == "Y"] <- 1
train$air[train$CentralAir == "N"] <- 0


# price <- summarize(group_by(train, Electrical), mean(SalePrice, na.rm=T))

# Only one NA and One Mix which has the lowest price. So l just let it go
train$standard_electric[train$Electrical == "SBrkr" | is.na(train$Electrical)] <- 1
train$standard_electric[!train$Electrical == "SBrkr" & !is.na(train$Electrical)] <- 0


# price <- summarize(group_by(train, KitchenQual), mean(SalePrice, na.rm=T))

train$kitchen[train$KitchenQual == "Ex"] <- 4
train$kitchen[train$KitchenQual == "Gd"] <- 3
train$kitchen[train$KitchenQual == "TA"] <- 2
train$kitchen[train$KitchenQual == "Fa"] <- 1


price <- summarize(group_by(train, FireplaceQu), mean(SalePrice, na.rm=T))

train$fire[train$FireplaceQu == "Ex"] <- 5
train$fire[train$FireplaceQu == "Gd"] <- 4
train$fire[train$FireplaceQu == "TA"] <- 3
train$fire[train$FireplaceQu == "Fa"] <- 2
train$fire[train$FireplaceQu == "Po"] <- 1



price <- summarize(group_by(train, GarageType), mean(SalePrice, na.rm=T))
train$gar_attach[train$GarageType == "CarPort"] <- 0
train$gar_attach[train$GarageType == "Detchd"] <- 1
train$gar_attach[train$GarageType %in% c("2Types", "Basment")] <- 2
train$gar_attach[train$GarageType == "Attchd"] <- 3
train$gar_attach[train$GarageType == "BuiltIn"] <- 4

# can be more detailed
# price <- summarize(group_by(train, GarageFinish), mean(SalePrice, na.rm=T))

train$gar_finish[train$GarageFinish %in% c("Fin", "RFn")] <- 1
train$gar_finish[!train$GarageFinish %in% c("Fin", "RFn")] <- 0


price <- summarize(group_by(train, GarageQual), mean(SalePrice, na.rm=T))

train$garqual[train$GarageQual == "Ex"] <- 5
train$garqual[train$GarageQual == "Gd"] <- 4
train$garqual[train$GarageQual == "TA"] <- 3
train$garqual[train$GarageQual == "Fa"] <- 2
train$garqual[train$GarageQual == "Po" | is.na(train$GarageQual)] <- 1


# price <- summarize(group_by(train, GarageCond), mean(SalePrice, na.rm=T))

train$garqual2[train$GarageCond == "Ex"] <- 5
train$garqual2[train$GarageCond == "Gd"] <- 4
train$garqual2[train$GarageCond == "TA"] <- 3
train$garqual2[train$GarageCond == "Fa"] <- 2
train$garqual2[train$GarageCond == "Po" | is.na(train$GarageCond)] <- 1


price <- summarize(group_by(train, PavedDrive), mean(SalePrice, na.rm=T))

train$paved_drive[train$PavedDrive == "Y"] <- 2
train$paved_drive[train$PavedDrive == "N"] <- 0
train$paved_drive[train$PavedDrive == "P"] <- 1


# price <- summarize(group_by(train, Functional), mean(SalePrice, na.rm=T))

train$housefunction[train$Functional %in% c("Typ", "Mod")] <- 2
train$housefunction[train$Functional %in% c("Maj1", "Min1", "Min2")] <- 1
train$housefunction[train$Functional %in% c("Sev", "Maj2")] <- 0


# price <- summarize(group_by(train, SaleType), mean(SalePrice, na.rm=T))

train$sale_cat[train$SaleType %in% c("New", "Con")] <- 5
train$sale_cat[train$SaleType %in% c("CWD", "ConLI")] <- 4
train$sale_cat[train$SaleType %in% c("WD")] <- 3
train$sale_cat[train$SaleType %in% c("COD", "ConLw", "ConLD")] <- 2
train$sale_cat[train$SaleType %in% c("Oth")] <- 1


price <- summarize(group_by(train, SaleCondition), mean(SalePrice, na.rm=T))

# price[order(price$`mean(SalePrice, na.rm = T)`),]

train$sale_cond[train$SaleCondition %in% c("Partial")] <- 4
train$sale_cond[train$SaleCondition %in% c("Normal", "Alloca")] <- 3
train$sale_cond[train$SaleCondition %in% c("Family","Abnorml")] <- 2
train$sale_cond[train$SaleCondition %in% c("AdjLand")] <- 1


# price <- summarize(group_by(train, MSZoning), mean(SalePrice, na.rm=T))
train$zone[train$MSZoning %in% c("FV")] <- 4
train$zone[train$MSZoning %in% c("RL")] <- 3
train$zone[train$MSZoning %in% c("RH","RM")] <- 2
train$zone[train$MSZoning %in% c("C (all)")] <- 1

train$Street <- NULL
train$LotShape <- NULL
train$LandContour <- NULL
train$Utilities <- NULL
train$LotConfig <- NULL
train$LandSlope <- NULL
train$Neighborhood <- NULL
train$Condition1 <- NULL
train$Condition2 <- NULL
train$BldgType <- NULL
train$HouseStyle <- NULL
train$RoofStyle <- NULL
train$RoofMatl <- NULL

train$Exterior1st <- NULL
train$Exterior2nd <- NULL
train$MasVnrType <- NULL
train$ExterQual <- NULL
train$ExterCond <- NULL

train$Foundation <- NULL
train$BsmtQual <- NULL
train$BsmtCond <- NULL
train$BsmtExposure <- NULL
train$BsmtFinType1 <- NULL
train$BsmtFinType2 <- NULL

train$Heating <- NULL
train$HeatingQC <- NULL
train$CentralAir <- NULL
train$Electrical <- NULL
train$KitchenQual <- NULL
train$FireplaceQu <- NULL

train$GarageType <- NULL
train$GarageFinish <- NULL
train$GarageQual <- NULL
train$GarageCond <- NULL
train$PavedDrive <- NULL

train$Functional <- NULL
train$PoolQC <- NULL
train$Fence <- NULL
train$MiscFeature <- NULL
train$SaleType <- NULL
train$SaleCondition <- NULL
train$MSZoning <- NULL
train$Alley <- NULL

train$Id <- NULL
train <- train[-c(1299), ]

correlations <- cor(train[,c(2,3,4,5,6,7,8)], use="everything")
corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")

correlations <- cor(train[,c(5,6,7,8,9,10,11,12,13,14)], use="everything")
corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")

correlations <- cor(train[,c(26:35, 16:25)], use="everything")
corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")

correlations <- cor(train[,c(5,6,7,8, 16:25)], use="everything")
corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")

correlations <- cor(train[,c(5,6,7,8, 26:35)], use="everything")
corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")

correlations <- cor(train[,c(5,6,7,8, 36:45)], use="everything")
corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")

correlations <- cor(train[,c(5,6,7,8, 66:73)], use="everything")
corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")

pairs(~YearBuilt+OverallQual+TotalBsmtSF+GrLivArea,data=train,
      main="Simple Scatterplot Matrix")


library(car)

scatterplot(SalePrice ~ YearBuilt, data=train,  xlab="Year Built", ylab="Sale Price", grid=FALSE)
scatterplot(SalePrice ~ YrSold, data=train,  xlab="Year Sold", ylab="Sale Price", grid=FALSE)
scatterplot(SalePrice ~ FlrSF1st, data=train,  xlab="Square Footage Floor 1", ylab="Sale Price", grid=FALSE)

# A backup datatable without interaction terms
backup <- train
# return to the original datatable whenever you want
train <- backup

#Interactions based on correlation
train$year_qual <- train$YearBuilt*train$OverallQual #overall condition
train$year_r_qual <- train$YearRemodAdd*train$OverallQual #quality x remodel
train$qual_bsmt <- train$OverallQual*train$TotalBsmtSF #quality x basement size
train$livarea_qual <- train$OverallQual*train$GrLivArea #quality x living area
train$qual_bath <- train$OverallQual*train$FullBath #quality x baths
train$qual_ext <- train$OverallQual*train$exterior_cond #quality x exterior
train$car_area <- train$GarageCars*train$GarageArea # my try

ROOT.DIR <- ".."

ID.VAR <- "Id"
TARGET.VAR <- "SalePrice"

# extract only candidate feature names
candidate.features <- setdiff(names(train),c(ID.VAR,TARGET.VAR))
data.type <- sapply(candidate.features,function(x){class(train[[x]])})
table(data.type)

print(data.type)

# deterimine data types
explanatory.attributes <- setdiff(names(train),c(ID.VAR,TARGET.VAR))
data.classes <- sapply(explanatory.attributes,function(x){class(train[[x]])})

# categorize data types in the data set?
unique.classes <- unique(data.classes)

attr.data.types <- lapply(unique.classes,function(x){names(data.classes[data.classes==x])})
names(attr.data.types) <- unique.classes

# pull out the response variable
response <- train$SalePrice

# remove identifier and response variables
train <- train[candidate.features]

set.seed(13)
bor.results <- Boruta(train,response,
                      maxRuns=101,
                      doTrace=0)

### Boruta results
cat("\nSummary of Boruta run:\n")
print(bor.results)

cat("\n\nRelevant Attributes:\n")
getSelectedAttributes(bor.results)

#The following plot shows the relative importance of each candidate explanatory attribute.
#The x-axis represents each of candidate explanatory variables.  Green color indicates
#the attributes that are relevant to prediction.  Red indicates attributes that
#are not relevant.  Yellow color indicates attributes that may or may not be relevant to 
#predicting the response variable.
plot(bor.results)

cat("\n\nAttribute Importance Details:\n")
options(width=125)
arrange(cbind(attr=rownames(attStats(bor.results)), attStats(bor.results)),desc(medianImp))

CONFIRMED_ATTR <- c("MSSubClass",        "LotFrontage",       "LotArea",           "OverallQual",      
                    "OverallCond",       "YearBuilt",         "YearRemodAdd",      "MasVnrArea",       
                    "BsmtFinSF1",        "BsmtUnfSF",         "TotalBsmtSF",       "FlrSF1st",         
                    "FlrSF2nd",          "GrLivArea",         "BsmtFullBath",      "FullBath",        
                    "HalfBath",          "BedroomAbvGr",      "KitchenAbvGr",      "TotRmsAbvGrd",     
                    "Fireplaces",        "GarageYrBlt",       "GarageCars",        "GarageArea",      
                    "WoodDeckSF",        "OpenPorchSF",       "regshape",          "gentle_slope",     
                    "nbhd_price_level",  "twnhs_end_or_1fam", "house_style_level", "exterior_1",       
                    "exterior_2",        "exterior_cond",     "found_concrete",    "bsmt_cond1",      
                    "bsmt_exp",          "bsmt_fin1",         "heatqual",          "air",              
                    "kitchen",           "gar_attach",        "gar_finish",        "paved_drive",      
                    "housefunction",     "zone",              "year_qual",         "year_r_qual",      
                    "qual_bsmt",         "livarea_qual",      "qual_bath",         "qual_ext",         
                    "car_area"          )

TENTATIVE_ATTR <- c("exterior_mason_1", "roof_hip_shed", "two_pos_feature", "sale_cat", 
                    "bsmt_cond2")
REJECTED_ATTR <- c("standard_electric", "BsmtFinSF2", "fire", "flat", "garqual", "ScreenPorch", "EnclosedPorch", 
                   "culdesac_fr3", "sale_cond", "pos_features_2", "gasheat", "bsmt_fin2", "BsmtHalfBath",
                   "garqual2", "PoolArea", "exterior_cond2", "YrSold", "SsnPorch3", "MoSold", "MiscVal", "LowQualFinSF")

PREDICTOR_ATTR <- c(CONFIRMED_ATTR,TENTATIVE_ATTR,REJECTED_ATTR)

# Determine data types in the data set
data_types <- sapply(PREDICTOR_ATTR,function(x){class(train[[x]])})
unique_data_types <- unique(data_types)

# Separate attributes by data type
DATA_ATTR_TYPES <- lapply(unique_data_types,function(x){ names(data_types[data_types == x])})
names(DATA_ATTR_TYPES) <- unique_data_types


# create folds for training
set.seed(13)
data_folds <- createFolds(train$SalePrice, k=5)

train <- train[-c(1299),]
train <- train[-c(1325),]
train <- train[-c(524),]
train <- train[-c(1325),]
train <- train[-c(1325),]

## Model Prep ##

set.seed(123)

sample = sample.split(train$OverallQual, SplitRatio = 0.75)
training = subset(train, sample == TRUE)
testing  = subset(train, sample == FALSE)


###A Linear Model
lm_model_15 <- lm( SalePrice ~ . -TotalBsmtSF - GrLivArea - YearBuilt - YrSold - total_1st -
                     Remod_Built - fire - gar_attach - gar_finish - gasheat - heatqual - 
                     air - standard_electric - , data=training)
summary(lm_model_15)

par(mfrow = c(2,2))  # Plot 4 charts in one plot - 2 by 2.
plot(lm_model_15)# Plot model 1 diagnostics
par(mfrow = c(1,1))  # Reset plot options to 1 chart in one plot.


# RMSE of linear regression model
prediction <- predict(lm_model_15, testing, type="response")
model_output <- cbind(testing, prediction)

model_output$log_prediction <- log(model_output$prediction)
model_output$log_SalePrice <- log(model_output$SalePrice)

#Test with RMSE
#0.157 ,0.143 after dropping 1325 twice, 0.111 after dropping 1325 three times.
# What is wrong with 1325??????
rmse(model_output$log_SalePrice,model_output$log_prediction)



###A Random Forest

model_1_RF <- randomForest(SalePrice ~ . , data=training, mtry = 27, ntree = 500, maxnodes = 43)
# varImpPlot(model_1_RF)

model_2_RF <- randomForest(SalePrice ~ MSSubClass + LotFrontage + LotArea + OverallQual
                           + OverallCond + YearBuilt + YearRemodAdd + MasVnrArea + BsmtFinSF1
                           + BsmtUnfSF + TotalBsmtSF + FlrSF1st + FlrSF2nd + GrLivArea + BsmtFullBath
                           + FullBath + HalfBath + BedroomAbvGr + KitchenAbvGr + TotRmsAbvGrd 
                           + Fireplaces + GarageYrBlt + GarageCars + GarageArea + WoodDeckSF
                           + OpenPorchSF + regshape + gentle_slope + nbhd_price_level + twnhs_end_or_1fam
                           + house_style_level + exterior_1 + exterior_2 + exterior_cond + found_concrete
                           + bsmt_cond1 + bsmt_exp + bsmt_fin1 + heatqual + air + kitchen +gar_attach
                           + gar_finish + paved_drive + housefunction + zone + year_qual + year_r_qual      
                           + qual_bsmt + livarea_qual + qual_bath + qual_ext + car_area, data = training,
                           mtry = 18, ntree = 500, maxnodes = 43)


# Predict using the test set
prediction <- predict(model_1_RF, testing)
model_output <- cbind(testing, prediction)

model_output$log_prediction <- log(model_output$prediction)
model_output$log_SalePrice <- log(model_output$SalePrice)

#Test with RMSE
#0.15 0.115 if l drop the 5 data
rmse(model_output$log_SalePrice,model_output$log_prediction)

# Predict using the test set
prediction <- predict(model_2_RF, testing)
model_output <- cbind(testing, prediction)

model_output$log_prediction <- log(model_output$prediction)
model_output$log_SalePrice <- log(model_output$SalePrice)

#Test with RMSE
#0.15 0.115 if l drop the 5 data
rmse(model_output$log_SalePrice,model_output$log_prediction)

# Train the random forest model
# Define the control
trControl <- trainControl(method = "cv", number = 10, search = "grid")
# - cv means cross validation method will be used in this case
# number: number of folders to be used
# search: use the search grid method

set.seed(1234)
# Run the model
rf_default <- train(SalePrice ~ .,
                    data = training,
                    method = "rf", # the abbreviation of random forest
                    metric = "RMSE",
                    trControl = trControl)
# Print the results
print(rf_default)

set.seed(1234)
tuneGrid <- expand.grid(.mtry = c(15: 25))
rf_mtry <- train(SalePrice ~.,
                 data = training,
                 method = "rf",
                 metric = "RMSE",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE,
                 nodesize = 14,
                 ntree = 600)
print(rf_mtry)
best_mtry = rf_mtry$bestTune$mtry # which is 27
best_rmse = max(rf_mtry$results$RMSE)

store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(5: 15)) {
  set.seed(1234)
  rf_maxnode <- train(SalePrice ~.,
                      data = training,
                      method = "rf",
                      metric = "RMSE",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 600)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)
for (maxnodes in c(16: 25)) {
  set.seed(1234)
  rf_maxnode <- train(SalePrice ~.,
                      data = training,
                      method = "rf",
                      metric = "RMSE",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 600)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)
for (maxnodes in c(46: 55)) {
  set.seed(1234)
  rf_maxnode <- train(SalePrice ~.,
                      data = training,
                      method = "rf",
                      metric = "RMSE",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 600)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)
# l would like to choose 43

store_maxtrees <- list()
for (ntree in c(300, 350, 400, 450, 500, 550, 600, 800, 1000, 1500)) {
  set.seed(5678)
  rf_maxtrees <- train(SalePrice~.,
                       data = training,
                       method = "rf",
                       metric = "RMSE",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       importance = TRUE,
                       nodesize = 14,
                       maxnodes = 43,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)


# xgboost part
training1 <- training
testing1 <- testing
training1$log_SalePrice <- log(training$SalePrice)
testing1$log_SalePrice <- log(testing$SalePrice)

#Create matrices from the data frames
trainData<- as.matrix(training1, rownames.force=NA)
testData<- as.matrix(testing1, rownames.force=NA)

#Turn the matrices into sparse matrices
train2 <- as(trainData, "sparseMatrix")
test2 <- as(testData, "sparseMatrix")

#####
#colnames(train2)
#Cross Validate the model

vars <- c(1:36, 38:80) #choose the columns we want to use in the prediction matrix

trainD <- xgb.DMatrix(data = train2[,vars], label = train2[,"SalePrice"]) #Convert to xgb.DMatrix format

param <- list(colsample_bytree = .7,
              subsample = .7,
              booster = "gbtree",
              max_depth = 10,
              eta = 0.01,
              eval_metric = "rmse",
              objective="reg:linear")


#Train the model using those parameters
bstSparse <-
  xgb.train(params = param,
            data = trainD,
            nrounds = 800,
            watchlist = list(train = trainD),
            verbose = TRUE,
            print_every_n = 50,
            nthread = 2)

testD <- xgb.DMatrix(data = test2[,vars])
#Column names must match the inputs EXACTLY
prediction <- predict(bstSparse, testD) #Make the prediction based on the half of the training data set aside

#Put testing prediction and test dataset all together
test3 <- as.data.frame(as.matrix(test2))
prediction <- as.data.frame(as.matrix(prediction))
colnames(prediction) <- "prediction"
model_output <- cbind(test3, prediction)

model_output$log_prediction <- log(model_output$prediction)
model_output$log_SalePrice <- log(model_output$SalePrice)

#Test with RMSE

rmse(model_output$log_SalePrice,model_output$log_prediction)
# 0.1453  0.107 if l drop 5 data or add two features

library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # a java-based platform
library(pdp)          # model visualization
library(ggplot2)      # model visualization
library(lime)         # model visualization
set.seed(123)

# train GBM model
gbm.fit <- gbm(
  formula = SalePrice ~ .,
  distribution = "gaussian",
  data = training,
  n.trees = 10000,
  interaction.depth = 1,
  shrinkage = 0.001,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  
# print results
print(gbm.fit)

# get MSE and compute RMSE
sqrt(min(gbm.fit$cv.error))
## [1] 29133.33

# plot loss function as a result of n trees added to the ensemble
# 1000
gbm.perf(gbm.fit, method = "cv")


# for reproducibility
set.seed(123)

# train GBM model
gbm.fit2 <- gbm(
  formula = SalePrice ~ .,
  distribution = "gaussian",
  data = training,
  n.trees = 5000,
  interaction.depth = 3,
  shrinkage = 0.1,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  

# find index for n trees with minimum CV error
min_MSE <- which.min(gbm.fit2$cv.error)

# get MSE and compute RMSE
sqrt(gbm.fit2$cv.error[min_MSE])
## [1] 23112.1

# plot loss function as a result of n trees added to the ensemble
gbm.perf(gbm.fit2, method = "cv")

hyper_grid <- expand.grid(
  shrinkage = c(.01, 0.02, .025),
  interaction.depth = c(4, 5),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(.65, 0.67, .7), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid)

# randomize data
random_index <- sample(1:nrow(training), nrow(training))
random_train <- training[random_index, ]

# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune <- gbm(
    formula = SalePrice ~ .,
    distribution = "gaussian",
    data = training,
    n.trees = 5000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$valid.error))
}

hyper_grid %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)
#shrinkage 0.01 interaction.depth 5 n.minobsinnode 5, bag.fraction 0.67 optimal_trees 3234

# submission part
test_pred <- fread("test.csv")

colnames(test_pred)[colnames(test_pred) =="1stFlrSF"] <- "FlrSF1st"
colnames(test_pred)[colnames(test_pred) =="2ndFlrSF"] <- "FlrSF2nd"
colnames(test_pred)[colnames(test_pred) =="3SsnPorch"] <- "SsnPorch3"

test_pred = test_pred %>% mutate_if(is.character, as.factor)

test_pred$Utilities<- NULL # 1400+ Pave No big use
test_pred$RoofMatl<- NULL # 1400+ Pave No big use
test_pred$MiscFeature <- NULL # 1400+ Pave No big use
test_pred$Alley <- NULL # too many missing values
test_pred$PoolQC <- NULL # too many missing values
test_pred$Fence <- NULL # too many missing values, should l impute it or delete it

summary(test_pred)
which(apply(test_pred, 2, function(x) any(is.na(x)))) # to see if there are any missing values
colSums(is.na(test_pred))

class_mod <- rpart(Electrical ~ ., data = test_pred[!is.na(test_pred$Electrical), ], method = "class", na.action = na.omit)
Electrical_pred <- predict(class_mod, test_pred[is.na(test_pred$Electrical),])
test_pred$Electrical[is.na(test_pred$Electrical)] <- "SBrkr"

class_mod <- rpart(MSZoning ~ . , data = test_pred[!is.na(test_pred$MSZoning), ], method = "class", na.action = na.omit)
MSZoning_pred <- predict(class_mod, test_pred[is.na(test_pred$MSZoning),])
a <- colnames(MSZoning_pred)[max.col(MSZoning_pred, ties.method = "first")]
a <- as.data.frame(a)
MSZoning_pred <- cbind(MSZoning_pred, a)
test_pred$MSZoning[is.na(test_pred$MSZoning)] <- MSZoning_pred$a


anova_mod <- rpart(LotFrontage ~ ., data=test_pred[!is.na(train$LotFrontage), ], method="anova", na.action=na.omit)  # since rad is a factor
LotFrontage_pred <- predict(anova_mod, test_pred[is.na(test_pred$LotFrontage),])
#ptratio_pred <- predict(anova_mod, BostonHousing[is.na(BostonHousing$ptratio), ])
#anova_mod <- rpart(ptratio ~ . - medv, data=BostonHousing[!is.na(BostonHousing$ptratio), ], method="anova", na.action=na.omit)  # since ptratio is numeric
test_pred$LotFrontage[is.na(test_pred$LotFrontage)] <- LotFrontage_pred
test_pred$LotFrontage <- round(test_pred$LotFrontage, digit = 2)

class_mod <- rpart(MasVnrType ~ . , data = test_pred[!is.na(test_pred$MasVnrType), ], method = "class", na.action = na.omit)
MasVnrType_pred <- predict(class_mod, test_pred[is.na(test_pred$MasVnrType),])
a <- colnames(MasVnrType_pred)[max.col(MasVnrType_pred, ties.method = "first")]
a <- as.data.frame(a)
MasVnrType_pred <- cbind(MasVnrType_pred, a)
test_pred$MasVnrType[is.na(test_pred$MasVnrType)] <- MasVnrType_pred$a

anova_mod <- rpart(MasVnrArea ~ . , data = test_pred[!is.na(test_pred$MasVnrArea), ], method = "anova", na.action = na.omit)
MasVnrArea_pred <- predict(anova_mod, test_pred[is.na(test_pred$MasVnrArea),])
test_pred$MasVnrArea[is.na(test_pred$MasVnrArea)] <- MasVnrArea_pred
test_pred$MasVnrArea <- round(test_pred$MasVnrArea, digits = 2)

class_mod <- rpart(BsmtQual ~ . , data = test_pred[!is.na(test_pred$BsmtQual), ], method = "class", na.action = na.omit)
BsmtQual_pred <- predict(class_mod, test_pred[is.na(test_pred$BsmtQual),])
a <- colnames(BsmtQual_pred)[max.col(BsmtQual_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtQual_pred <- cbind(BsmtQual_pred, a)
test_pred$BsmtQual[is.na(test_pred$BsmtQual)] <- BsmtQual_pred$a

class_mod <- rpart(BsmtCond ~ . , data = test_pred[!is.na(test_pred$BsmtCond), ], method = "class", na.action = na.omit)
BsmtCond_pred <- predict(class_mod, test_pred[is.na(test_pred$BsmtCond), ])
a <- colnames(BsmtCond_pred)[max.col(BsmtCond_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtCond_pred <- cbind(BsmtCond_pred, a)
test_pred$BsmtCond[is.na(test_pred$BsmtCond)] <- BsmtCond_pred$a

class_mod <- rpart(BsmtExposure ~ . , data = test_pred[!is.na(test_pred$BsmtExposure), ], method = "class", na.action = na.omit)
BsmtExp_pred <- predict(class_mod, test_pred[is.na(test_pred$BsmtExposure), ])
a <- colnames(BsmtExp_pred)[max.col(BsmtExp_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtExp_pred <- cbind(BsmtExp_pred, a)
test_pred$BsmtExposure[is.na(test_pred$BsmtExposure)] <- BsmtExp_pred$a

class_mod <- rpart(BsmtFinType1 ~ . , data = test_pred[!is.na(test_pred$BsmtFinType1), ], method = "class", na.action = na.omit)
BsmtExp_pred <- predict(class_mod, test_pred[is.na(test_pred$BsmtFinType1), ])
a <- colnames(BsmtExp_pred)[max.col(BsmtExp_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtExp_pred <- cbind(BsmtExp_pred, a)
test_pred$BsmtFinType1[is.na(test_pred$BsmtFinType1)] <- BsmtExp_pred$a

class_mod <- rpart(BsmtFinType2 ~ . , data = test_pred[!is.na(test_pred$BsmtFinType2), ], method = "class", na.action = na.omit)
BsmtExp_pred <- predict(class_mod, test_pred[is.na(test_pred$BsmtFinType2), ])
a <- colnames(BsmtExp_pred)[max.col(BsmtExp_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtExp_pred <- cbind(BsmtExp_pred, a)
test_pred$BsmtFinType2[is.na(test_pred$BsmtFinType2)] <- BsmtExp_pred$a

class_mod <- rpart(FireplaceQu ~ . , data = test_pred[!is.na(test_pred$FireplaceQu), ], method = "class", na.action = na.omit)
BsmtExp_pred <- predict(class_mod, test_pred[is.na(test_pred$FireplaceQu), ])
a <- colnames(BsmtExp_pred)[max.col(BsmtExp_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtExp_pred <- cbind(BsmtExp_pred, a)
test_pred$FireplaceQu[is.na(test_pred$FireplaceQu)] <- BsmtExp_pred$a

class_mod <- rpart(GarageType ~ . , data = test_pred[!is.na(test_pred$GarageType), ], method = "class", na.action = na.omit)
BsmtExp_pred <- predict(class_mod, test_pred[is.na(test_pred$GarageType), ])
a <- colnames(BsmtExp_pred)[max.col(BsmtExp_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtExp_pred <- cbind(BsmtExp_pred, a)
test_pred$GarageType[is.na(test_pred$GarageType)] <- BsmtExp_pred$a

anova_mod <- rpart(GarageYrBlt ~ . , data = test_pred[!is.na(test_pred$GarageYrBlt), ], method = "anova", na.action = na.omit)
MasVnrArea_pred <- predict(anova_mod, test_pred[is.na(test_pred$GarageYrBlt),])
test_pred$GarageYrBlt[is.na(test_pred$GarageYrBlt)] <- MasVnrArea_pred
test_pred$GarageYrBlt <- round(test_pred$GarageYrBlt, digits = 0)

# GarageFinish, GarageQual, GarageCond are all missing for 81 rows in the database
class_mod <- rpart(GarageFinish ~ . , data = test_pred[!is.na(test_pred$GarageFinish), ], method = "class", na.action = na.omit)
BsmtExp_pred <- predict(class_mod, test_pred[is.na(test_pred$GarageFinish), ])
a <- colnames(BsmtExp_pred)[max.col(BsmtExp_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtExp_pred <- cbind(BsmtExp_pred, a)
test_pred$GarageFinish[is.na(test_pred$GarageFinish)] <- BsmtExp_pred$a

class_mod <- rpart(GarageQual ~ . , data = test_pred[!is.na(test_pred$GarageQual), ], method = "class", na.action = na.omit)
BsmtExp_pred <- predict(class_mod, test_pred[is.na(test_pred$GarageQual), ])
a <- colnames(BsmtExp_pred)[max.col(BsmtExp_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtExp_pred <- cbind(BsmtExp_pred, a)
test_pred$GarageQual[is.na(test_pred$GarageQual)] <- BsmtExp_pred$a

class_mod <- rpart(GarageCond ~ . , data = test_pred[!is.na(test_pred$GarageCond), ], method = "class", na.action = na.omit)
BsmtExp_pred <- predict(class_mod, test_pred[is.na(test_pred$GarageCond), ])
a <- colnames(BsmtExp_pred)[max.col(BsmtExp_pred, ties.method = "first")]
a <- as.data.frame(a)
BsmtExp_pred <- cbind(BsmtExp_pred, a)
test_pred$GarageCond[is.na(test_pred$GarageCond)] <- BsmtExp_pred$a

class_mod <- rpart(Exterior1st ~ . , data = test_pred[!is.na(test_pred$Exterior1st), ], method = "class", na.action = na.omit)
Ext1_pred <- predict(class_mod, test_pred[is.na(test_pred$Exterior1st), ])
a <- colnames(Ext1_pred)[max.col(Ext1_pred, ties.method = "first")]
a <- as.data.frame(a)
Ext1_pred <- cbind(Ext1_pred, a)
test_pred$Exterior1st[is.na(test_pred$Exterior1st)] <- Ext1_pred$a

class_mod <- rpart(Exterior2nd ~ . , data = test_pred[!is.na(test_pred$Exterior2nd), ], method = "class", na.action = na.omit)
Ext1_pred <- predict(class_mod, test_pred[is.na(test_pred$Exterior2nd), ])
a <- colnames(Ext1_pred)[max.col(Ext1_pred, ties.method = "first")]
a <- as.data.frame(a)
Ext1_pred <- cbind(Ext1_pred, a)
test_pred$Exterior2nd[is.na(test_pred$Exterior2nd)] <- Ext1_pred$a

anova_mod <- rpart(BsmtFinSF1 ~ . , data = test_pred[!is.na(test_pred$BsmtFinSF1), ], method = "anova", na.action = na.omit)
MasVnrArea_pred <- predict(anova_mod, test_pred[is.na(test_pred$BsmtFinSF1),])
test_pred$BsmtFinSF1[is.na(test_pred$BsmtFinSF1)] <- MasVnrArea_pred
test_pred$BsmtFinSF1 <- round(test_pred$BsmtFinSF1, digits = 2)

anova_mod <- rpart(BsmtUnfSF ~ . , data = test_pred[!is.na(test_pred$BsmtUnfSF), ], method = "anova", na.action = na.omit)
MasVnrArea_pred <- predict(anova_mod, test_pred[is.na(test_pred$BsmtUnfSF),])
test_pred$BsmtUnfSF[is.na(test_pred$BsmtUnfSF)] <- MasVnrArea_pred
test_pred$BsmtUnfSF <- round(test_pred$BsmtUnfSF, digits = 2)

anova_mod <- rpart(BsmtFullBath ~ . , data = test_pred[!is.na(test_pred$BsmtFullBath), ], method = "anova", na.action = na.omit)
MasVnrArea_pred <- predict(anova_mod, test_pred[is.na(test_pred$BsmtFullBath),])
test_pred$BsmtFullBath[is.na(test_pred$BsmtFullBath)] <- MasVnrArea_pred
test_pred$BsmtFullBath <- round(test_pred$BsmtFullBath, digits = 0)

class_mod <- rpart(BsmtHalfBath ~ . , data = test_pred[!is.na(test_pred$BsmtHalfBath), ], method = "class", na.action = na.omit)
Ext1_pred <- predict(class_mod, test_pred[is.na(test_pred$BsmtHalfBath), ])
a <- colnames(Ext1_pred)[max.col(Ext1_pred, ties.method = "first")]
a <- as.data.frame(a)
Ext1_pred <- cbind(Ext1_pred, a)
test_pred$BsmtHalfBath[is.na(test_pred$BsmtHalfBath)] <- Ext1_pred$a

class_mod <- rpart(Functional ~ . , data = test_pred[!is.na(test_pred$Functional), ], method = "class", na.action = na.omit)
Ext1_pred <- predict(class_mod, test_pred[is.na(test_pred$Functional), ])
a <- colnames(Ext1_pred)[max.col(Ext1_pred, ties.method = "first")]
a <- as.data.frame(a)
Ext1_pred <- cbind(Ext1_pred, a)
test_pred$Functional[is.na(test_pred$Functional)] <- Ext1_pred$a

class_mod <- rpart(GarageCars ~ . , data = test_pred[!is.na(test_pred$GarageCars), ], method = "class", na.action = na.omit)
Ext1_pred <- predict(class_mod, test_pred[is.na(test_pred$GarageCars), ])
a <- colnames(Ext1_pred)[max.col(Ext1_pred, ties.method = "first")]
a <- as.data.frame(a)
Ext1_pred <- cbind(Ext1_pred, a)
test_pred$GarageCars[is.na(test_pred$GarageCars)] <- Ext1_pred$a

anova_mod <- rpart(GarageArea ~ . , data = test_pred[!is.na(test_pred$GarageArea), ], method = "anova", na.action = na.omit)
MasVnrArea_pred <- predict(anova_mod, test_pred[is.na(test_pred$GarageArea),])
test_pred$GarageArea[is.na(test_pred$GarageArea)] <- MasVnrArea_pred
test_pred$GarageArea <- round(test_pred$GarageArea, digits = 2)

class_mod <- rpart(SaleType ~ . , data = test_pred[!is.na(test_pred$SaleType), ], method = "class", na.action = na.omit)
Ext1_pred <- predict(class_mod, test_pred[is.na(test_pred$SaleType), ])
a <- colnames(Ext1_pred)[max.col(Ext1_pred, ties.method = "first")]
a <- as.data.frame(a)
Ext1_pred <- cbind(Ext1_pred, a)
test_pred$SaleType[is.na(test_pred$SaleType)] <- Ext1_pred$a

class_mod <- rpart(gentle_slope ~ . , data = test_pred[!is.na(test_pred$gentle_slope), ], method = "class", na.action = na.omit)
Ext1_pred <- predict(class_mod, test_pred[is.na(test_pred$gentle_slope), ])
a <- colnames(Ext1_pred)[max.col(Ext1_pred, ties.method = "first")]
a <- as.data.frame(a)
Ext1_pred <- cbind(Ext1_pred, a)
test_pred$gentle_slope[is.na(test_pred$gentle_slope)] <- Ext1_pred$a

class_mod <- rpart(KitchenQual ~ . , data = test_pred[!is.na(test_pred$KitchenQual), ], method = "class", na.action = na.omit)
Ext1_pred <- predict(class_mod, test_pred[is.na(test_pred$KitchenQual), ])
a <- colnames(Ext1_pred)[max.col(Ext1_pred, ties.method = "first")]
a <- as.data.frame(a)
Ext1_pred <- cbind(Ext1_pred, a)
test_pred$KitchenQual[is.na(test_pred$KitchenQual)] <- Ext1_pred$a

test_pred$regshape[test_pred$LotShape == "Reg"] <- 1
test_pred$regshape[test_pred$LotShape != "Reg"] <- 0

test_pred$flat[test_pred$LandContour == "Lvl"] <- 1
test_pred$flat[test_pred$LandContour == "Bnk"] <- 0
test_pred$flat[test_pred$LandContour == "Low"] <- 2
test_pred$flat[test_pred$LandContour == "HLS"] <- 3

test_pred$gentle_slope[test_pred$LandSlope == "Gtl"] <- 1
test_pred$gentle_slope[test_pred$LandSlope != "Gtl"] <- 0

# lotconfig <- summarize(group_by(train, LotConfig), mean(SalePrice, na.rm=T))
# lotconfig
test_pred$culdesac_fr3[test_pred$LotConfig %in% c("CulDSac", "FR3")] <- 1
test_pred$culdesac_fr3[!test_pred$LotConfig %in% c("CulDSac", "FR3")] <- 0


#nbhdprice[order(nbhdprice$`mean(SalePrice, na.rm = T)`),]

# use the criteria made from train set
test_pred$nbhd_price_level[test_pred$Neighborhood %in% nbhdprice_lo$Neighborhood] <- 1
test_pred$nbhd_price_level[test_pred$Neighborhood %in% nbhdprice_med$Neighborhood] <- 2
test_pred$nbhd_price_level[test_pred$Neighborhood %in% nbhdprice_hi$Neighborhood] <- 3
test_pred$nbhd_price_level[test_pred$Neighborhood %in% nbhdprice_suphi$Neighborhood] <- 4


test_pred$pos_features_1[test_pred$Condition1 %in% c("PosA", "PosN", "RRNn")] <- 2
test_pred$pos_features_1[!test_pred$Condition1 %in% c("PosA", "PosN", "RRNn", "Feedr", "RRAe", "Artery")] <- 1
test_pred$pos_features_1[test_pred$Condition1 %in% c("Feedr", "RRAe", "Artery")] <- 0

# cond2 <- summarize(group_by(train, Condition2), mean(SalePrice, na.rm=T))

test_pred$pos_features_2[!test_pred$Condition1 %in% c("Norm", "RRAe", "PosA", "PosN")] <- 0
test_pred$pos_features_2[test_pred$Condition1 %in% c("Norm", "RRAe")] <- 1
test_pred$pos_features_2[test_pred$Condition1 %in% c("PosA", "PosN")] <- 2


# summarize(group_by(train, BldgType), mean(SalePrice, na.rm=T))

test_pred$twnhs_end_or_1fam[test_pred$BldgType %in% c("1Fam", "TwnhsE")] <- 1
test_pred$twnhs_end_or_1fam[!test_pred$BldgType %in% c("1Fam", "TwnhsE")] <- 0

test_pred$house_style_level[test_pred$HouseStyle %in% housestyle_lo$HouseStyle] <- 1
test_pred$house_style_level[test_pred$HouseStyle %in% housestyle_med$HouseStyle] <- 2
test_pred$house_style_level[test_pred$HouseStyle %in% housestyle_hi$HouseStyle] <- 3


# roofstyle_price <- summarize(group_by(train, RoofStyle), mean(SalePrice, na.rm=T))
# Could be more catalogies?
test_pred$roof_hip_shed[test_pred$RoofStyle %in% c("Hip", "Shed", "Flat")] <- 1
test_pred$roof_hip_shed[!test_pred$RoofStyle %in% c("Hip", "Shed", "Flat")] <- 0


test_pred$exterior_1[test_pred$Exterior1st %in% matl_lo_1$Exterior1st] <- 1
test_pred$exterior_1[test_pred$Exterior1st %in% matl_med_1$Exterior1st] <- 2
test_pred$exterior_1[test_pred$Exterior1st %in% matl_hi_1$Exterior1st] <- 3


test_pred$exterior_2[test_pred$Exterior2nd %in% matl_lo$Exterior2nd] <- 1
test_pred$exterior_2[test_pred$Exterior2nd %in% matl_med$Exterior2nd] <- 2
test_pred$exterior_2[test_pred$Exterior2nd %in% matl_hi$Exterior2nd] <- 3
test_pred$exterior_2[test_pred$Exterior2nd %in% matl_suphi$Exterior2nd] <- 4

test_pred$exterior_mason_1[test_pred$MasVnrType %in% c("Stone", "BrkFace") | is.na(test_pred$MasVnrType)] <- 1
test_pred$exterior_mason_1[!test_pred$MasVnrType %in% c("Stone", "BrkFace") & !is.na(test_pred$MasVnrType)] <- 0

test_pred$exterior_cond[test_pred$ExterQual == "Ex"] <- 4
test_pred$exterior_cond[test_pred$ExterQual == "Gd"] <- 3
test_pred$exterior_cond[test_pred$ExterQual == "TA"] <- 2
test_pred$exterior_cond[test_pred$ExterQual == "Fa"] <- 1


# price <- summarize(group_by(train, ExterCond), mean(SalePrice, na.rm=T))

test_pred$exterior_cond2[test_pred$ExterCond == "Ex"] <- 5
test_pred$exterior_cond2[test_pred$ExterCond == "Gd"] <- 4
test_pred$exterior_cond2[test_pred$ExterCond == "TA"] <- 3
test_pred$exterior_cond2[test_pred$ExterCond == "Fa"] <- 2
test_pred$exterior_cond2[test_pred$ExterCond == "Po"] <- 1

# price <- summarize(group_by(train, Foundation), mean(SalePrice, na.rm=T))

test_pred$found_concrete[test_pred$Foundation == "PConc"] <- 2
test_pred$found_concrete[test_pred$Foundation == "Slab"] <- 0
test_pred$found_concrete[!test_pred$Foundation %in% c("PConc", "Slab")] <- 1


# price <- summarize(group_by(train, BsmtQual), mean(SalePrice, na.rm=T))

test_pred$bsmt_cond1[test_pred$BsmtQual == "Ex"] <- 5
test_pred$bsmt_cond1[test_pred$BsmtQual == "Gd"] <- 4
test_pred$bsmt_cond1[test_pred$BsmtQual == "TA"] <- 3
test_pred$bsmt_cond1[test_pred$BsmtQual == "Fa"] <- 2
test_pred$bsmt_cond1[is.na(test_pred$BsmtQual)] <- 1


# price <- summarize(group_by(train, BsmtCond), mean(SalePrice, na.rm=T))

test_pred$bsmt_cond2[test_pred$BsmtCond == "Gd"] <- 5
test_pred$bsmt_cond2[test_pred$BsmtCond == "TA"] <- 4
test_pred$bsmt_cond2[test_pred$BsmtCond == "Fa"] <- 3
test_pred$bsmt_cond2[is.na(test_pred$BsmtCond)] <- 2
test_pred$bsmt_cond2[test_pred$BsmtCond == "Po"] <- 1


# price <- summarize(group_by(train, BsmtExposure), mean(SalePrice, na.rm=T))

test_pred$bsmt_exp[test_pred$BsmtExposure == "Gd"] <- 5
test_pred$bsmt_exp[test_pred$BsmtExposure == "Av"] <- 4
test_pred$bsmt_exp[test_pred$BsmtExposure == "Mn"] <- 3
test_pred$bsmt_exp[test_pred$BsmtExposure == "No"] <- 2
test_pred$bsmt_exp[is.na(test_pred$BsmtExposure)] <- 1


# price <- summarize(group_by(train, BsmtFinType1), mean(SalePrice, na.rm=T))

test_pred$bsmt_fin1[test_pred$BsmtFinType1 == "GLQ"] <- 5
test_pred$bsmt_fin1[test_pred$BsmtFinType1 == "Unf"] <- 4
test_pred$bsmt_fin1[test_pred$BsmtFinType1 == "ALQ"] <- 3
test_pred$bsmt_fin1[test_pred$BsmtFinType1 %in% c("BLQ", "Rec", "LwQ")] <- 2
test_pred$bsmt_fin1[is.na(test_pred$BsmtFinType1)] <- 1



# price <- summarize(group_by(train, BsmtFinType2), mean(SalePrice, na.rm=T))

test_pred$bsmt_fin2[test_pred$BsmtFinType2 == "ALQ"] <- 6
test_pred$bsmt_fin2[test_pred$BsmtFinType2 == "Unf"] <- 5
test_pred$bsmt_fin2[test_pred$BsmtFinType2 == "GLQ"] <- 4
test_pred$bsmt_fin2[test_pred$BsmtFinType2 %in% c("Rec", "LwQ")] <- 3
test_pred$bsmt_fin2[test_pred$BsmtFinType2 == "BLQ"] <- 2
test_pred$bsmt_fin2[is.na(test_pred$BsmtFinType2)] <- 1

# price <- summarize(group_by(train, Heating), mean(SalePrice, na.rm=T))


test_pred$gasheat[test_pred$Heating %in% c("GasA", "GasW")] <- 2
test_pred$gasheat[test_pred$Heating %in% c("Wall", "OthW")] <- 1
test_pred$gasheat[!test_pred$Heating %in% c("GasA", "GasW", "Wall", "Othw")] <- 0


# price <- summarize(group_by(train, HeatingQC), mean(SalePrice, na.rm=T))

test_pred$heatqual[test_pred$HeatingQC == "Ex"] <- 5
test_pred$heatqual[test_pred$HeatingQC == "Gd"] <- 4
test_pred$heatqual[test_pred$HeatingQC == "TA"] <- 3
test_pred$heatqual[test_pred$HeatingQC == "Fa"] <- 2
test_pred$heatqual[test_pred$HeatingQC == "Po"] <- 1


# Only two potential values
test_pred$air[test_pred$CentralAir == "Y"] <- 1
test_pred$air[test_pred$CentralAir == "N"] <- 0


# price <- summarize(group_by(train, Electrical), mean(SalePrice, na.rm=T))

# Only one NA and One Mix which has the lowest price. So l just let it go
test_pred$standard_electric[test_pred$Electrical == "SBrkr" | is.na(test_pred$Electrical)] <- 1
test_pred$standard_electric[!test_pred$Electrical == "SBrkr" & !is.na(test_pred$Electrical)] <- 0


# price <- summarize(group_by(train, KitchenQual), mean(SalePrice, na.rm=T))

test_pred$kitchen[test_pred$KitchenQual == "Ex"] <- 4
test_pred$kitchen[test_pred$KitchenQual == "Gd"] <- 3
test_pred$kitchen[test_pred$KitchenQual == "TA"] <- 2
test_pred$kitchen[test_pred$KitchenQual == "Fa"] <- 1

test_pred$fire[test_pred$FireplaceQu == "Ex"] <- 5
test_pred$fire[test_pred$FireplaceQu == "Gd"] <- 4
test_pred$fire[test_pred$FireplaceQu == "TA"] <- 3
test_pred$fire[test_pred$FireplaceQu == "Fa"] <- 2
test_pred$fire[test_pred$FireplaceQu == "Po"] <- 1

test_pred$gar_attach[test_pred$GarageType == "CarPort"] <- 0
test_pred$gar_attach[test_pred$GarageType == "Detchd"] <- 1
test_pred$gar_attach[test_pred$GarageType %in% c("2Types", "Basment")] <- 2
test_pred$gar_attach[test_pred$GarageType == "Attchd"] <- 3
test_pred$gar_attach[test_pred$GarageType == "BuiltIn"] <- 4

# can be more detailed
# price <- summarize(group_by(train, GarageFinish), mean(SalePrice, na.rm=T))

test_pred$gar_finish[test_pred$GarageFinish %in% c("Fin", "RFn")] <- 1
test_pred$gar_finish[!test_pred$GarageFinish %in% c("Fin", "RFn")] <- 0

test_pred$garqual[test_pred$GarageQual == "Ex"] <- 5
test_pred$garqual[test_pred$GarageQual == "Gd"] <- 4
test_pred$garqual[test_pred$GarageQual == "TA"] <- 3
test_pred$garqual[test_pred$GarageQual == "Fa"] <- 2
test_pred$garqual[test_pred$GarageQual == "Po" | is.na(test_pred$GarageQual)] <- 1


# price <- summarize(group_by(train, GarageCond), mean(SalePrice, na.rm=T))

test_pred$garqual2[test_pred$GarageCond == "Ex"] <- 5
test_pred$garqual2[test_pred$GarageCond == "Gd"] <- 4
test_pred$garqual2[test_pred$GarageCond == "TA"] <- 3
test_pred$garqual2[test_pred$GarageCond == "Fa"] <- 2
test_pred$garqual2[test_pred$GarageCond == "Po" | is.na(test_pred$GarageCond)] <- 1

test_pred$paved_drive[test_pred$PavedDrive == "Y"] <- 2
test_pred$paved_drive[test_pred$PavedDrive == "N"] <- 0
test_pred$paved_drive[test_pred$PavedDrive == "P"] <- 1


# price <- summarize(group_by(train, Functional), mean(SalePrice, na.rm=T))

test_pred$housefunction[test_pred$Functional %in% c("Typ", "Mod")] <- 2
test_pred$housefunction[test_pred$Functional %in% c("Maj1", "Min1", "Min2")] <- 1
test_pred$housefunction[test_pred$Functional %in% c("Sev", "Maj2")] <- 0


# price <- summarize(group_by(train, SaleType), mean(SalePrice, na.rm=T))

test_pred$sale_cat[test_pred$SaleType %in% c("New", "Con")] <- 5
test_pred$sale_cat[test_pred$SaleType %in% c("CWD", "ConLI")] <- 4
test_pred$sale_cat[test_pred$SaleType %in% c("WD")] <- 3
test_pred$sale_cat[test_pred$SaleType %in% c("COD", "ConLw", "ConLD")] <- 2
test_pred$sale_cat[test_pred$SaleType %in% c("Oth")] <- 1


# price[order(price$`mean(SalePrice, na.rm = T)`),]

test_pred$sale_cond[test_pred$SaleCondition %in% c("Partial")] <- 4
test_pred$sale_cond[test_pred$SaleCondition %in% c("Normal", "Alloca")] <- 3
test_pred$sale_cond[test_pred$SaleCondition %in% c("Family","Abnorml")] <- 2
test_pred$sale_cond[test_pred$SaleCondition %in% c("AdjLand")] <- 1


# price <- summarize(group_by(train, MSZoning), mean(SalePrice, na.rm=T))
test_pred$zone[test_pred$MSZoning %in% c("FV")] <- 4
test_pred$zone[test_pred$MSZoning %in% c("RL")] <- 3
test_pred$zone[test_pred$MSZoning %in% c("RH","RM")] <- 2
test_pred$zone[test_pred$MSZoning %in% c("C (all)")] <- 1

test_pred$Street <- NULL
test_pred$LotShape <- NULL
test_pred$LandContour <- NULL
test_pred$Utilities <- NULL
test_pred$LotConfig <- NULL
test_pred$LandSlope <- NULL
test_pred$Neighborhood <- NULL
test_pred$Condition1 <- NULL
test_pred$Condition2 <- NULL
test_pred$BldgType <- NULL
test_pred$HouseStyle <- NULL
test_pred$RoofStyle <- NULL
test_pred$RoofMatl <- NULL

test_pred$Exterior1st <- NULL
test_pred$Exterior2nd <- NULL
test_pred$MasVnrType <- NULL
test_pred$ExterQual <- NULL
test_pred$ExterCond <- NULL

test_pred$Foundation <- NULL
test_pred$BsmtQual <- NULL
test_pred$BsmtCond <- NULL
test_pred$BsmtExposure <- NULL
test_pred$BsmtFinType1 <- NULL
test_pred$BsmtFinType2 <- NULL

test_pred$Heating <- NULL
test_pred$HeatingQC <- NULL
test_pred$CentralAir <- NULL
test_pred$Electrical <- NULL
test_pred$KitchenQual <- NULL
test_pred$FireplaceQu <- NULL

test_pred$GarageType <- NULL
test_pred$GarageFinish <- NULL
test_pred$GarageQual <- NULL
test_pred$GarageCond <- NULL
test_pred$PavedDrive <- NULL

test_pred$Functional <- NULL
test_pred$PoolQC <- NULL
test_pred$Fence <- NULL
test_pred$MiscFeature <- NULL
test_pred$SaleType <- NULL
test_pred$SaleCondition <- NULL
test_pred$MSZoning <- NULL
test_pred$Alley <- NULL

test_pred$Id <- NULL

#Interactions based on correlation
test_pred$year_qual <- test_pred$YearBuilt*test_pred$OverallQual #overall condition
test_pred$year_r_qual <- test_pred$YearRemodAdd*test_pred$OverallQual #quality x remodel
test_pred$qual_bsmt <- test_pred$OverallQual*test_pred$TotalBsmtSF #quality x basement size
test_pred$livarea_qual <- test_pred$OverallQual*test_pred$GrLivArea #quality x living area
test_pred$qual_bath <- test_pred$OverallQual*test_pred$FullBath #quality x baths
test_pred$qual_ext <- test_pred$OverallQual*test_pred$exterior_cond #quality x exterior
test_pred$car_area <- test_pred$GarageCars*test_pred$GarageArea # my try


#Create matrices from the data frames
testData<- as.matrix(test_pred, rownames.force=NA)

#Turn the matrices into sparse matrices
test_set <- as(testData, "sparseMatrix")

#####
#colnames(train2)
#Cross Validate the model

vars <- c(1:79) #choose the columns we want to use in the prediction matrix

testD <- xgb.DMatrix(data = test_set[,vars])
#Column names must match the inputs EXACTLY
prediction <- predict(bstSparse, testD) #Make the prediction based on the half of the training data set aside

test_pred1 <- fread("test.csv")
#Put testing prediction and test dataset all together
prediction <- as.data.frame(as.matrix(prediction))
colnames(prediction) <- "SalePrice"
prediction <- cbind(test_pred1$Id,prediction)
colnames(prediction) <- c("Id", "SalePrice")



write.csv(prediction, "submission4.csv", row.names = FALSE)
