library(quantmod)
library(mlbench)
library(caret)
library(e1071)

Symbols <- c("SP500", "VXVCLS", "CPIAUCSL", "FEDFUNDS", "GS10", "T10Y2YM", "UNRATE")
getSymbols(Symbols, src = "FRED", from = "2007-01-01")

SP500ret <- periodReturn(SP500, "monthly", type = "log", indexAt = "firstof")

#VXVCLS <- na.locf(VXVCLS, na.rm = FALSE, fromLast = TRUE)
#VXVMo <- to.period(VXVCLS, 'months', indexAt = 'firstof')
SP500Vol <- na.omit(ROC(to.period(VXVCLS, 'months', indexAt = 'firstof')),1)[,4]

#CPIAUCSL <- na.locf(CPIAUCSL, na.rm = FALSE, fromLast = TRUE)
CPIROC <- na.omit(ROC(to.period(CPIAUCSL, 'months', indexAt = 'firstof')),1)[,4]

FFROC <- na.omit(ROC(to.period(FEDFUNDS, 'months', indexAt = 'firstof')),1)[,4]

#GS10 <- na.locf(GS10, na.rm = FALSE, fromLast =TRUE)
GS10ROC <- na.omit(ROC(to.period(GS10, 'months', indexAt = 'firstof')),1)[,4]

#T10Y2YM <- na.locf(T10Y2YM, na.rm = FALSE, fromLast =TRUE)
T10Y2MROC <- na.omit(ROC(to.period(T10Y2YM, 'months', indexAt = 'firstof')),1)[,4]

#UNRATE <- na.locf(T10Y2YM, na.rm = FALSE, fromLast =TRUE)
URROC <- na.omit(ROC(to.period(UNRATE, 'months', indexAt = 'firstof')),1)[,4]

SPData <- round(na.omit(cbind(SP500Vol, CPIROC, FFROC, GS10ROC, T10Y2MROC, URROC )),4)

dataset <- as.data.frame(SPData)
State <- cbind(ifelse(SP500ret > 0,"Up","Down"))
State <- lag(State, -1)
State <- as.data.frame(State)

dataset <- na.omit(cbind(dataset, tail(State,nrow(dataset))))
colnames(dataset) <- c("Vol", "CPI", "FedFund", "TSY10", "TSYSprd", "Unemploy", "State")

#### IDENTIFY BEST FEATURES OF THE LIST FOR THE MODEL ####
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(dataset[,1:6], dataset[,7], sizes=c(1:6), rfeControl=control)
# summarize the results
print(results)

#### IDENTIFY HIGHLY CORRELATED FEATURES TO REMOVE REDUNDANCY ####
correlationMatrix <- round(cor(dataset[,1:6]),4)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

#### REMOVE REDUNDANDT FEATURE ####

dataset2 <- cbind(dataset[,c(1:3,5:7)])
LiveData <- cbind(dataset[,c(1:3,5:7)])

#### IDENTIFY BEST FEATURES OF THE LIST FOR THE MODEL (ex-redundant features) ####
control2 <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results2 <- rfe(dataset2[,1:5], dataset2[,6], sizes=c(1:5), rfeControl=control2)
# summarize the results
print(results2)

#### CREATED LIST of 80% OF ROWS IN ORIGINAL DATASET TO TRAIN THE MODEL ####
validation_index <- createDataPartition(dataset2$State, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- dataset2[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataset2 <- dataset2[validation_index,]

#### SUMMARIZE DATASET ####
# dimensions of dataset
dim(dataset2)

# list types for each attribute
sapply(dataset2, class)

# list the levels for the class
levels(dataset2$State)

# summarize the class distribution
percentage <- prop.table(table(dataset2$State)) * 100
cbind(freq=table(dataset2$State), percentage=percentage)

summary(dataset2)

#### VISUALIZE DATASETS ####
# split input and output
x <- dataset2[,1:5]
y <- dataset2[,6]

# boxplot for each attribute on one image
par(mfrow=c(1,5))
for(i in 1:5) {
  boxplot(x[,i], main=names(dataset[-6])[i])
}

# barplot for class breakdown
plot(y)

# scatterplot matrix (requires ellipse package to be available)
featurePlot(x=x, y=y, plot="ellipse")

# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box")

# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

#### EVALUATE SOME ALGORITHMS ####
# Run algorithms using 10-fold cross validation
# if method = "repeadedcv", "repeats =" can be used
# repeats can be dangerous if model is trained in "deteministic manner"
control <- trainControl(method="cv", number=10)

###Metric "RMSE" & "Rsquared" for Regression; "Accuracy" & "Kappa" for Classification
metric <- "Accuracy"

#### BUILD MODELS ####
# A) LINEAR ALGORITHMS
fit.lda <- train(State~., data=dataset2, method="lda", metric=metric, trControl=control)

# B) NONLINEAR ALGORITHMS
# CART
fit.cart <- train(State~., data=dataset2, method="rpart", metric=metric, trControl=control)
# kNN
fit.knn <- train(State~., data=dataset2, method="knn", metric=metric, trControl=control)

# C) ADVANCED ALGORITHMS
# SVM
fit.svm <- train(State~., data=dataset2, method="svmRadial", metric=metric, trControl=control)
# Random Forest
fit.rf <- train(State~., data=dataset2, method="rf", metric=metric, trControl=control)

#### SELECT BEST MODEL ####
# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.svm)

#### MAKE PREDICTIONS USING BEST MODEL ####
# estimate skill of the Model on the validation dataset
predictions <- predict(fit.svm, validation)
confusionMatrix(predictions, validation$State)

########## FINALIZE MODEL ##########
# Save Final Model #
final_model <- svm(State~., LiveData)
# save the model to disk
saveRDS(final_model, "./final_model.rds")

# load the model
super_model <- readRDS("./final_model.rds")
print(super_model)
# make a predictions on "new data" using the final model
final_predictions <- predict(super_model, LiveData[,1:5])
confusionMatrix(final_predictions, LiveData$State)
#####################################
