library(quantmod)
library(Quandl)
library(mlbench)
library(caret)
library(e1071)

Quandl.api_key("LDvEJuPkQWrPGpHwokVx")

FFFut <- Quandl("CHRIS/CME_FF1", type = "xts", start_date = '2007-01-01')
EURDFut <- Quandl("CHRIS/CME_ED1", type = "xts", start_date = '2007-01-01')
EURDF.1m <- Quandl("CHRIS/CME_EM1", type = "xts", start_date = '2007-01-01')

Symbols <- c("USDONTD156N", "DFF", "DGS3MO")
getSymbols(Symbols, src = "FRED", from = "2007-01-01")

Libor <- na.locf(USDONTD156N, na.rm = FALSE, fromLast = TRUE)
FedFd.7d <- na.locf(DFF, na.rm = FALSE, fromLast = TRUE)

Features <- na.omit(cbind(EURDFut[,6], EURDF.1m[,6], Libor, FedFd.7d))
dataset <- as.data.frame(Features)

TSY.chg <- na.omit(DGS3MO - lag(DGS3MO,1))
State <- cbind(ifelse(TSY.chg > 0,"Up","Down"))
StateLag<- lag(State, -1)
State <- na.omit(as.data.frame(State))
State <- tail(State,nrow(dataset))

dataset <- na.omit(cbind(dataset, State))
names(dataset) <- c("EURDFut", "1mEURFut", "Libor", "7dFedFd", "State")

#### IDENTIFY BEST FEATURES OF THE LIST FOR THE MODEL ####

control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(dataset[,1:4], dataset[,5], sizes=c(1:4), rfeControl=control)
# summarize the results
print(results)

#### IDENTIFY HIGHLY CORRELATED FEATURES TO REMOVE REDUNDANCY ####
correlationMatrix <- round(cor(dataset[,1:4]),4)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

#### REMOVE REDUNDANDT FEATURE ####

dataset2 <- cbind(dataset[,c(1:3,5)])
LiveData <- cbind(dataset[,c(1:3,5)])

#### IDENTIFY BEST FEATURES OF THE LIST FOR THE MODEL (ex-redundant features) ####
control2 <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results2 <- rfe(dataset2[,1:3], dataset2[,4], sizes=c(1:3), rfeControl=control2)
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
x <- dataset2[,1:3]
y <- dataset2[,4]

# boxplot for each attribute on one image
par(mfrow=c(1,3))
for(i in 1:3) {
  boxplot(x[,i], main=names(dataset[-4])[i])
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
metric <- "Accuracy"

#### BUILD MODELS ####
# A) LINEAR ALGORITHMS
fit.lda <- train(State~., data=dataset2, method="lda", metric=metric, trControl=control)

# B) NONLINEAR ALGORITHMS
# CART
# fit.cart <- train(State~., data=dataset2, method="rpart", metric=metric, trControl=control)
# kNN
fit.knn <- train(State~., data=dataset2, method="knn", metric=metric, trControl=control)

# C) ADVANCED ALGORITHMS
# SVM
fit.svm <- train(State~., data=dataset2, method="svmRadial", metric=metric, trControl=control)
# Random Forest
fit.rf <- train(State~., data=dataset2, method="rf", metric=metric, trControl=control)

#### SELECT BEST MODEL ####
# summarize accuracy of models
#results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
results <- resamples(list(lda=fit.lda, knn=fit.knn, svm=fit.svm, rf=fit.rf))
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

knn.final <- predict(fit.knn, LiveData[,1:3], type = "prob")

# load the model
super_model <- readRDS("./final_model.rds")
print(super_model)
# make a predictions on "new data" using the final model
final_predictions <- predict(super_model, LiveData[,1:3])
confusionMatrix(final_predictions, LiveData$State)
#####################################
pred.model <- svm(State~., LiveData, probability = TRUE)
predict(pred.model, LiveData[,1:3], probability = TRUE)

