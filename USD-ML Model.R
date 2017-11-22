library(quantmod)
library(mlbench)
library(caret)
library(e1071)

Symbols <- c("IR", "IQ", "GS10", "DTWEXM", "SP500")
getSymbols(Symbols, src = "FRED", from = "2007-01-01")

USD.mo <- to.monthly(DTWEXM, indeAt = 'firstof')
USD.chg <- ROC(USD.mo,1)

SP500.mo <- to.monthly(SP500, indexAt = 'firstof')
SP5.chg <- ROC(SP500.mo,1)

State <- cbind(ifelse(SP5.chg[,4] > 0,"Up", "Down"))
State <- lag(State, -1)
Live <- State
State <- as.data.frame(State)
Live.df <- as.data.frame(Live)
Live.df <- na.omit(Live.df)

RawData <- as.data.frame(na.omit(cbind(IR, IQ, GS10, USD.mo[,4])))

LiveData <- na.omit(cbind(tail(RawData, nrow(Live.df)), Live.df))
names(LiveData) <- c("ImportIndex", "ExportIndex", "UST10Y", "USD", "State")

USDData <- na.omit(cbind(tail(RawData, nrow(State)), State))
names(USDData) <- c("ImportIndex", "ExportIndex", "UST10Y", "USD", "State")

#### IDENTIFY BEST FEATURES OF THE LIST FOR THE MODEL ####
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(USDData[,1:4], USDData[,5], sizes=c(1:4), rfeControl=control)
# summarize the results
print(results)

#### IDENTIFY HIGHLY CORRELATED FEATURES TO REMOVE REDUNDANCY ####
correlationMatrix <- round(cor(USDData[,1:4]),4)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

#### CREATED LIST of 80% OF ROWS IN ORIGINAL DATASET TO TRAIN THE MODEL ####
validation_index <- createDataPartition(USDData$State, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- USDData[-validation_index,]
# use the remaining 80% of data to training and testing the models
USDData <- USDData[validation_index,]

#### SUMMARIZE DATASET ####
# dimensions of dataset
dim(USDData)

# list types for each attribute
sapply(USDData, class)

# list the levels for the class
levels(USDData$State)

# summarize the class distribution
percentage <- prop.table(table(USDData$State)) * 100
cbind(freq=table(USDData$State), percentage=percentage)

summary(USDData)

#### VISUALIZE DATASETS ####
# split input and output
x <- USDData[,1:4]
y <- USDData[,5]

# boxplot for each attribute on one image
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(USDData[-5])[i])
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

#### CONTROL METHODS ####
#### Rolling Window aka Forward Chainging / Walk Forward Analysis best for time series analysis ####
## Resampling Methods in Caret ##
## "boot", "boot632", "optimism_boot", "boot_all", "cv", "repeatedcv", "LOOCV", "LGOCV" ##
## "none" (only fits one model to the entire training set),
## "oob" (only for random forest, bagged trees, bagged earth, bagged flexible discriminant analysis,
## or conditional tree forest models)
## timeslice, "adaptive_cv", "adaptive_boot" or "adaptive_LGOCV"

#control <- trainControl(method = "timeslice", initialWindow = 97,  fixedWindow = TRUE, horizon = 1)

####
###Metric "RMSE" & "Rsquared" for Regression; "Accuracy" & "Kappa" for Classification
metric <- "Accuracy"

#### BUILD MODELS ####
# A) LINEAR ALGORITHMS
fit.lda <- train(State~., data=USDData, method="lda", metric=metric, trControl=control)

# B) NONLINEAR ALGORITHMS
# CART
fit.cart <- train(State~., data=USDData, method="rpart", metric=metric, trControl=control)
# kNN
fit.knn <- train(State~., data=USDData, method="knn", metric=metric, trControl=control)

# C) ADVANCED ALGORITHMS
# SVM
fit.svm <- train(State~., data=USDData, method="svmRadial", metric=metric, trControl=control)
# Random Forest
fit.rf <- train(State~., data=USDData, method="rf", metric=metric, trControl=control)

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
final_model <- svm(State~., LiveData) ###The entire Dataset
# save the model to disk
saveRDS(final_model, "./USD_model.rds")

# load the model
super_model <- readRDS("./USD_model.rds")
print(super_model)

### Predictive Model with probabilities ###
pred.model <- svm(State~., LiveData, probability = TRUE)
predict(pred.model, LiveData[,1:4], probability = TRUE)

KNN_Final_Model <- knnreg(State~., LiveData)
predict(KNN_Final_Model, LiveData[,1:4])

# make a predictions on "new data" using the final model
final_predictions <- predict(super_model, LiveData[,1:4])
confusionMatrix(final_predictions, LiveData$State)
#####################################


