library(quantmod)
library(mlbench)
library(caret)

Symbols <- c("SP500", "FEDFUNDS", "GS10")
getSymbols(Symbols, src = "FRED", from = "2007-01-01")

SP500ret <- periodReturn(SP500, "weekly", type = "log", indexAt = "firstof")

FFROC <- na.omit(ROC(to.period(FEDFUNDS, 'weeks', indexAt = 'firstof')),1)[,4]

GS10ROC <- na.omit(ROC(to.period(GS10, 'weeks', indexAt = 'firstof')),1)[,4]

SPData <- round(na.omit(cbind(FFROC, GS10ROC)),4)

dataset <- as.data.frame(SPData)
State <- cbind(ifelse(SP500ret > 0,"Up","Down"))
State <- as.data.frame(State)

dataset <- cbind(tail(dataset,nrow(State)), State)
colnames(dataset) <- c("FedFund", "TSY10", "State")

#### IDENTIFY BEST FEATURES OF THE LIST FOR THE MODEL ####
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(dataset[,1:2], dataset[,3], sizes=c(1:2), rfeControl=control)
# summarize the results
print(results)

#### IDENTIFY HIGHLY CORRELATED FEATURES TO REMOVE REDUNDANCY ####
correlationMatrix <- round(cor(dataset[,1:2]),4)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

#### SUMMARIZE DATASET ####
# dimensions of dataset
dim(dataset)

# list types for each attribute
sapply(dataset, class)

# list the levels for the class
levels(dataset$State)

# summarize the class distribution
percentage <- prop.table(table(dataset$State)) * 100
cbind(freq=table(dataset$State), percentage=percentage)

summary(dataset)

#### VISUALIZE DATASETS ####
# split input and output
x <- dataset[,1:2]
y <- dataset[,3]

# boxplot for each attribute on one image
par(mfrow=c(1,2))
for(i in 1:2) {
  boxplot(x[,i], main=names(dataset[-3])[i])
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
fit.lda <- train(State~., data=dataset, method="lda", metric=metric, trControl=control)

# B) NONLINEAR ALGORITHMS
# CART
fit.cart <- train(State~., data=dataset, method="rpart", metric=metric, trControl=control)
# kNN
fit.knn <- train(State~., data=dataset, method="knn", metric=metric, trControl=control)

# C) ADVANCED ALGORITHMS
# SVM
fit.svm <- train(State~., data=dataset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
fit.rf <- train(State~., data=dataset, method="rf", metric=metric, trControl=control)

#### SELECT BEST MODEL ####
# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.lda)

#### MAKE PREDICTIONS USING BEST MODEL ####
# estimate skill of the Model on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$State)

#### PREDICTIONS ON LIVE DATA ####
ModelPred <- predict(fit.lda,dataset)
confusionMatrix(ModelPred, dataset$State)
last(ModelPred)
