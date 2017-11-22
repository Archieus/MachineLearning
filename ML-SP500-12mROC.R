library(quantmod)
library(mlbench)
library(caret)
Sys.setenv(TZ = "UTC")

Symbols <- c("SP500")
getSymbols(Symbols, src = "FRED", from = "2007-01-01")

SP500ret <- periodReturn(SP500, "monthly", type = "log", indexAt = "firstof")
SP500ROC <- ROC(to.period(SP500, 'months', indexAt = 'firstof'),12)[,4]

SPState <- cbind(ifelse(SP500ret[,1] > 0 ,"Up", "Down"))

SPState <- as.data.frame(SPState)

SPdata <- na.omit(cbind(as.data.frame(SP500ROC), SPState))
colnames(SPdata) <- c("ROC", "State")

#### CREATED LIST of 80% OF ROWS IN ORIGINAL DATASET TO TRAIN THE MODEL ####
validation_index <- createDataPartition(SPdata$State, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- SPdata[-validation_index,]
# use the remaining 80% of data to training and testing the models
SPState <- SPdata[validation_index,]

#### SUMMARIZE DATASET ####
# dimensions of dataset
dim(SPdata)

# list types for each attribute
sapply(SPdata, class)

# list the levels for the class
levels(SPdata$State)

# summarize the class distribution
percentage <- prop.table(table(SPdata$State)) * 100
cbind(freq=table(SPdata$State), percentage=percentage)

summary(SPdata)


control <- trainControl(method="cv", number=10)
metric <- "Accuracy"
#### BUILD MODELS ####
# A) LINEAR ALGORITHMS
fit.lda <- train(State~., data=SPdata, method="lda", metric=metric, trControl=control)

# B) NONLINEAR ALGORITHMS
# CART
fit.cart <- train(State~., data=SPdata, method="rpart", metric=metric, trControl=control)
# kNN
fit.knn <- train(State~., data=SPdata, method="knn", metric=metric, trControl=control)

# C) ADVANCED ALGORITHMS
# SVM
fit.svm <- train(State~., data=SPdata, method="svmRadial", metric=metric, trControl=control)
# Random Forest
fit.rf <- train(State~., data=SPdata, method="rf", metric=metric, trControl=control)

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

