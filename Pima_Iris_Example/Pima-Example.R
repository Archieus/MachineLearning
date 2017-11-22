library(caret)
library(mlbench)
library(caretEnsemble)

# load the dataset
data(PimaIndiansDiabetes)

#### Spot-Check Algorithms ####
# prepare 10-fold cross validation
trainControl <- trainControl(method="cv", number=10)
# estimate accuracy of logistic regression
set.seed(7)
fit.lr <- train(diabetes~., data=PimaIndiansDiabetes, method="glm", trControl=trainControl)
# estimate accuracy of linear discriminate analysis
fit.lda <- train(diabetes~., data=PimaIndiansDiabetes, method="lda", trControl=trainControl)
# collect resampling statistics
results <- resamples(list(LR=fit.lr, LDA=fit.lda))
# summarize results
summary(results)
#### Model Comparison and Selection ####
# plot the results
dotplot(results)

#### Improve Accuracy with Ensemble Predictions ####
# create sub-models
trainControl2 <- trainControl(method="cv", number=5, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('knn', 'glm')
models <- caretList(diabetes~., data=PimaIndiansDiabetes, trControl=trainControl2, methodList=algorithmList)
print(models)

# learn how to best combine the predictions
stackControl <- trainControl(method="cv", number=5, savePredictions=TRUE, classProbs=TRUE)
stack.glm <- caretStack(models, method="glm", trControl=stackControl)
print(stack.glm)

