#######  @ Argimiro Arratia, 2014,   NNET and SVM modeling
###### http://computationalfinance.lsi.upc.edu

wdir="~/the path to the data"
setwd(wdir)

########Visual Test of normality#################
require(quantmod)
appl = readRDS("AAPL.rds")
apRd= periodReturn(appl,period="daily")
dsd=density(apRd) #estimate density of daily log ret
yl=c(min(dsd$y),max(dsd$y)) #set y limits
hist(apRd,probability=T,xlab="APPLE returns",main=NULL,ylim=yl)
lines(dsd)
##plot the normal density with mean, stdv of apRd
a=seq(min(apRd),max(apRd),0.001)
lines(a,dnorm(a,mean(apRd),sd(apRd)),col="red")

##Repeat above with period="weekly", "monthly".
##Run a Shapiro-wilk normality test
shapiroTest(apRd)
##############################################

#######  @ Argimiro Arratia, 2014,   NNET and SVM modeling
###### http://computationalfinance.lsi.upc.edu

wdir="~/the path to the data"
setwd(wdir)

########Nonlinear models#############################
####### SVM and Neural networks ############
library(e1071) ##for svm
library(nnet)
library(kernlab)
library(quantmod)
library(caret) ##for some data handling functions
library(Metrics)##Measures of prediction error:mse, mae
library(xts)

##Data:sp500m the S&P500 monthly readings from Jan. 1990 to Jan. 2012
sp500m = readRDS("sp500m.rds")
plot(sp500m['1910/1990'])

tau=1 #data is monthly. Try tau=12 (year), tau=1 (monthly)
ret=diff(log(sp500m),diff=tau)  ##compute tau-period returns

##Model Inputs:
##Define matrix of features (each column is a feature)
#Features: lags 1,2,3,5
feat = merge(na.trim(lag(ret,1)),na.trim(lag(ret,2)),na.trim(lag(ret,3)),na.trim(lag(ret,5)),
             #add other features here,
             all=FALSE)

##add TARGET. We want to predict RETURN
dataset = merge(feat,ret,all=FALSE)

colnames(dataset) = c("lag.1", "lag.2", "lag.3","lag.5",
                      #names of other features,
                      "TARGET")

##Divide data into training (75%) and testing (25%). Use caret methods
index = 1:nrow(dataset)
trainindex= createDataPartition(index,p=0.75,list=FALSE)
##process class sets as data frames
training = as.data.frame(dataset[trainindex,])
rownames(training) = NULL
testing = as.data.frame(dataset[-trainindex,])
rownames(testing) = NULL

##Train model
##############################################
##OPTION LAZY: one svm, one nnet built w/o tuning  (or tune by hand)
#parameters that can be tuned
#type="C" ##classification
type="eps-regression" ##regression
u= -2 ## -3,-2,-1,0,1,2,3
gam=10^{u}; w= 4.5 ##1.5,-1,0.5,2,3,4
cost=10^{w}
##The higher the cost produce less support vectors, increases accuracy
##However we may overfit
svmFit = svm (training[,-ncol(training)], training[,ncol(training)],
              type=type,
              kernel= "radial",
              gamma=gam,
              cost=cost
)
summary(svmFit)
##build SVM predictor
predsvm = predict(svmFit, testing[,-ncol(testing)])
##A nnet with size hidden layers +skip layer. Max iteration 10^4,
size=6
nnetFit = nnet(training[,-ncol(training)], training[,ncol(training)],
               size=size,skip=T, maxit=10^4,decay=10^{-2},trace=F,linout=T)
summary(nnetFit) ##gives description w/weights

##build NNET predictor type="raw"
prednet<-predict(nnetFit,testing[,-ncol(testing)],type="raw")

################end of Option Lazy ##############################

###EVALUATION
actualTS=testing[,ncol(testing)] ##the true series to predict
predicTS=predsvm ##choose appropriate
predicTS = prednet

##1. Evaluation for return prediction. Residual sum of squares
ssr= sum((actualTS - predicTS)^2); ssr
##Normalize Residual Mean Square Error (NRMSE)
nrmse = sqrt(ssr/((length(actualTS)-1)*var(actualTS))); nrmse
##percentage of outperforming direct sample mean (sample expected value)
pcorrect = (1-nrmse)*100; pcorrect
##For visual comparison
yl=c(min(actualTS,predicTS),max(actualTS,predicTS)) #set y limits
plot(actualTS,predicTS,ylim=yl)