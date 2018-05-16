# PHD PART 2 
rm(list = ls(all.names = all()))

setwd("E:\\Insofe_BigData\\PHD\\PART2")

library(sparklyr)
library(dplyr)
library(DMwR)
library(randomForest)
library(sqldf)

#spark_install(version = "2.1.0")

#sc <- spark_connect(master = "local")

test_data1 <- read.csv("Test.csv", header = T, sep = ",")
test_data_add <- read.csv("Test_AdditionalData.csv", header = T, sep = ",",as.is = T )

train_data1 <- read.csv("Train.csv", header = T, sep = ",")
train_data_add <- read.csv("Train_AdditionalData.csv", header = T, sep = "," ,as.is = T )

mergeSQlTrainA <- sqldf( " select a.*, TestA as TestA_1 from train_data1 a LEFT JOIN train_data_add ON (ID = TestA) ")
mergeSQlTrainB <- sqldf( " select DISTINCT a.*, TestB from mergeSQlTrainA a LEFT JOIN train_data_add ON (ID = TestB) ") 
mergeSQlTrain2 <- sqldf( " select a.*, case when TestA_1 is null or TestB is null then 2 when TestA_1 is not null and TestB is not null then 1 else 3 end test_results from mergeSQlTrainB a")


#mergeSQlTrainJoined <- inner_join(mergeSQlTrainA,mergeSQlTrainB, by=c("ID"))

View(mergeSQlTrain2)
#%>%
 # filter(!is.na(AC.Name)|n()==1)%>%


mergeSQlTestA <- sqldf( " select a.*, TestA as TestA_1 from test_data1 a LEFT JOIN test_data_add ON (ID = TestA) ")
mergeSQlTestB <- sqldf( " select DISTINCT a.*, TestB from mergeSQlTestA a LEFT JOIN test_data_add ON (ID = TestB) ") 

mergeSQlTest2 <- sqldf( " select a.*, case when TestA_1 is null or TestB is null then 2 when TestA_1 is not null and TestB is not null then 1 else 3 end test_results from mergeSQlTestB a")

mergeSQlTrain2$TestA_1 <- NULL
mergeSQlTrain2$TestB <- NULL

mergeSQlTest2$TestA_1 <- NULL
mergeSQlTest2$TestB <- NULL


nms <- c("Lubrication"              ,
         "Bearing_Vendor"           ,
         "Compression_ratio"        ,
         "Cylinder_arragement"      ,
         "Varaible_Valve_Timing_VVT",
         "Direct_injection"         ,
         "displacement"             ,
         "Max_Torque"               ,
         "Crankshaft_Design"        ,
         "y"                       ,
         "material_grade"          ,
         "Valve_Type"              ,
         "Fuel_Type"               ,
         "cam_arrangement"         ,
         "Turbocharger"            ,
         "Cylinder_deactivation"   ,
         "main_bearing_type"       ,
         "piston_type"             ,
         "Peak_Power"              ,
         "Liner_Design"            ,
         "test_results"    )


tms <- c("Lubrication"              ,
         "Bearing_Vendor"           ,
         "Compression_ratio"        ,
         "Cylinder_arragement"      ,
         "Varaible_Valve_Timing_VVT",
         "Direct_injection"         ,
         "displacement"             ,
         "Max_Torque"               ,
         "Crankshaft_Design"        ,
         "material_grade"          ,
         "Valve_Type"              ,
         "Fuel_Type"               ,
         "cam_arrangement"         ,
         "Turbocharger"            ,
         "Cylinder_deactivation"   ,
         "main_bearing_type"       ,
         "piston_type"             ,
         "Peak_Power"              ,
         "Liner_Design"            ,
         "test_results"    )

trainData <-  mergeSQlTrain2
testData <- mergeSQlTest2

trainData$y<-ifelse(trainData$y=="pass", 1, 0)

# Low Importance
trainData$Varaible_Valve_Timing_VVT <- NULL
trainData$main_bearing_type <- NULL
trainData$displacement <- NULL
testData$Varaible_Valve_Timing_VVT <- NULL
testData$main_bearing_type <- NULL
testData$displacement <- NULL

trainData$ID <- NULL
testData$ID  <- NULL

trainData<-centralImputation(trainData)
testData<-centralImputation(testData)

trainData <- trainData %>% mutate_if(is.character,as.factor)
testData <- testData %>% mutate_if(is.character,as.factor)

#main_bearing_type Journal

#trainData<-trainData[!(trainData$main_bearing_type=="Journal"),]

#trainData[nms] <- lapply(trainData[nms], as.factor) 
#testData[tnm] <-  lapply(testData[nms], as.factor) 

str(trainData)
str(testData)

colnames(testData)
colnames(trainData)

colnames(trainData)[2] <- "Target"

# Get the NA's
pMiss <- function(x){sum(is.na(x))/length(x)*100}

apply(trainData,2,pMiss)
apply(testData,2,pMiss)


#trainData <- trainData[order(trainData$ID),]
#testData <- testData[order(testData$ID),]

colnames(trainData)

str(trainData)
str(testData)

apply(trainData,2,pMiss)

View(trainData)

library(caTools)
set.seed(12345)

sum(is.na(trainData))

sample = sample.split(trainData, SplitRatio = .80)

train = subset(trainData, sample == TRUE)
test = subset(trainData, sample == FALSE)

summary(train)
str(train)
str(test)
str(testData)

# Build Model


sum(is.na(trainData))
sum(is.na(testData))

RF<- randomForest(Target ~ .,mtry= 8, data=trainData, keep.forest=TRUE, ntree=100)

print(RF)
RF$predicted
RF$importance

# plot (directly prints the important attributes)
varImpPlot(RF)

test_predicted <- predict(RF,test)


library(e1071)
library(caret)
# Accuracy 
confusionMatrix(data=test_predicted,reference=test$Target)

##test_data1
actual_test_predicted <- predict(RF,testData)

print( 'Actual Test Metrics')
View(actual_test_predicted)

#confusionMatrix(data=actual_test_predicted,reference=testData$Target)
testData[order(testData[,1]),]

drops <- c("ID")

testData <- testData[ -c(1)]
actual_test_predicted <- predict(RF,testData)


library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)

library(xgboost)


xgb <- xgboost(data = data.matrix(trainData[,-1]), 
               label = trainData$Target, 
               eta = 0.1,
               max_depth = 15, 
               nround=25, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "merror",
               objective = "multi:softprob",
               num_class = 12,
               nthread = 3
)
