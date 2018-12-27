
library(corrgram)
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(usdm)
library(DMwR)
library(dplyr)

rm(list=ls())
setwd("F:/Data Analytics/Edwisor Project/Bike Rental Project/R")
#getwd()
bk_data=read.csv("bike_rental.csv")
bk_factors=c('season','yr','mnth','holiday','weekday','workingday','weathersit')
bk_numeric=c('temp','atemp','hum','windspeed','casual','registered','cnt')
# Seperating out categorical features
for (i in bk_factors) 
{
  bk_data[,i]=as.factor(bk_data[,i])
}

# EXPLORATORY DATA ANALYSIS
#seasonal variation of count
ggplot(bk_data,aes(x=instant,y=cnt,colour=season))+
  geom_point()+geom_smooth(method='lm')+
  labs(title="seasonal variation of count")

#hum vs count for various seasons
ggplot(bk_data,aes(x=hum,y=cnt))+
  geom_point()+geom_smooth(method='lm')+
  facet_wrap(~season)

#weather vs cnt
ggplot(bk_data,aes(x=weathersit,y=cnt))+
  geom_bar(fill='red',stat='identity')+theme_bw()+
  scale_x_discrete(breaks=c(1,2,3),
                   labels=c("clear","misty","light rain"))+
  labs(x='weather',title="weather vs count")

# boxplot Visualisation
par(mfrow=c(2,4))
for (i in bk_numeric)
{
  boxplot(bk_data[,i],xlab=c(i))
}
par(mfrow=c(1,1))
# Outlier Removal
for (i in bk_numeric)
{
  outliers = bk_data[,i][bk_data[,i] %in% boxplot.stats(bk_data[,i])$out]
  bk_data = bk_data[which(!bk_data[,i] %in% outliers),]
}
#write.csv(bk_data,"bk_data_outlier.csv")

# Selecting numerical data and performing correlation analysis
bk_data_num=dplyr::select(bk_data,bk_numeric)
corrgram(bk_data_num, order = F,upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

# Feature Engineering
# we can see that temp and atemp are highly correlated. so atemp is removed
# instant is removed as it is merely an index and dteday is removed as we have seperate month and day columns

col_rem=c('instant','atemp','dteday')
bk_data=dplyr::select(bk_data,-col_rem)

#MODEL PHASE

# TRAIN AND TEST DATA SAMPLING
train_sample=bk_data[sample(nrow(bk_data),nrow(bk_data)*0.8),]
train_index=as.numeric(rownames(train_sample))
test_sample=bk_data[-train_index,]

# FUNCTIONS FOR MODEL BUILDING AND MAPE
final_model = function(model1,model2){
  Prediction_1 = data.frame(predict(model1, test_sample[,1:10]))
  Prediction_2 = data.frame(predict(model2, test_sample[,1:10]))
  Prediction=cbind(Prediction_1,Prediction_2)
  colnames(Prediction)=c('casual','registered')
  Prediction=mutate(Prediction,cnt=casual+registered)
  return(Prediction)
}

mape= function(actual,predicted) {(mean(abs(actual-predicted)/actual)*100)
}

# RANDOM FOREST
rf_model_1=randomForest(casual~season+yr+mnth+holiday+weekday+workingday+weathersit+temp+hum+windspeed,train_sample,importance = TRUE, ntree = 500)
rf_model_2=randomForest(registered~season+yr+mnth+holiday+weekday+workingday+weathersit+temp+hum+windspeed,train_sample,importance = TRUE, ntree = 500)
RF_predictions=final_model(rf_model_1,rf_model_2)

print("RANDOM FOREST",quote=F)
mape(test_sample$casual,RF_predictions$casual)
mape(test_sample$registered,RF_predictions$registered)
mape(test_sample$cnt,RF_predictions$cnt)

# DECISION TREE  
dec_model_1=rpart(casual~season+yr+mnth+holiday+weekday+workingday+weathersit+temp+hum+windspeed,data=train_sample,method="anova")
dec_model_2=rpart(registered~season+yr+mnth+holiday+weekday+workingday+weathersit+temp+hum+windspeed,data=train_sample,method="anova")
dec_predictions=final_model(dec_model_1,dec_model_2)

print("DECISION TREE",quote=F)
mape(test_sample$casual,dec_predictions$casual)
mape(test_sample$registered,dec_predictions$registered)
mape(test_sample$cnt,dec_predictions$cnt)


#Visualisation of Decision Tree
rpart.plot(dec_model_1, box.palette="RdBu", shadow.col="gray", nn=TRUE)
rpart.plot(dec_model_2, box.palette="RdBu", shadow.col="gray", nn=TRUE)

# LINEAR REGRESSION

# checking for multicollinearity problem
vifcor(bk_data[,c('hum','temp','windspeed')], th = 0.9)

lm_model_1 = lm(casual~season+yr+mnth+holiday+weekday+workingday+weathersit+temp+hum+windspeed, data = train_sample)
lm_model_2 = lm(registered~season+yr+mnth+holiday+weekday+workingday+weathersit+temp+hum+windspeed, data = train_sample)
reg_predictions=final_model(lm_model_1,lm_model_2)

print("LINEAR REGRESSION",quote=F)
mape(test_sample$casual,reg_predictions$casual)
mape(test_sample$registered,reg_predictions$registered)
mape(test_sample$cnt,reg_predictions$cnt)

#KNN
bk_data_KNN=bk_data
bk_data_KNN[-train_index,c("casual","registered","cnt")]=NA
bk_data_KNN=knnImputation(bk_data_KNN,k=3)

print("KNN",quote=F)
mape(test_sample$casual,bk_data_KNN[-train_index,"casual"])
mape(test_sample$registered,bk_data_KNN[-train_index,"registered"])
mape(test_sample$cnt,bk_data_KNN[-train_index,"cnt"])
