library(randomForest)
#library(caret)
#library(ggbiplot)

# module load R/3.6.2


Xz = gzfile('matrix/mat.csv.gz','rt')  
X = read.csv(Xz,header=F, colClasses='integer') 

Yz = gzfile('matrix/classes.csv.gz','rt')  
Y = read.csv(Yz,header=F, colClasses='factor') 

# random Forest
print('loaded data')
dat.rf <- randomForest(
			type='classification',
			X, 
			Y$V1, 
			prox=FALSE,
)
dat.rf
print(importance(dat.rf)

#varImpPlot(dat.rf)

#ImpMeasure<-data.frame(importance(dat.rf))
#ImpMeasure$Vars<-row.names(ImpMeasure)
#important_genus <- ImpMeasure[ImpMeasure$MeanDecreaseGini > 0,2]

#Yhat <- predict(dat.rf, X)

#confusionMatrix(Yhat, Y)
