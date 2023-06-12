import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error


print("Loading the Original Dataset.....")
df=pd.read_csv("AAPL.csv")
print(df.head(50))

#now the data we are going for is the "close" column which will be used for predictions
#now we should know that data being fed to stacked LSTM should be scaled
print("Loading the close column and plotting the figure.....")
df1=df.reset_index()['close']
print(df1)

plt.figure()
plt.plot(df1)
plt.savefig("plot.png")

scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
# print(df1)

# now we are going to split the data into train and test, but we should know that
# this is sequence data where data of a daya depends on previous days
# therefore our split should be different
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data=df1[0:training_size,:]
test_data=df1[training_size:len(df1),:1]

#now we need to also divide this training dataset into X_train and Y_train and testing to
#X_test and y_test

def create_dataset(dataset,time_step=1):
    X_data=[]
    y_data=[]

    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        b=dataset[i+time_step,0]
        X_data.append(a)
        y_data.append(b)

    return np.array(X_data),np.array(y_data)

print("Creating the training dataset....")
time_step=100
X_train,y_train=create_dataset(train_data,time_step)
X_test,y_test=create_dataset(test_data,time_step)

# print(y_train)
# print(y_test)

#LSTM accepts dimensional tobe 3-D (samples,time_step,fetaures) (N,T,D)
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

# NOW creating a stacked LSTM model as our data is ready to be fed
print("Loading the model......")
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer="adam")
print(model.summary())

print("Fitting the model....")
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)

#now let's do the prediction 
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

#now we scale back the values to the original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

print("Mean squared error for y_train and train_predict ",math.sqrt(mean_squared_error(y_train,train_predict)))
print("Mean squared error for y_test and test_predict ",math.sqrt(mean_squared_error(y_test,test_predict)))

#now plotting the plot for the testing dataset and how it performed 

#shift train predictions for plotting
look_back=100
trainPredictplot=np.empty_like(df1)
trainPredictplot[:,:]=np.nan
trainPredictplot[look_back:len(train_predict)+look_back, :] = train_predict

# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictplot)
plt.plot(testPredictPlot)
plt.savefig("plot2.png")
plt.show()

model.save("Stock_predictor.model", save_format="h5")
#now we are done with the testing part of the dataset and test predictions
#therefore now we move forward with the forecasting part and predict the data 
#or stock prices for the next 30 days

#we know the length of the test_data is 441 therefore to predict for the new date
#after the last data we need the record of thelast 100 days to predict the next day

x_input=test_data[341:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()
print(temp_input)
#now we need to write a logic for the next 30 days prediction

list_output=[]
n_steps=100
i=0

while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        list_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        list_output.extend(yhat.tolist())
        i=i+1
    

print(list_output)

#now plotting the graphs

day_new=np.arange(1,101)
day_pred=np.arange(101,131)

plt.plot(day_new,scaler.inverse_transform(df1[1158:]))
plt.plot(day_pred,scaler.inverse_transform(list_output))
plt.savefig("plot3.png")
plt.show()

df3=df1.tolist()
df3.extend(list_output)
plt.plot(df3[1200:])
plt.savefig("plot4.png")
plt.show()

df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)
plt.savefig("plot5.png")
plt.show()
