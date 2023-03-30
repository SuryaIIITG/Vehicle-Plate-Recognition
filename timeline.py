from datetime import datetime
import  time
import pandas as pd

def car_time():   
     
#Data stored in csv file
    raw_data={'date':[time.asctime(time.localtime(time.time()))],'vehicle_number':[text]} 
    df=pd.DataFrame(raw_data,columns=['date','vehicle_number'])
    #df.to_csv('data.csv') 


