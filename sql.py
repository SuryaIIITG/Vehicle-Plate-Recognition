# importing required libraries


import mysql.connector


# dataBase = mysql.connector.connect(
# host ="localhost",
# user ="root",
# passwd ="root12345",database="anpr_db")

# print("Connection has established",dataBase)

# db = dataBase.cursor()

# create database
#db.execute("CREATE DATABASE ANPR_db")

# create table into anpr_db
#db.execute("create table number_data(time DATETIME(6),Number varchar(20))")
#print("Table created! ")  

   # Function for insert value into database
def insert_value(dateTime,Number_plate):
    dataBase = mysql.connector.connect(
    host ="localhost",
    user ="root",
    passwd ="root12345",database="anpr_db")

    print("Connection has established",dataBase)

    db = dataBase.cursor()
    db.execute("insert into number_data(time,Number) values(%s,%s)",(dateTime,Number_plate)) 
    print("Data inserted")
    dataBase.commit()


# insert_value('2023-07-06 14:04:25','UP32NH3369')



# for table show
"""
db.execute("show tables")
for i in db:
    print(i)

    """

# Inserting values into database

#db.execute("insert into number_data (time,Number) values(%s,%s)",(item1,item3))
#db.execute("insert into number_data (time,Number) values(%s,%s)",('2022-05-15 11:55:12','UP43HG6789'))
# dataBase.commit()

# Disconnecting from the server
#dataBase.close()
