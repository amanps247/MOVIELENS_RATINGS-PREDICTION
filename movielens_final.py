# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.model_selection import train_test_split
from sklearn import metrics
#np.set_printoptions(threshold=100)
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier

#Data acquisition of the movies dataset
df_movie=pd.read_csv('movies.dat', sep = '::', engine='python')
df_movie.columns =['MovieIDs','MovieName','Category']
df_movie.dropna(inplace=True)
df_movie.head()

#Data acquisition of the rating dataset
df_rating = pd.read_csv("ratings.dat",sep='::', engine='python')
df_rating.columns =['ID','MovieID','Ratings','TimeStamp']
df_rating.dropna(inplace=True)
df_rating.head()

#Data acquisition of the users dataset
df_user = pd.read_csv("users.dat",sep='::',engine='python')
df_user.columns =['UserID','Gender','Age','Occupation','Zip-code']
df_user.dropna(inplace=True)
df_user.head()

df = pd.concat([df_movie, df_rating,df_user], axis=1)
df.head()


# =============================================================================
# Master_Data.replace(to_replace ="Weste", value ="Western") 
# Master_Data.replace(to_replace ="Wester", value ="Western") 
# Master_Data.replace(to_replace ="Actio", value ="Action") 
# Master_Data.replace(to_replace ="Adventur", value ="Adventure") 
# Master_Data.replace(to_replace ="Animatio", value ="Animation") 
# Master_Data.replace(to_replace ="Children'", value ="Children's") 
# Master_Data.replace(to_replace ="Come", value ="Comedy") 
# Master_Data.replace(to_replace ="Crim", value ="Crime") 
# Master_Data.replace(to_replace ="Documenta", value ="Documentary") 
# Master_Data.replace(to_replace ="Documentar", value ="Documentary") 
# Master_Data.replace(to_replace ="Dra", value ="Drama") 
# Master_Data.replace(to_replace ="Dram", value ="Drama") 
# Master_Data.replace(to_replace ="Fanta", value ="Fantasy") 
# Master_Data.replace(to_replace ="Fantas", value ="Fantasy") 
# Master_Data.replace(to_replace ="Fil-Noi", value ="Fil-Noir") 
# Master_Data.replace(to_replace ="Horr", value ="Horror") 
# Master_Data.replace(to_replace ="Horro", value ="Horror") 
# Master_Data.replace(to_replace ="Music", value ="Musical") 
# Master_Data.replace(to_replace ="Musica", value ="Musical") 
# Master_Data.replace(to_replace ="Myster", value ="Mystery") 
# Master_Data.replace(to_replace ="Roman", value ="Romance") 
# Master_Data.replace(to_replace ="Romanc", value ="Romance") 
# Master_Data.replace(to_replace ="Sci-F", value ="Sci-Fi") 
# Master_Data.replace(to_replace ="Thrille", value ="Thriller") 
# Master_Data.replace(to_replace ="Wa", value ="War") 
# Master_Data.replace(to_replace ="Horr", value ="Horror") 
# =============================================================================


#Task 1
#Visualize user age distribution
df['Age'].value_counts().plot(kind='barh',alpha=0.7,figsize=(10,10))
plt.show()

df.Age.plot.hist(bins=25)
plt.title("Distribution of users' ages")
plt.ylabel('count of users')
plt.xlabel('Age')

#split dataset into age groups
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
df['age_group'] = pd.cut(df.Age, range(0, 81, 10), right=False, labels=labels)
df[['Age', 'age_group']].drop_duplicates()[:10]
print(df.head())
#Create a histogram for movie
df.Age.plot.hist(bins=25)
plt.title("Movie & Rating")
plt.ylabel('MovieID')
plt.xlabel('Ratings')

#Create a histogram for age
df.Age.plot.hist(bins=25)
plt.title("Age & Rating")
plt.ylabel('Age')
plt.xlabel('Ratings')

#Create a histogram for occupation
df.Age.plot.hist(bins=25)
plt.title("Occupation & Rating")
plt.ylabel('Occupation')
plt.xlabel('Ratings')


#Visualize overall rating by users
df['Ratings'].value_counts().plot(kind='bar',alpha=0.7,figsize=(10,10))
plt.show()




#Task 2
groupedby_movieName = df.groupby('MovieName')
groupedby_rating = df.groupby('Ratings')
groupedby_uid = df.groupby('UserID')
#groupedby_age = df.loc[most_50.index].groupby(['MovieName', 'age_group'])

movies = df.groupby('MovieName').size().sort_values(ascending=True)[:1000]
print(movies)

ToyStory_data = groupedby_movieName.get_group('Toy Story 2 (1999)')
print(ToyStory_data.shape)

#Find and visualize the user rating of the movie “Toy Story”
plt.figure(figsize=(10,10))
plt.scatter(ToyStory_data['MovieName'],ToyStory_data['Ratings'])
plt.title('Plot showing  the user rating of the movie “Toy Story”')
plt.show()

#Find and visualize the viewership of the movie “Toy Story” by age group
print(ToyStory_data[['MovieName','age_group']])




#Task 3
#Find and visualize the top 25 movies by viewership rating
top_25 = df[25:]
top_25['Ratings'].value_counts().plot(kind='barh',alpha=0.6,figsize=(7,7))
plt.show()



#Task 4
#DETAILS FOR USER 2916
#Visualize the rating data by user of user id = 2696
userid_2696 = groupedby_uid.get_group(2696)
print(userid_2696[['UserID','Ratings']])




#Task 5
#the unique features and modelling

df1 = df.iloc[:3882,:]
print(df1.columns)

#Use the following features:movie id,age,occupation
x = df1[['MovieID','Age','Occupation']].values

#Use rating as label
y = df1[['Ratings']].values

#Create train and test data set
train, test, train_labels, test_labels = train_test_split(x,y,test_size=0.33,random_state=42)



# =============================================================================
# #printing side by side
# y_pred_disp = pd.Series(y_pred) 
# y_pred_disp = y_pred_disp.to_numpy(dtype=None, copy=False)
# y_test_disp = y_test.to_numpy(dtype=None, copy=False)
# np.set_printoptions(precision=0)
# print(np.concatenate((y_pred_disp.reshape(len(y_pred),1), y_test_disp.reshape(len(y_test),1)),1))
# 
#rounding(required for regression)
# 
# y_pred_acc=y_pred
# y_pred = regressor.predict(x_test)
# 
# for i in range(0, len(y_pred)): 
#     if ((y_pred[i]- int(y_pred[i]))>=0.1):
#         y_pred[i] = math.ceil(y_pred[i])
#     elif ((y_pred[i]- int(y_pred[i]))<0.5):
#         y_pred[i] = math.floor(y_pred[i])
#         #y_pred[i] = int(y_pred[i] + 1)
# for i in range(0, len(y_pred)): 
#    y_pred[i] = int(y_pred[i]) 
#    
# =============================================================================


# Random Forest

random_forest = RandomForestClassifier(n_estimators=3)
random_forest.fit(train, train_labels)
Y_pred = random_forest.predict(test)
random_forest.score(train, train_labels)

print("accuracy score : ",random_forest.score(train, train_labels)*100)