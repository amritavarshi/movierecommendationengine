import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Readig the csv file - IMDB 5000 MOVIES DATASET
df=pd.read_csv('movie_dataset.csv')
'''print(df.columns)'''

#select features
features=['keywords','cast','genres','director']

for feature in features:
    df[feature]=df[feature].fillna('')

#create a coloumn in dataframe which combines all the above features
def combine_features(row):
    try:
        return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
    except:
        print ("Error",row)

df["combine_features"] = df.apply(combine_features,axis=1)
'''print ("Combined Features",df["combine_features"].head())'''

#create count matrix for all combined features
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combine_features"])

#Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix) 
movie_user_likes = "Avatar"

#get index of the movie from its title
def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
movie_index=get_index_from_title(movie_user_likes)
similar_movies=list(enumerate(cosine_sim[movie_index])) 

'''to get the similar movies in descending order of similarityscore, more similar
movies should come first'''
sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)

#To print titles of first fifty movies
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]
i=0
for movie in sorted_similar_movies:
    print (get_title_from_index(movie[0]))
    i+=1
    if i>50:
        break
    
    
    
    





