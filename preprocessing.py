from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def one_hot_encoder(df, col): 
    return pd.get_dummies(df[col].apply(lambda x: f"{col} {x}"))

anime = pd.read_csv('anime.csv')
anime.genre = anime.genre.astype(str)
anime.name = anime.name.apply(lambda x: x.lower()) 
genres = list(anime.genre)
genre_list = [genre.split(',') for genre in genres]
gen = []
for g in genre_list:
    for i in g:
        gen.append(i.strip())
gen = list(set(gen))
for i in gen:
    anime[i] = 0
    anime[i] = np.where(anime.genre.str.contains(i),
                       1,
                       0)
anime_matrix = anime.drop(columns=['genre'])
anime_matrix.type.fillna('-', inplace=True)
anime_matrix = pd.concat([anime_matrix, one_hot_encoder(anime_matrix, 'type')], axis=1).\
drop(columns=['type'])
anime_matrix = anime_matrix[[i for i in anime_matrix.columns if i not in ['name', 'anime_id']]]
anime_matrix.drop(columns='episodes', inplace=True)
anime_matrix.drop(columns='members', inplace=True)
anime_matrix.rating.fillna(0, inplace=True)

scaler = StandardScaler()
scaled_anime_matrix = scaler.fit_transform(anime_matrix)
anime_matrix_cosine = cosine_similarity(scaled_anime_matrix)
anime_matrix_cosine_df = pd.DataFrame(anime_matrix_cosine)
anime_matrix_cosine_df.to_csv('anime_matrix_cosine_df.csv', index=False)