import pandas as pd
import numpy as np
import gradio as gr
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

def one_hot_encoder(df, col): 
    return pd.get_dummies(df[col].apply(lambda x: f"{col} {x}"))

anime = pd.read_csv('anime.csv')
anime.genre = anime.genre.astype(str)
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

def recommend(name):
    name = name.strip().lower()
    title = anime[anime.name.str.contains(name)]
    if len(title) < 1:
        return title.to_html(header=True, index=False)
    index = title.head(1).index[0]
    score = anime_matrix_cosine_df[str(index)].sort_values(ascending=False)
    result =  pd.concat([score.to_frame(), anime[['name', 'rating', 'episodes']]], axis=1)
    result = result.rename(columns={str(index): 'score'}).\
    sort_values(by=['score'], ascending=False)
    result = result[~result.index.isin(title.index)]
    result.score = result.score.apply(lambda x: round(x, 2))
    return result.head(15).to_html(header=True, index=False)

if __name__ == '__main__':
    iface = gr.Interface(fn=recommend, 
                     inputs=gr.inputs.Textbox(placeholder="Enter anime title here...",
                                             label='Title'), 
                     outputs="html",
                     title='Anime Recommendation Engine')
    iface.launch(debug=True)