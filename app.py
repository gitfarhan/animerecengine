import pandas as pd
import numpy as np
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

anime = pd.read_csv('anime.csv')
anime.name = anime.name.lower() 
anime_matrix_cosine_df = pd.read_csv('anime_matrix_cosine_df.csv')

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