
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, stop_after_attempt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib
from flask import Flask, request, jsonify
import config
import os
import time

app = Flask(__name__)
# ==========================
# Autenticação no Spotify
# ==========================

api_key = config.LASTFM_API_KEY
api_url = config.LASTFM_API_URL


# Gerenciamento de taxa
rate_limit_delay = 30  # 30 segundos de atraso para evitar erros 429

def rate_limited_request(func, *args, **kwargs):
    max_retries = 5
    delay = rate_limit_delay

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            if e.response and e.response.status_code == 429:  # Erro de limite de requisição
                if attempt < max_retries - 1:
                    print(f"Rate limit hit. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Espera exponencial
                else:
                    raise
            else:
                raise

def get_track_data(track_name, artist_name):
    params = {
        'method': 'track.getInfo',
        'api_key': api_key,
        'artist': artist_name,
        'track': track_name,
        'format': 'json'
    }
    response = rate_limited_request(requests.get, api_url, params=params)
    track_info = response.json()

    if 'track' in track_info:
        track = track_info['track']
        track_data = {
            'track_id': track.get('id', 'N/A'),  # Usamos um identificador adequado, não a URL
            'artist': artist_name,
            'track': track.get('name', 'Unknown'),
            'album': track.get('album', {}).get('title', 'Unknown'),
            'listeners': int(track.get('listeners', 0)),
            'playcount': int(track.get('playcount', 0)),
            'tags': ', '.join(tag['name'] for tag in track.get('toptags', {}).get('tag', [])),
            'genre': ', '.join(tag['name'] for tag in track.get('toptags', {}).get('tag', []))
        }
        return track_data
    return None

def add_track_to_csv(track_data, csv_path):
    df = pd.read_csv(csv_path)
    if not ((df['track'] == track_data['track']) & (df['artist'] == track_data['artist'])).any():
        new_df = pd.DataFrame([track_data])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(csv_path, index=False)
        return True  # Indica que a música foi adicionada
    return False  # Indica que a música já existia

def train_model(csv_path):
    df = pd.read_csv(csv_path)
    # Use as colunas que têm valores numéricos para o treinamento
    features = df[['listeners', 'playcount']].fillna(0).values
    model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    model.fit(features)
    joblib.dump(model, 'music_recommender_model.pkl')
    return model

def recommend_songs(track_data, model, df):
    features = [[
        track_data.get('listeners', 0), track_data.get('playcount', 0)
    ]]
    
    features = pd.DataFrame(features).fillna(0).values  # Garantir que não haja NaN
    distances, indices = model.kneighbors(features)

    # Obter os índices das músicas recomendadas
    recommended_indices = indices[0]

    # Filtrar o DataFrame para não incluir a música solicitada
    df_filtered = df[~((df['track'] == track_data['track']) & 
                       (df['artist'] == track_data['artist']))]
    
    recommendations = []
    for idx in recommended_indices:
        if idx < len(df_filtered):
            rec_track = df_filtered.iloc[idx]
            recommendations.append({
                'track': rec_track['track'],
                'artist': rec_track['artist']
            })
    
    return recommendations

if os.path.exists('music_recommender_model.pkl'):
    model = joblib.load('music_recommender_model.pkl')
else:
    model = train_model('songs.csv')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    track_name = data.get('track_name')
    artist_name = data.get('artist_name')

    if not track_name or not artist_name:
        return jsonify({'error': 'track_name and artist_name are required'}), 400

    track_data = get_track_data(track_name, artist_name)
    if not track_data:
        return jsonify({'error': 'Track not found'}), 404

    if add_track_to_csv(track_data, 'songs.csv'):
        global model
        model = train_model('songs.csv')  # Re-treina o modelo se uma nova música for adicionada
    
    df = pd.read_csv('songs.csv')
    recommendations = recommend_songs(track_data, model, df)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)