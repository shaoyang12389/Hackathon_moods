import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import random

#Preparation du fichier df_segment
st.markdown(
    """
    <style>
    a {
        text-decoration: none;
        color: red;  /* Couleur du texte du lien */
        background-color: none;  /* Couleur de surlignage du lien */
    }
    </style>
    """,
    unsafe_allow_html=True
)

raw = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv")
raw_unique = raw.drop_duplicates(subset = 'track_id', keep = 'first')
track_feature = pd.read_csv("https://raw.githubusercontent.com/shaoyang12389/Hackathon_moods/main/tracks.csv")
moods = pd.read_csv("https://raw.githubusercontent.com/shaoyang12389/Hackathon_moods/main/data_moods.csv")

#image = Image.open('C:\\Users\\91865\\Desktop\\Streamlit Demo\\data\\sal.jpg') 

logo = "https://raw.githubusercontent.com/shaoyang12389/Hackathon_moods/main/Images/verse_logo.jpg"
image_calm = 'https://raw.githubusercontent.com/shaoyang12389/Hackathon_moods/main/Images/calm.JPG'
image_chill = 'https://raw.githubusercontent.com/shaoyang12389/Hackathon_moods/main/Images/chillout.JPG'
image_energetic = 'https://raw.githubusercontent.com/shaoyang12389/Hackathon_moods/main/Images/energetic.JPG'
image_explicit = 'https://raw.githubusercontent.com/shaoyang12389/Hackathon_moods/main/Images/explicit.JPG'
image_happy = 'https://raw.githubusercontent.com/shaoyang12389/Hackathon_moods/main/Images/happy.JPG'
image_party = 'https://raw.githubusercontent.com/shaoyang12389/Hackathon_moods/main/Images/party.JPG'
image_road = 'https://raw.githubusercontent.com/shaoyang12389/Hackathon_moods/main/Images/road.JPG'
image_sad = 'https://raw.githubusercontent.com/shaoyang12389/Hackathon_moods/main/Images/sad.JPG'
image_sport = 'https://raw.githubusercontent.com/shaoyang12389/Hackathon_moods/main/Images/sport.JPG'

st.image(logo, width=200)
st.subheader("Bienvenue dans un univers de recommandations musicales \U0001F3B5")


tab1, tab2 = st.tabs(["Sur le pouce", "Recommandation"])



#Titre et centrage
#st.title("\t Titre centré")


# Sport
q1_energy = raw_unique['energy'].quantile(0.25)
q3_energy = raw_unique['energy'].quantile(0.75)
iqr_energy = q3_energy - q1_energy
lower_bound_energy = q1_energy - 1.5 * iqr_energy
upper_bound_energy = q3_energy + 1.5 * iqr_energy

raw_sport = raw_unique[(raw_unique["energy"] >= q3_energy) & (raw_unique["energy"] < upper_bound_energy)]

#-----------------------------------------

q1_loudness = raw_unique['loudness'].quantile(0.25)
q3_loudness = raw_unique['loudness'].quantile(0.75)
iqr_loudness = q3_loudness - q1_loudness
lower_bound_loudness = q1_loudness - 1.5 * iqr_loudness
upper_bound_loudness = q3_loudness + 1.5 * iqr_loudness

raw_sport = raw_sport[(raw_sport["loudness"] >= q3_loudness) & (raw_sport["loudness"] < upper_bound_loudness)]

#-----------------------------------------

q1_acousticness = raw_unique['acousticness'].quantile(0.25)
q3_acousticness = raw_unique['acousticness'].quantile(0.75)
iqr_acousticness = q3_acousticness - q1_acousticness
lower_bound_acousticness = q1_acousticness - 1.5 * iqr_acousticness
upper_bound_acousticness = q3_acousticness + 1.5 * iqr_acousticness

sport = raw_sport[(raw_sport["acousticness"] <= q1_acousticness) & (raw_sport["acousticness"] > lower_bound_acousticness)]
sport['URL'] = 'https://open.spotify.com/intl-fr/track/' + sport['track_id']

# Explicit

tracks_explicit = pd.DataFrame(track_feature, columns = ['id', 'explicit'])
raw_explicit = pd.merge(raw_unique,
                        tracks_explicit,
                        left_on = 'track_id',
                        right_on = 'id',
                        how = 'inner')

q1_explicit = raw_explicit['loudness'].quantile(0.25)
q3_explicit = raw_explicit['loudness'].quantile(0.75)
iqr_explicit = q3_explicit - q1_explicit
lower_bound_explicit = q1_explicit - 1.5 * iqr_explicit
upper_bound_explicit = q3_explicit + 1.5 * iqr_explicit

enerve = raw_explicit[(raw_explicit["explicit"] == True) &  (raw_explicit['loudness']> q1_explicit) & (raw_explicit["loudness"] < upper_bound_explicit)]
enerve['URL'] = 'https://open.spotify.com/intl-fr/track/' + enerve['track_id']

# Chill

q1_valence = raw_unique['valence'].quantile(0.25)
q3_valence = raw_unique['valence'].quantile(0.75)
iqr_valence = q3_valence - q1_valence
lower_bound_valence = q1_valence- 1.5 * iqr_valence
upper_bound_valence = q3_valence + 1.5 * iqr_valence

raw_chill = raw_unique[(raw_unique["valence"] >= q3_valence) & (raw_unique["valence"] < upper_bound_valence)]

#-----------------------------------------

q1_energy = raw_unique['energy'].quantile(0.25)
q3_energy = raw_unique['energy'].quantile(0.75)
iqr_energy = q3_energy - q1_energy
lower_bound_energy = q1_energy - 1.5 * iqr_energy
upper_bound_energy = q3_energy + 1.5 * iqr_energy

raw_chill = raw_chill[(raw_chill["energy"] < q1_energy) & (raw_chill["energy"] > lower_bound_energy)]

#-----------------------------------------

q1_acousticness = raw_unique['acousticness'].quantile(0.25)
q3_acousticness = raw_unique['acousticness'].quantile(0.75)
iqr_acousticness = q3_acousticness - q1_acousticness
lower_bound_acousticness = q1_acousticness- 1.5 * iqr_acousticness
upper_bound_acousticness = q3_acousticness + 1.5 * iqr_acousticness

raw_chill = raw_chill[(raw_chill["acousticness"] >= q3_acousticness) & (raw_chill["acousticness"] < upper_bound_acousticness)]

#-----------------------------------------

chill = raw_chill[(raw_chill["tempo"] < 110) &  (raw_chill["tempo"] > 90)]
chill['URL'] = 'https://open.spotify.com/intl-fr/track/' + chill['track_id']

# Party

q1_danceability = raw_unique['danceability'].quantile(0.25)
q3_danceability = raw_unique['danceability'].quantile(0.75)
iqr_danceability = q3_danceability - q1_danceability
lower_bound_danceability = q1_danceability- 1.5 * iqr_danceability
upper_bound_danceability = q3_danceability + 1.5 * iqr_danceability

raw_danceability = raw_unique[(raw_unique["danceability"] >= q3_danceability) & (raw_unique["danceability"] < upper_bound_danceability)]

#-----------------------------------------

q1_energy = raw_unique['energy'].quantile(0.25)
q3_energy = raw_unique['energy'].quantile(0.75)
iqr_energy = q3_energy - q1_energy
lower_bound_energy = q1_energy- 1.5 * iqr_energy
upper_bound_energy = q3_energy + 1.5 * iqr_energy

raw_danceability = raw_danceability[(raw_danceability["energy"] >= q3_energy) & (raw_danceability["energy"] < upper_bound_energy)]

#-----------------------------------------

party = raw_danceability[(raw_danceability["tempo"] > 120) ]
party['URL'] = 'https://open.spotify.com/intl-fr/track/' + party['track_id']


q1_energy = raw_unique['energy'].quantile(0.25)
q3_energy = raw_unique['energy'].quantile(0.75)
iqr_energy = q3_energy - q1_energy
lower_bound_energy = q1_energy - 1.5 * iqr_energy
upper_bound_energy = q3_energy + 1.5 * iqr_energy

raw_road = raw_unique[(raw_unique["energy"] >= q3_energy) & (raw_unique["energy"] < upper_bound_energy)]

#-----------------------------------------

q1_road = raw_road['valence'].quantile(0.25)
q3_road = raw_road['valence'].quantile(0.75)
iqr_road = q3_road - q1_road
lower_bound_road = q1_road- 1.5 * iqr_road
upper_bound_road = q3_road + 1.5 * iqr_road

subgenre_list = ['neo soul', 'album rock', 'classic rock', 'indie poptimism', 'hard rock', 'hip pop', 'southern hip hop']
road = raw_road[raw_road["playlist_subgenre"].isin(subgenre_list) & (raw_road["valence"] >= q3_road) & (raw_road["valence"] < upper_bound_road)]

#-----------------------------------------

road['URL'] = 'https://open.spotify.com/intl-fr/track/' + road['track_id']


# ACCESS_TOKEN = '60e001f56cb94dc5b4f4aa7999424a8b'

# # Faites une requête GET à l'API de Spotify pour obtenir les albums liés à votre compte
# headers = {
#     'Authorization': f'Bearer {ACCESS_TOKEN}'
# }

# url = 'https://api.spotify.com/v1/me/shows?limit=5'  # Vous pouvez ajuster le nombre d'albums à afficher en changeant le paramètre limit

# response = requests.get(url, headers=headers)

# if response.status_code == 200:
#     data = response.json()
#     albums = data['items']

    
with tab1:
    liste_humeur = ['sport', 'chill', 'enerve', 'party', 'road']
    choix_humeur = st.selectbox('Quelle est votre humeur ?',(liste_humeur), index=0)
    if choix_humeur == 'sport':
        st.image(image_sport, use_column_width=True)
        for index, row in sport.iterrows():
            song_sport = row['track_name']
            artist_name = row['track_artist']
            song_url = row['URL']
        #st.write(f"{song_name} - {artist_name}")
            #if st.button(f"trsc{song_sport}"):
            st.markdown(f'<a href="{song_url}" target="_blank">{song_sport} - {artist_name}</a>', unsafe_allow_html=True)
    #st.table(sport)
    elif choix_humeur == 'chill':
        st.image(image_chill, use_column_width=True)
        for index, row in chill.iterrows():
            song_chill = row['track_name']
            artist_name = row['track_artist']
            song_url = row['URL']
        #st.write(f"{song_name} - {artist_name}")
            #if st.button(f"{song_chill}"):
            st.markdown(f'<a href="{song_url}" target="_blank">{song_chill} - {artist_name}</a>', unsafe_allow_html=True)
    #st.table(chill)
    elif choix_humeur == 'enerve':
        st.image(image_explicit, use_column_width=True)
        for index, row in enerve.iterrows():
            song_nrv = row['track_name']
            artist_name = row['track_artist']
            song_url = row['URL']
        #st.write(f"{song_name} - {artist_name}")
            #if st.button(f"{song_rnv}"):
            st.markdown(f'<a href="{song_url}" target="_blank">{song_nrv} - {artist_name}</a>', unsafe_allow_html=True)
    #st.table(enerve)
    elif choix_humeur == 'party':
        st.image(image_party, use_column_width=True)
        for index, row in party.iterrows():
            song_party = row['track_name']
            artist_name = row['track_artist']
            song_url = row['URL']
        #st.write(f"{song_name} - {artist_name}")
            #if st.button(f"Track {song_party}"):
            st.markdown(f'<a href="{song_url}" target="_blank">{song_party} - {artist_name}</a>', unsafe_allow_html=True)
    #st.table(party)
    elif choix_humeur == 'road':
        st.image(image_road, use_column_width=True)
        for index, row in road.iterrows():
            song_road = row['track_name']
            artist_name = row['track_artist']
            song_url = row['URL']
            st.markdown(f'<a href="{song_url}" target="_blank">{song_road} - {artist_name}</a>', unsafe_allow_html=True)
        #st.write(f"{song_name} - {artist_name}")
            #if st.button(f"{song_party}"):
                #st.markdown(f'<a href="{song_url}" target="_blank">Ecouter sur Spotify</a>', unsafe_allow_html=True)
    # row_count = 5
    # url = road.loc[row_count, 'URL']
    # st.markdown(f'<a href="{url}" target="_blank"><button>Play</button></a>', unsafe_allow_html=True)


# b6db5377f2c144c0b879d16448fe596c
# 60e001f56cb94dc5b4f4aa7999424a8b


    # Afficher les boutons d'albums et les pochettes d'albums correspondantes
    # for album in albums:
    #     album_name = album['name']
    #     album_image_url = album['external_urls']['spotify']
    #     if st.button(f"Écouter {album_name}"):
    #         st.image(album_image_url, caption=album_name, use_column_width=True)
    #st.table(road)
    # for index, row in road.iterrows():
    #    song_url = "E:/CREATIONS/WCS/Verse/NeverGonnaGiveYouUp.mp3"
    #    st.audio(song_url, format='audio/mp3', start_time=0)

# Intégrer API youtobe (Token expiré)
#def get_youtube_video_url_by_title(api_key, title):
    #youtube = build('youtube', 'v3', developerKey=api_key)

    # Rechercher des vidéos en utilisant le titre de la chanson
    #search_response = youtube.search().list(
        #q=title,
        #part='id',
        #type='video',
        #maxResults=1).execute()

    #if 'items' in search_response:
        # Récupérer l'ID de la première vidéo trouvée
        #video_id = search_response['items'][0]['id']['videoId']
        # Construire l'URL de la vidéo
        #video_url = f'https://www.youtube.com/watch?v={video_id}'
        #return video_url

    #return None

with tab2:
# Définir des variables
    X = moods[['danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'valence', 'loudness', 'tempo']]
    y = moods['mood']

    #Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled)
    X_scaled.columns = X.columns

    # Création de modèle
    model_knn = KNeighborsClassifier(weights="distance", n_neighbors=200, metric = 'cosine')

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42, train_size=0.75)
    model_knn.fit(X_train, y_train)

    # Définir X de raw
    X_raw = raw_unique[['danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'valence', 'loudness', 'tempo']]

    # Standariser les X_raw
    X_raw_scaled = StandardScaler().fit_transform(X_raw)
    X_raw_scaled = pd.DataFrame(X_raw_scaled)
    X_raw_scaled.columns = X_raw.columns

    # Trouver des somme de vecteur pour chaque catégorie
    # Pour "happy"
    mood_happy = moods[moods["mood"]=='Happy']
    index_happy = mood_happy.index.tolist()
    mood_scaled_happy = X_scaled[X_scaled.index.isin(index_happy)]
    vec_happy = pd.DataFrame(mood_scaled_happy.sum() / len(mood_scaled_happy)).T

    # Pour "Sad"
    mood_sad = moods[moods["mood"]=='Sad']
    index_sad = mood_sad.index.tolist()
    mood_scaled_sad = X_scaled[X_scaled.index.isin(index_sad)]
    vec_sad = pd.DataFrame(mood_scaled_sad.sum() / len(mood_scaled_sad)).T

    # Pour "Energetic"
    mood_energetic = moods[moods["mood"]=='Energetic']
    index_energetic = mood_energetic.index.tolist()
    mood_scaled_energetic = X_scaled[X_scaled.index.isin(index_energetic)]
    vec_energetic = pd.DataFrame(mood_scaled_energetic.sum() / len(mood_scaled_energetic)).T

    # Pour "Calm"
    mood_calm = moods[moods["mood"]=='Calm']
    index_calm = mood_calm.index.tolist()
    mood_scaled_calm = X_scaled[X_scaled.index.isin(index_calm)]
    vec_calm = pd.DataFrame(mood_scaled_calm.sum() / len(mood_scaled_calm)).T

    def sentiment_song(emotion):
        if emotion.lower() == "happy":
            vec = vec_happy
        if emotion.lower() == "sad":
            vec = vec_sad
        if emotion.lower() == "energetic":
            vec = vec_energetic
        if emotion.lower() == "calm":
            vec = vec_calm
        # Recommandation selon la somme vectorielle
        reco = model_knn.kneighbors(vec)[1][0]
        random_reco = random.sample(list(reco), 50)
        reco_50 = raw_unique.iloc[random_reco,[1,2,5]]
        reco_50 = reco_50.reset_index(drop=True)
        reco_50.rename(columns={"track_name": "Titre de chanson", "track_artist": "Artiste", "track_album_name": "Titre d'album"}, inplace=True)

        return reco_50
    liste_emotion = ['Happy', 'Sad', 'Energetic', 'Calm']
    choix_emotion = st.selectbox('Quelle est votre emotion ?',(liste_emotion), index=0)
    if st.button("Obtenir des recommandations"):
        result = sentiment_song(choix_emotion)
        if result is not None:
            st.subheader(f"Recommandations pour l'émotion '{choix_emotion}':")
            st.dataframe(result)

# Youtobe API (Expirer)
            #api_key = 'AIzaSyD3PVXDQlCv1ppgo2eDsCg5gRusMmvsy4o'
            #song_title = result.iloc[0]["Titre de chanson"]
            #song_artist = result.iloc[0]["Artiste"]

            #title = song_title + " - " + song_artist

            #video_url = get_youtube_video_url_by_title(api_key, title)


        #if video_url:
            #st.video(video_url)
        #else:
            #st.warning(f"Aucune vidéo trouvée pour '{song_title}'.")
        #st.dataframe(result)

# if choix_humeur == 'happy':
#     result = sentiment_song("happy")
#     st.dataframe(result)


