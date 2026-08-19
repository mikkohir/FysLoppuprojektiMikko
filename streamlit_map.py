import streamlit as st
import pandas as pd # datan tuonti, käsittely ja tallennus
import numpy as np # matemaattiset tsydeemit - käy läpi teoriatunnit uudelleen
import folium # ja tätä visualisointiin matplotlibin sijaan!
from scipy.signal import butter,filtfilt
from streamlit_folium import st_folium
from math import radians, cos, sin, asin, sqrt




st.title("Päivän kävelymatka")

# Funktio suodatukseen, käytetään alipäästösuodatinta
def butter_lowpass_filter(data, cutoff, fs, nyq, order):
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Haversinen kaava

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r

# Luetaan .csv-tiedostot

df = pd.read_csv("Linear Acceleration.csv")
gdf = pd.read_csv("Location.csv")

# Parametrit

f = df['Linear Acceleration x (m/s^2)'] 
t = df['Time (s)']
T = df['Time (s)'].max()
n = len(t)
fs = n/T
nyq = fs/2
order = 3
cutoff = 1/0.5

# Suodatin

df['Suodatettu a_x (m/s^2)'] = butter_lowpass_filter(f, cutoff, fs, nyq, order)
filtered_signal = df['Suodatettu a_x (m/s^2)']

st.title("Kiihtyvyysdata")

# Kiihtyvyysdatan komponenteista valittu x, sillä liike tuntui näkyvän siinä parhaiten.
# Suodatetun kiihtyvyysdatan kuvaaja

st.subheader("Suodatettu kiihtyvyysdata, x-komponentti")

df = df.rename(columns={"Time (s)": "Aika (s)"})
st.line_chart(
    df,
    x="Aika (s)",
    y=["Suodatettu a_x (m/s^2)"]
)


st.subheader("Tulokset")


# Askelten määrä laskettuna nollakohtien ylityksen perusteella

jaksot = 0

for i in range(n-1):
    if filtered_signal[i] / filtered_signal[i+1] < 0:
        jaksot += 1

askeleet = jaksot / 2

st.write("Askelten määrä:", round(askeleet))


# Fourier-analyysi, ja askelten määrä Fourier-muunnoksen perusteella

N = len(f)
fourier = np.fft.fft(f,N) # Fourier
psd = fourier*np.conj(fourier)/N # Tehospektri
dt = t[1]-t[0] # Näytteenottoväli
freq = np.fft.fftfreq(N,dt)
L = np.arange(1,int(N/2))
f_max = freq[L][psd[L] == np.max(psd[L])][0] 


askelmäärä = np.max(t)*f_max

st.write("Askelten määrä fourier-muunnoksella: ", round(askelmäärä))


# Kokonaismatka, keskinopeus sekä askelpituus

matka = np.zeros(len(gdf)) 
aikaEro = np.zeros(len(gdf)) 


for i in range(len(gdf)-1):
    matka[i] = haversine(gdf['Longitude (°)'][i],gdf['Latitude (°)'][i],gdf['Longitude (°)'][i+1],gdf['Latitude (°)'][i+1])
    aikaEro[i] = gdf['Time (s)'][i+1] - gdf['Time (s)'][i]

aika = np.sum(aikaEro) 
kokonaisMatka = np.sum(matka)
keskiNopeus = np.divide(kokonaisMatka * 1000, aika)
askelPituus = np.divide(kokonaisMatka * 1000, askeleet)


st.write("Kokonaismatka :", round(kokonaisMatka, 2),"km")
st.write("Keskinopeus :", round(keskiNopeus,2),"m/s")
st.write("Askelen pituus :", round(askelPituus,1),"m")


# Tehospektri

st.title("Tehospektri")

chart_data = pd.DataFrame(np.transpose(np.array([freq[L],psd[L].real])), columns=["freq", "psd"])
st.line_chart(chart_data, x = 'freq', y = 'psd' , y_label = 'Teho',x_label = 'Taajuus (Hz)')


# Piirretään kartta

start_lat = gdf['Latitude (°)'].mean()
start_long = gdf['Longitude (°)'].mean()
map = folium.Map(location = [start_lat,start_long], zoom_start = 23)

folium.PolyLine(gdf[['Latitude (°)','Longitude (°)']], color = 'red', weight = 4, opacity = 1).add_to(map)

st_map = st_folium(map, width=1200, height=650)


