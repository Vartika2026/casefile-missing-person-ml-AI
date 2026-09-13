import folium
from folium.plugins import MarkerCluster

def build_map(last_lat,last_lon, predicted, trajectories, anomaly_points):
    m = folium.Map(location=[last_lat,last_lon], zoom_start=12, tiles="CartoDB positron")
    folium.Marker([last_lat,last_lon], tooltip="Last known location", icon=folium.Icon(color="red",icon="info-sign")).add_to(m)
    pred_group=folium.FeatureGroup(name="Predicted Areas")
    for p in predicted:
        folium.CircleMarker([p["lat"],p["lon"]], radius=10, tooltip=f'{p["name"]} | Probability {p["probability"]:.1%} | Priority {p["priority"]}',
                            popup=p["explanation"]).add_to(pred_group)
    pred_group.add_to(m)
    fg=folium.FeatureGroup(name="Anomalous Locations")
    mc=MarkerCluster().add_to(fg)
    for lat,lon in anomaly_points[:200]:
        folium.CircleMarker([lat,lon],radius=4,tooltip="Anomalous movement point").add_to(mc)
    fg.add_to(m)
    if len(predicted)>1:
        coords=[[last_lat,last_lon]]+[[p["lat"],p["lon"]] for p in predicted[:4]]
        folium.PolyLine(coords, tooltip="Probable route").add_to(m)
    folium.LayerControl().add_to(m)
    return m
