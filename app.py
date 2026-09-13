import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_folium import st_folium

from src.pipeline import load_data, preprocess, add_features, train_models, markov_transitions, predict_route
from src.mapping import build_map

st.set_page_config(page_title="CASEFILE ML Investigation", page_icon="🧭", layout="wide")

@st.cache_data
def prepare():
    traj, cases = load_data()
    df = add_features(preprocess(traj))
    df, metrics = train_models(df)
    return df, cases, metrics

df, cases, metrics = prepare()

st.title("🧭 CASEFILE — AI-Powered Movement Investigation Support")
st.caption("Academic simulation using synthetic case identities and synthetic GPS trajectories. Predictions are probabilistic and are not proof of a person's location.")

with st.sidebar:
    st.header("Case Selection")
    case_id = st.selectbox("Fictional Case", cases.Case_ID)
    case = cases[cases.Case_ID==case_id].iloc[0]
    st.markdown("### Case information")
    st.write(f"**Person:** {case.Person_ID}")
    st.write(f"**Age group:** {case.Age_Group}")
    st.write(f"**Last seen:** {case.Last_Seen_Time}")
    st.write(f"**Day:** {case.Day}")
    st.write(f"**Weather:** {case.Weather}")
    st.write(f"**Usual area:** {case.Usual_Area}")

# Match case to historical person
hist = df[df.user_id==case.Person_ID].copy()
if hist.empty:
    hist=df.copy()

# Prediction using last observation
last = hist.sort_values("timestamp").iloc[-1]
rf, feature_names, _ = __import__("joblib").load(Path(__file__).resolve().parents[1]/"models/location_model.pkl")
feature_row = pd.DataFrame([{
    "hour": pd.Timestamp(case.Last_Seen_Time).hour,
    "weekday": pd.Timestamp(case.Last_Seen_Time).dayofweek,
    "is_weekend": int(pd.Timestamp(case.Last_Seen_Time).dayofweek>=5),
    "latitude": case.Last_Latitude,
    "longitude": case.Last_Longitude,
    "distance_km": hist.distance_km.mean(),
    "speed_kmh": case.Average_Speed,
    "total_distance_km": hist.distance_km.sum(),
    "avg_speed_kmh": case.Average_Speed
}])[feature_names]
probs = rf.predict_proba(feature_row)[0]

cluster_centers = df.groupby("cluster")[["latitude","longitude"]].mean().to_dict("index")
predicted=[]
for c,p in sorted(zip(rf.classes_,probs), key=lambda x:x[1], reverse=True):
    c_int=int(c)
    center=cluster_centers[c_int]
    freq=(df.cluster==c_int).mean()
    route_sim=1.0 if c_int==int(last.cluster) else 0.6
    dist_km=float(np.sqrt(((center["latitude"]-case.Last_Latitude)*111)**2+((center["longitude"]-case.Last_Longitude)*85)**2))
    distance_rel=max(0,1-dist_km/15)
    time_rel=float((df[df.cluster==c_int].hour==pd.Timestamp(case.Last_Seen_Time).hour).mean())
    anomaly=float(df[df.cluster==c_int].anomaly.mean())
    score=100*(0.30*p + 0.20*freq + 0.15*route_sim + 0.15*distance_rel + 0.10*time_rel + 0.10*anomaly)
    if score>=81: priority="Very High"
    elif score>=61: priority="High"
    elif score>=31: priority="Medium"
    else: priority="Low"
    explanation=[]
    if freq>0.10: explanation.append("high historical visit frequency")
    if route_sim>0.8: explanation.append("similar movement pattern")
    if time_rel>0.08: explanation.append("current time matches historical visits")
    if distance_rel>0.6: explanation.append("distance is consistent with historical behavior")
    predicted.append({"name":f"Cluster {c_int}","probability":float(p),"priority":priority,"score":float(score),
                      "lat":center["latitude"],"lon":center["longitude"],
                      "explanation":"; ".join(explanation) or "model probability contributed most to the ranking"})

predicted=sorted(predicted,key=lambda x:x["score"],reverse=True)

tabs=st.tabs(["📍 Probable Locations","🗺️ Map & Route","🚨 Anomalies","📊 Movement Analysis","🔎 Explainability","🧪 Evaluation"])

with tabs[0]:
    st.subheader("Ranked probable areas")
    table=pd.DataFrame(predicted[:8])[["name","probability","score","priority","lat","lon"]]
    table.columns=["Area","Probability","Search Score","Priority","Latitude","Longitude"]
    st.dataframe(table.style.format({"Probability":"{:.1%}","Search Score":"{:.1f}","Latitude":"{:.5f}","Longitude":"{:.5f}"}),use_container_width=True)
    st.info("Search score follows the project guide's suggested weights: ML probability 30%, historical frequency 20%, route similarity 15%, distance relevance 15%, time relevance 10%, anomaly evidence 10%.")

with tabs[1]:
    trans=markov_transitions(df)
    route=predict_route(int(last.cluster),trans,steps=4)
    st.write("**Markov probable cluster route:**"," → ".join(map(str,route)))
    amap=build_map(case.Last_Latitude,case.Last_Longitude,predicted[:5],df,df[df.anomaly==1][["latitude","longitude"]].values.tolist())
    st_folium(amap,width=None,height=650)

with tabs[2]:
    n=int(hist.anomaly.sum())
    st.metric("Historical anomaly points for selected person",n)
    st.write("Isolation Forest flags movement patterns that differ from the learned normal pattern. An anomaly does not automatically mean suspicious or criminal behavior.")
    an=hist[hist.anomaly==1]
    if len(an):
        st.dataframe(an[["timestamp","latitude","longitude","speed_kmh","anomaly_score"]].sort_values("anomaly_score",ascending=False).head(20),use_container_width=True)
    else:
        st.success("No anomaly points were flagged for this synthetic person.")

with tabs[3]:
    c1,c2,c3=st.columns(3)
    c1.metric("Average daily distance (km)",f"{hist.groupby(hist.timestamp.dt.date).distance_km.sum().mean():.2f}")
    c2.metric("Average speed (km/h)",f"{hist.speed_kmh.mean():.2f}")
    c3.metric("Locations/clusters visited",hist.cluster.nunique())
    fig=px.histogram(hist,x="hour",nbins=24,title="Movement frequency by hour")
    st.plotly_chart(fig,use_container_width=True)
    fig2=px.scatter(hist,x="longitude",y="latitude",color="cluster",title="Movement clusters")
    st.plotly_chart(fig2,use_container_width=True)

with tabs[4]:
    st.subheader("Why are the top areas ranked highly?")
    for p in predicted[:5]:
        st.markdown(f"**{p['name']} — {p['priority']} ({p['score']:.1f}/100)**")
        st.write(p["explanation"])
    imp=pd.DataFrame({"Feature":feature_names,"Importance":rf.feature_importances_}).sort_values("Importance",ascending=False)
    st.plotly_chart(px.bar(imp,x="Importance",y="Feature",orientation="h",title="Random Forest feature importance"),use_container_width=True)

with tabs[5]:
    st.subheader("Location model evaluation")
    st.write({k:v for k,v in metrics.items() if k!="confusion_matrix"})
    cm=np.array(metrics["confusion_matrix"])
    st.write("Confusion matrix")
    st.dataframe(pd.DataFrame(cm),use_container_width=False)
    st.caption("Top-k metrics measure whether the true area appears within the model's top 3 or top 5 candidates. Anomaly detection should be evaluated with false-positive/false-negative considerations rather than accuracy alone.")

st.divider()
st.caption("CASEFILE is an academic demonstration. Do not use model output to make real-world decisions about missing persons or to label individuals as suspicious/criminal.")
