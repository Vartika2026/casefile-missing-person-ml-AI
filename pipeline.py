from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT/"models"

def load_data():
    traj = pd.read_csv(ROOT/"data/raw/synthetic_gps_trajectories.csv", parse_dates=["timestamp"])
    cases = pd.read_csv(ROOT/"data/synthetic/synthetic_cases.csv", parse_dates=["Last_Seen_Time"])
    return traj, cases

def preprocess(traj):
    df = traj.copy()
    df = df.drop_duplicates()
    df = df[df.latitude.between(-90,90) & df.longitude.between(-180,180)]
    df = df.sort_values(["user_id","timestamp"])
    df["hour"] = df.timestamp.dt.hour
    df["day"] = df.timestamp.dt.day
    df["weekday"] = df.timestamp.dt.dayofweek
    df["month"] = df.timestamp.dt.month
    df["is_weekend"] = (df.weekday >= 5).astype(int)
    return df

def add_features(df):
    out = df.copy()
    g = out.groupby("user_id")
    out["total_distance_km"] = g.distance_km.transform("sum")
    out["avg_speed_kmh"] = g.speed_kmh.transform("mean")
    out["max_speed_kmh"] = g.speed_kmh.transform("max")
    out["movement_frequency"] = g.user_id.transform("size")
    out["lat_roll"] = g.latitude.transform(lambda s: s.rolling(5, min_periods=1).mean())
    out["lon_roll"] = g.longitude.transform(lambda s: s.rolling(5, min_periods=1).mean())
    return out

def train_models(df):
    MODEL_DIR.mkdir(exist_ok=True)
    # Movement clustering
    Xc = df[["latitude","longitude"]].fillna(0)
    scaler_c = StandardScaler().fit(Xc)
    Xcs = scaler_c.transform(Xc)
    km = KMeans(n_clusters=8, random_state=42, n_init=10).fit(Xcs)
    df["cluster"] = km.labels_
    joblib.dump((scaler_c, km), MODEL_DIR/"clustering_model.pkl")

    # Anomaly detection
    Xa = df[["latitude","longitude","distance_km","speed_kmh","hour","weekday"]].fillna(0)
    scaler_a = StandardScaler().fit(Xa)
    iso = IsolationForest(n_estimators=250, contamination=0.03, random_state=42)
    iso.fit(scaler_a.transform(Xa))
    df["anomaly"] = (iso.predict(scaler_a.transform(Xa)) == -1).astype(int)
    df["anomaly_score"] = -iso.decision_function(scaler_a.transform(Xa))
    joblib.dump((scaler_a, iso), MODEL_DIR/"anomaly_model.pkl")

    # Location prediction: target is the area nearest to each synthetic cluster center.
    centers = df.groupby("cluster")[["latitude","longitude"]].mean()
    def nearest_area(lat, lon):
        dist = ((centers.latitude-lat)*111)**2 + ((centers.longitude-lon)*85)**2
        return str(dist.idxmin())
    df["target_cluster"] = [nearest_area(a,b) for a,b in zip(df.latitude,df.longitude)]

    features = ["hour","weekday","is_weekend","latitude","longitude","distance_km","speed_kmh","total_distance_km","avg_speed_kmh"]
    X = df[features].fillna(0)
    y = df["target_cluster"].astype(str)
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    rf = RandomForestClassifier(n_estimators=250,max_depth=14,min_samples_leaf=2,random_state=42,class_weight="balanced")
    rf.fit(Xtr,ytr)
    pred = rf.predict(Xte)
    proba = rf.predict_proba(Xte)
    metrics = {
        "accuracy": accuracy_score(yte,pred),
        "precision": precision_score(yte,pred,average="weighted",zero_division=0),
        "recall": recall_score(yte,pred,average="weighted",zero_division=0),
        "f1": f1_score(yte,pred,average="weighted",zero_division=0),
        "confusion_matrix": confusion_matrix(yte,pred).tolist(),
        "top1": accuracy_score(yte,pred),
        "top3": top_k_accuracy_score(yte,proba,k=min(3,len(rf.classes_)),labels=rf.classes_),
        "top5": top_k_accuracy_score(yte,proba,k=min(5,len(rf.classes_)),labels=rf.classes_),
    }
    joblib.dump((rf,features,metrics), MODEL_DIR/"location_model.pkl")
    return df, metrics

def markov_transitions(df):
    # Cluster sequence per person, transition probability matrix.
    seqs = df.sort_values(["user_id","timestamp"]).groupby("user_id")["cluster"].apply(list)
    trans = {}
    for seq in seqs:
        for a,b in zip(seq[:-1],seq[1:]):
            trans.setdefault(int(a),{})
            trans[int(a)][int(b)] = trans[int(a)].get(int(b),0)+1
    for a in trans:
        total=sum(trans[a].values())
        for b in trans[a]:
            trans[a][b] /= total
    return trans

def predict_route(start_cluster, trans, steps=4):
    route=[int(start_cluster)]
    current=int(start_cluster)
    for _ in range(steps):
        choices=trans.get(current,{})
        if not choices: break
        nxt=max(choices,key=choices.get)
        route.append(int(nxt))
        current=int(nxt)
    return route
