import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(42)

# Synthetic city-like coordinate system around Beijing.
areas = {
    "A_Central": (39.9042, 116.4074),
    "B_West": (39.9150, 116.3500),
    "C_North": (39.9550, 116.3900),
    "D_East": (39.9100, 116.4550),
    "E_South": (39.8650, 116.4100),
    "F_Park": (39.9650, 116.4300),
    "G_Station": (39.9000, 116.3250),
    "H_Market": (39.9250, 116.4700),
}

def make_trajectories(n_people=40, days=45, points_per_day=18):
    rows = []
    start = pd.Timestamp("2025-01-01 06:00:00")
    names = list(areas)
    for p in range(1, n_people + 1):
        home = names[p % len(names)]
        preferred = [home, names[(p+1) % len(names)], names[(p+2) % len(names)]]
        for d in range(days):
            base = start + pd.Timedelta(days=d)
            for k in range(points_per_day):
                hour = 6 + int(k * 16 / max(points_per_day-1, 1))
                minute = int(rng.integers(0, 60))
                ts = base.normalize() + pd.Timedelta(hours=hour, minutes=minute)
                area = preferred[int(rng.choice(len(preferred), p=[0.55,0.30,0.15]))]
                lat, lon = areas[area]
                lat += rng.normal(0, 0.0035)
                lon += rng.normal(0, 0.0045)
                rows.append([f"P{p:03d}", lat, lon, ts])
    df = pd.DataFrame(rows, columns=["user_id","latitude","longitude","timestamp"])
    df = df.sort_values(["user_id","timestamp"]).reset_index(drop=True)
    df["prev_lat"] = df.groupby("user_id")["latitude"].shift()
    df["prev_lon"] = df.groupby("user_id")["longitude"].shift()
    df["distance_km"] = (
        np.sqrt(((df.latitude-df.prev_lat)*111)**2 +
                ((df.longitude-df.prev_lon)*85)**2)
    ).fillna(0)
    dt_h = df.groupby("user_id")["timestamp"].diff().dt.total_seconds().div(3600)
    df["speed_kmh"] = (df["distance_km"] / dt_h.replace(0, np.nan)).replace([np.inf,-np.inf], np.nan).fillna(0).clip(0,150)
    df["hour"] = df.timestamp.dt.hour
    df["weekday"] = df.timestamp.dt.dayofweek
    df["is_weekend"] = (df.weekday >= 5).astype(int)
    return df.drop(columns=["prev_lat","prev_lon"])

def make_cases(traj):
    rng2 = np.random.default_rng(7)
    case_rows = []
    for i in range(12):
        person = f"P{int(rng2.integers(1,41)):03d}"
        p = traj[traj.user_id == person].sample(1, random_state=i).iloc[0]
        case_rows.append({
            "Case_ID": f"MP-2026-{i+1:03d}",
            "Person_ID": person,
            "Age_Group": rng2.choice(["18-25","26-35","36-50","51+"]),
            "Gender": rng2.choice(["Female","Male","Not specified"]),
            "Last_Latitude": p.latitude,
            "Last_Longitude": p.longitude,
            "Last_Seen_Time": p.timestamp,
            "Day": p.timestamp.day_name(),
            "Weather": rng2.choice(["Clear","Cloudy","Rain"]),
            "Usual_Area": rng2.choice(list(areas)),
            "Average_Distance": round(float(traj[traj.user_id==person].distance_km.mean()),2),
            "Average_Speed": round(float(traj[traj.user_id==person].speed_kmh.mean()),2),
            "Previous_Area": rng2.choice(list(areas)),
            "Time_Since_Last_Seen": int(rng2.integers(1,7)),
            "Target_Area": rng2.choice(list(areas)),
        })
    return pd.DataFrame(case_rows)

if __name__ == "__main__":
    out = ROOT/"data"
    out.joinpath("raw").mkdir(parents=True, exist_ok=True)
    out.joinpath("synthetic").mkdir(parents=True, exist_ok=True)
    traj = make_trajectories()
    cases = make_cases(traj)
    traj.to_csv(out/"raw"/"synthetic_gps_trajectories.csv", index=False)
    cases.to_csv(out/"synthetic"/"synthetic_cases.csv", index=False)
    print("Generated", len(traj), "trajectory records and", len(cases), "cases.")
