import streamlit as st
import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson, Fullscreen, HeatMap
from streamlit.components.v1 import html

# ===========================
# 基本ユーティリティ
# ===========================

def clean_str(s):
    return str(s).replace("\u3000", "").strip()

def detect_condition(row):
    """熱中症・インフル・雪・コロナ疑い・その他 を判定"""
    def is_on(x):
        if pd.isna(x):
            return False
        s = str(x)
        return (s != "") and (s != "非該当") and (s != "0")

    if is_on(row.get("incident_condition_heatstroke")):
        return "熱中症"
    if is_on(row.get("incident_condition_flu")):
        return "インフル"
    if is_on(row.get("incident_condition_snow")):
        return "雪"
    if is_on(row.get("incident_condition_covid19_suspect")):
        return "コロナ疑い"
    return "その他/なし"

def classify_available(info):
    """obstruction_info から受け入れ可否を分類"""
    if pd.isna(info):
        return None
    s = str(info).strip()
    if s == "収容可":
        return True
    NG_WORDS = ["処置困難", "応答なし", "患者対応中", "満床"]
    if any(w in s for w in NG_WORDS):
        return False
    return False  # その他は一旦「不可」と扱う

def read_any(file_obj):
    """アップロードされたファイルを csv / xlsx 判定して読む"""
    name = file_obj.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file_obj)
    return pd.read_csv(file_obj)

# ===========================
# データ前処理（キャッシュ）
# ===========================

def build_lines(emg, addr, scene):
    # ID系クリーニング
    emg["case_id"] = emg["case_id"].astype(str).str.strip()
    scene["case_id"] = scene["case_id"].astype(str).str.strip()

    # 病院名まわり
    emg["related_hospital"] = emg["related_hospital"].astype(str).apply(clean_str)
    emg["hospital_name"] = emg["hospital_name"].astype(str).apply(clean_str)
    addr["hospital_name"] = addr["hospital_name"].astype(str).apply(clean_str)

    # 時刻
    emg["inquiry_end_time"] = pd.to_datetime(emg["inquiry_end_time"], errors="coerce")
    emg["call_time"] = pd.to_datetime(emg["call_time"], errors="coerce")

    # 時間帯
    emg["call_hour"] = emg["call_time"].dt.hour
    emg["time_band"] = pd.cut(
        emg["call_hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["0-6", "6-12", "12-18", "18-24"],
        right=False,
        include_lowest=True,
    )

    # 症状ラベル
    emg["main_condition"] = emg.apply(detect_condition, axis=1)

    # 現場座標
    scene2 = scene.rename(columns={"fX": "scene_lon", "fY": "scene_lat"})
    scene2 = (
        scene2[["case_id", "scene_lat", "scene_lon"]]
        .dropna(subset=["scene_lat", "scene_lon"])
        .drop_duplicates("case_id")
    )

    # 病院座標
    addr2 = addr.rename(columns={"fX": "hosp_lon", "fY": "hosp_lat"})
    addr2 = addr2[["hospital_name", "hosp_lat", "hosp_lon"]]

    # 問い合わせのある行
    emg_q = emg[emg["related_hospital"].str.len() > 0].copy()

    # 問い合わせ病院の座標
    rel_addr = addr2.rename(columns={
        "hospital_name": "related_hospital",
        "hosp_lat": "rel_lat",
        "hosp_lon": "rel_lon",
    })
    emg_rel = emg_q.merge(rel_addr, on="related_hospital", how="left")

    # 最終搬送病院の座標
    final_addr = addr2.rename(columns={
        "hospital_name": "hospital_name",
        "hosp_lat": "final_lat",
        "hosp_lon": "final_lon",
    })
    emg_both = emg_rel.merge(final_addr, on="hospital_name", how="left")

    # 現場座標付与
    lines = emg_both.merge(scene2, on="case_id", how="left")

    # 受入可否
    lines["is_available"] = lines["obstruction_info"].apply(classify_available)

    # 最終的に使う列のみ
    lines = lines[[
        "case_id",
        "related_hospital",
        "obstruction_info",
        "inquiry_end_time",
        "scene_lat", "scene_lon",
        "rel_lat", "rel_lon",
        "hospital_name",
        "final_lat", "final_lon",
        "is_available",
        "time_band",
        "main_condition",
    ]].copy()

    return lines

@st.cache_data
def build_lines_cached(emg, addr, scene):
    return build_lines(emg, addr, scene)

# ===========================
# マップ：現場↔病院 接続
# ===========================

def make_connection_map(day, highlight_top10=True):
    """
    現場↔病院 接続マップ
    - 現場：受入不可率が高い現場＝赤、それ以外オレンジ
    - 線：青=受入可, 赤=受入不可, 緑=最終搬送
    - 病院：問い合わせ件数別の色（青/濃い青/赤）
    """

    MAX_ROWS = 3000
    if len(day) > MAX_ROWS:
        day = day.sort_values("inquiry_end_time").iloc[-MAX_ROWS:].copy()

    day = day.dropna(subset=["scene_lat", "scene_lon", "rel_lat", "rel_lon"])
    if day.empty:
        m = folium.Map(location=[38.26, 140.87], zoom_start=11)
        return m

    center_lat = pd.concat([day["scene_lat"], day["rel_lat"]]).mean()
    center_lon = pd.concat([day["scene_lon"], day["rel_lon"]]).mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    Fullscreen().add_to(m)

    # ---- 現場（赤 / オレンジ）----
    scene_stats = (
        day.groupby(["case_id", "scene_lat", "scene_lon"])
        .agg(
            n_total=("case_id", "size"),
            n_ng=("is_available", lambda s: (s == False).sum()),
        )
        .reset_index()
    )
    scene_stats["reject_rate"] = scene_stats["n_ng"] / scene_stats["n_total"]
    THR = 0.5

    fg_s = folium.FeatureGroup(name="現場（赤=拒否多）", show=True)
    for _, r in scene_stats.iterrows():
        color = "red" if r["reject_rate"] >= THR else "orange"
        popup = (
            f"case_id: {r['case_id']}<br>"
            f"問い合わせ: {int(r['n_total'])}件<br>"
            f"受入不可: {int(r['n_ng'])}件<br>"
            f"受入不可率: {r['reject_rate']:.2f}"
        )
        folium.CircleMarker(
            [r["scene_lat"], r["scene_lon"]],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=popup,
        ).add_to(fg_s)
    fg_s.add_to(m)

    # ---- 線（可 / 不可 / 最終搬送）----
    fg_ok = folium.FeatureGroup(name="受入可（青）", show=True)
    fg_ng = folium.FeatureGroup(name="受入不可（赤）", show=True)
    fg_fin = folium.FeatureGroup(name="不可→最終搬送（緑）", show=False)

    day_ok = day[day["is_available"] == True]
    day_ng = day[day["is_available"] == False]
    day_ng_f = day_ng.dropna(subset=["final_lat", "final_lon"])

    for _, r in day_ok.iterrows():
        folium.PolyLine(
            [[r["scene_lat"], r["scene_lon"]], [r["rel_lat"], r["rel_lon"]]],
            color="blue", weight=2, opacity=0.7,
        ).add_to(fg_ok)

    for _, r in day_ng.iterrows():
        folium.PolyLine(
            [[r["scene_lat"], r["scene_lon"]], [r["rel_lat"], r["rel_lon"]]],
            color="red", weight=2, opacity=0.7,
        ).add_to(fg_ng)

    for _, r in day_ng_f.iterrows():
        folium.PolyLine(
            [[r["scene_lat"], r["scene_lon"]], [r["final_lat"], r["final_lon"]]],
            color="green", weight=2, opacity=0.7,
        ).add_to(fg_fin)

    fg_ok.add_to(m)
    fg_ng.add_to(m)
    fg_fin.add_to(m)

    # ---- 病院ピン ----
    fg_h = folium.FeatureGroup(name="病院ピン", show=True)
    hosp_stats = (
        day.dropna(subset=["rel_lat", "rel_lon"])
        .groupby("related_hospital")
        .agg(
            lat=("rel_lat", "first"),
            lon=("rel_lon", "first"),
            n_total=("case_id", "nunique"),
            n_ok=("is_available", lambda s: (s == True).sum()),
            n_ng=("is_available", lambda s: (s == False).sum()),
        )
        .reset_index()
    )

    thr = hosp_stats["n_total"].quantile(0.9) if len(hosp_stats) else None

    for _, r in hosp_stats.iterrows():
        base = "blue" if r["n_ok"] > 0 else "red"
        marker_color = "darkblue" if (thr and r["n_total"] >= thr and base == "blue") else base

        popup = (
            f"{r['related_hospital']}<br>"
            f"案件数: {int(r['n_total'])}件<br>"
            f"収容可: {int(r['n_ok'])}件<br>"
            f"受入不可: {int(r['n_ng'])}件"
        )

        folium.Marker(
            [r["lat"], r["lon"]],
            icon=folium.Icon(color=marker_color, icon="hospital-o", prefix="fa"),
            popup=popup,
        ).add_to(fg_h)

    fg_h.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m

# ===========================
# マップ：病院タイムライン
# ===========================

def make_hospital_timeline_map(df, step_minutes=10):
    """
    タイムライン表示用マップ
    - 灰色の病院ピン：期間中に一度でも問い合わせがあった病院の位置
    - 青/赤のポイント：時間経過とともに出現する問い合わせ
    """

    MAX_POINTS = 1500
    if len(df) > MAX_POINTS:
        df = df.sort_values("inquiry_end_time").iloc[-MAX_POINTS:]

    if df.empty:
        m = folium.Map(location=[38.26, 140.87], zoom_start=11)
        return m

    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    Fullscreen().add_to(m)

    # ---- 灰色の病院ピン（常に表示）----
    fg_h = folium.FeatureGroup(name="病院位置", show=True)
    hosp_pos = (
        df.dropna(subset=["lat", "lon"])
        .groupby("related_hospital")
        .agg(lat=("lat", "first"), lon=("lon", "first"))
        .reset_index()
    )
    for _, r in hosp_pos.iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]],
            radius=4,
            color="gray",
            fill=True,
            fill_opacity=0.7,
        ).add_to(fg_h)
    fg_h.add_to(m)

    # ---- 時系列ポイント（青=可 / 赤=不可）----
    feats = []
    for _, r in df.iterrows():
        if pd.isna(r["inquiry_end_time"]):
            continue
        color = "blue" if r["is_available"] else "red"
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {
                "time": r["inquiry_end_time"].isoformat(),
                "style": {"color": color, "fillColor": color, "radius": 6},
            },
        })

    tg = TimestampedGeoJson(
        {"type": "FeatureCollection", "features": feats},
        period=f"PT{int(step_minutes)}M",
        auto_play=False,
        loop=False,
        date_options="YYYY-MM-DD HH:mm",
    )
    tg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m

# ===========================
# マップ：救急需要×受入困難 ヒートマップ
# ===========================

def make_demand_difficulty_heatmap(day):
    """
    救急需要 × 受入困難 ヒートマップ
    - 単位：1現場（case_id）
    - 重み：その現場での「受入不可件数」 n_ng
      → 需要が多くて、かつ受入不可が多い地点ほど強く光る
    """

    # 現場ごとに集計
    scene_stats = (
        day.dropna(subset=["scene_lat", "scene_lon"])
        .groupby(["case_id", "scene_lat", "scene_lon"])
        .agg(
            n_total=("case_id", "size"),
            n_ng=("is_available", lambda s: (s == False).sum()),
        )
        .reset_index()
    )

    # 重み = n_ng（受入不可件数）
    scene_stats["weight"] = scene_stats["n_ng"].astype(float)

    # 0 の点はヒートマップにほぼ影響しないので、そのままでもOKだが
    # 全部 0 の場合はマップだけ返す
    if scene_stats["weight"].sum() == 0:
        m = folium.Map(location=[38.26, 140.87], zoom_start=11)
        folium.Marker(
            [38.26, 140.87],
            popup="この条件では受入困難な地点がありません。",
        ).add_to(m)
        return m

    center_lat = scene_stats["scene_lat"].mean()
    center_lon = scene_stats["scene_lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    Fullscreen().add_to(m)

    heat_data = [
        [row["scene_lat"], row["scene_lon"], row["weight"]]
        for _, row in scene_stats.iterrows()
    ]

    HeatMap(
        heat_data,
        radius=18,
        blur=15,
        max_zoom=13,
    ).add_to(m)

    return m

# ===========================
# Folium → Streamlit 埋め込み
# ===========================

def folium_to_streamlit(m, height=650):
    html(m._repr_html_(), height=height)

# ===========================
# Streamlit 本体
# ===========================

st.set_page_config(page_title="救急×病院 可視化", layout="wide")
st.title("🚑 救急×病院 受入状況 可視化アプリ")

st.sidebar.header("1️⃣ データアップロード")
emg_file = st.sidebar.file_uploader("emergency_data", ["csv", "xlsx"])
addr_file = st.sidebar.file_uploader("flu_with_address", ["csv", "xlsx"])
scene_file = st.sidebar.file_uploader("Book1_for_csis", ["csv", "xlsx"])

if not (emg_file and addr_file and scene_file):
    st.info("3つのファイルをアップロードしてください。")
    st.stop()

with st.spinner("前処理中..."):
    emg = read_any(emg_file)
    addr = read_any(addr_file)
    scene = read_any(scene_file)
    lines = build_lines_cached(emg, addr, scene)

# ---- 日付処理 ----
lines["date"] = pd.to_datetime(lines["inquiry_end_time"], errors="coerce").dt.date
date_series = lines["date"].dropna()

if date_series.empty:
    st.error("日付データがありません。")
    st.stop()

min_date, max_date = date_series.min(), date_series.max()

st.sidebar.header("2️⃣ フィルタ条件")

date_range = st.sidebar.date_input(
    "期間（開始〜終了）",
    (min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, (list, tuple)):
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, date_range

if start_date > end_date:
    start_date, end_date = end_date, start_date

mask = (lines["date"] >= start_date) & (lines["date"] <= end_date)
day_base = lines[mask].copy()

if day_base.empty:
    st.warning("この期間にはデータがありません。")
    st.stop()

# ---- 病院選択（件数順＋件数表示）----
hosp_counts = (
    day_base.dropna(subset=["related_hospital"])
    .groupby("related_hospital")["case_id"]
    .nunique()
    .reset_index(name="n_cases")
    .sort_values("n_cases", ascending=False)
)

labels = ["（全て）"]
lab_to_name = {"（全て）": None}
for _, r in hosp_counts.iterrows():
    lab = f"{r['related_hospital']}（{int(r['n_cases'])}件）"
    labels.append(lab)
    lab_to_name[lab] = r["related_hospital"]

hosp_label = st.sidebar.selectbox("病院（問い合わせ件数順）", labels)
hosp_val = lab_to_name[hosp_label]

# ---- 時間帯 / 症状 ----
time_opt = ["（全て）", "0-6", "6-12", "12-18", "18-24"]
time_sel = st.sidebar.selectbox("時間帯", time_opt)
time_val = None if time_sel == "（全て）" else time_sel

cond_opt = ["（全て）"] + sorted(day_base["main_condition"].dropna().unique())
cond_sel = st.sidebar.selectbox("症状", cond_opt)
cond_val = None if cond_sel == "（全て）" else cond_sel

# ---- マップ種別 ----
map_type = st.sidebar.radio(
    "表示マップ",
    ["現場↔病院 接続マップ", "病院タイムライン", "救急需要×受入困難ヒートマップ"],
)

# ---- フィルタ適用 ----
day = day_base.copy()
if hosp_val:
    day = day[day["related_hospital"] == hosp_val]
if time_val:
    day = day[day["time_band"] == time_val]
if cond_val:
    day = day[day["main_condition"] == cond_val]

if day.empty:
    st.warning("この条件のデータはありません。フィルタを調整してください。")
    st.stop()

st.write(f"### 期間: {start_date} 〜 {end_date} / レコード数: {len(day)}")

# ===========================
# マップ表示
# ===========================
if map_type == "現場↔病院 接続マップ":
    st.subheader("🗺 現場↔病院 接続マップ")
    m = make_connection_map(day)
    folium_to_streamlit(m)

elif map_type == "病院タイムライン":
    st.subheader("⏱ 病院タイムライン（10分刻み）")

    # タイムラインは重いので3日以内に制限
    if (end_date - start_date).days > 3:
        st.warning("タイムライン表示は3日以内にしてください。")
        st.stop()

    df = (
        day.dropna(subset=["rel_lat", "rel_lon"])
        .rename(columns={"rel_lat": "lat", "rel_lon": "lon"})
        [["related_hospital", "obstruction_info", "inquiry_end_time", "lat", "lon", "is_available"]]
        .sort_values("inquiry_end_time")
    )

    if df.empty:
        st.warning("この条件に該当するタイムラインデータがありません。")
    else:
        step = st.sidebar.slider("タイムラインの刻み（分）", 5, 60, 10, 5)
        m = make_hospital_timeline_map(df, step_minutes=step)
        folium_to_streamlit(m)

else:  # 救急需要×受入困難ヒートマップ
    st.subheader("🔥 救急需要 × 受入困難 ヒートマップ")
    m = make_demand_difficulty_heatmap(day)
    folium_to_streamlit(m)
