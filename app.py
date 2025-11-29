import streamlit as st
import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson, Fullscreen, BeautifyIcon  # ★追加
from datetime import datetime, date
from streamlit.components.v1 import html

# ===========================
# ユーティリティ
# ===========================

def clean_str(s):
    """前後の空白と全角スペースを取る"""
    s = str(s).replace("\u3000", "").strip()
    return s

def detect_condition(row):
    """熱中症・インフル・雪・コロナ疑い・その他 を判定"""
    def is_on(x):
        if pd.isna(x):
            return False
        s = str(x)
        return (s != "") and (s != "非該当") and (s != "0")

    if "incident_condition_heatstroke" in row and is_on(row["incident_condition_heatstroke"]):
        return "熱中症"
    if "incident_condition_flu" in row and is_on(row["incident_condition_flu"]):
        return "インフル"
    if "incident_condition_snow" in row and is_on(row["incident_condition_snow"]):
        return "雪"
    if "incident_condition_covid19_suspect" in row and is_on(row["incident_condition_covid19_suspect"]):
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
    return False  # その他も一旦不可として扱う


def read_any(file_obj):
    """アップロードされたファイルを csv / xlsx 判定して読む"""
    name = file_obj.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file_obj)
    else:
        return pd.read_csv(file_obj)


def build_lines(emg, addr, scene):
    """Colab でやっていた前処理をまとめて実行して、lines を返す"""

    # ------- 基本クリーニング -------
    emg["case_id"] = emg["case_id"].astype(str).str.strip()
    scene["case_id"] = scene["case_id"].astype(str).str.strip()

    emg["related_hospital"] = emg["related_hospital"].astype(str).apply(clean_str)
    emg["hospital_name"] = emg["hospital_name"].astype(str).apply(clean_str)
    addr["hospital_name"] = addr["hospital_name"].astype(str).apply(clean_str)

    emg["inquiry_end_time"] = pd.to_datetime(emg["inquiry_end_time"])
    emg["call_time"] = pd.to_datetime(emg["call_time"])

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

    # ------- 現場座標 -------
    scene2 = scene.rename(columns={"fX": "scene_lon", "fY": "scene_lat"})
    scene2 = (
        scene2[["case_id", "scene_lat", "scene_lon"]]
        .dropna(subset=["scene_lat", "scene_lon"])
        .drop_duplicates("case_id")
    )

    # ------- 病院座標 -------
    addr2 = addr.rename(columns={"fX": "hosp_lon", "fY": "hosp_lat"})
    addr2 = addr2[["hospital_name", "hosp_lat", "hosp_lon"]]

    # ------- 問い合わせのある行 -------
    emg_q = emg[emg["related_hospital"].str.len() > 0].copy()

    # 問い合わせ先病院座標
    rel_addr = addr2.rename(columns={
        "hospital_name": "related_hospital",
        "hosp_lat": "rel_lat",
        "hosp_lon": "rel_lon",
    })
    emg_rel = emg_q.merge(rel_addr, on="related_hospital", how="left")

    # 最終搬送病院座標
    final_addr = addr2.rename(columns={
        "hospital_name": "hospital_name",
        "hosp_lat": "final_lat",
        "hosp_lon": "final_lon",
    })
    emg_both = emg_rel.merge(final_addr, on="hospital_name", how="left")

    # 現場座標を付与
    lines = emg_both.merge(scene2, on="case_id", how="left")

    # 受入可否
    lines["is_available"] = lines["obstruction_info"].apply(classify_available)

    # 最終的に使う列だけ
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


def make_hospital_timeline_map(df_hosp_day, step_minutes=10):
    """その日の df_hosp を使って病院タイムラインマップを作成"""

    center_lat = df_hosp_day["lat"].mean()
    center_lon = df_hosp_day["lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    Fullscreen().add_to(m)

    features = []
    for _, row in df_hosp_day.iterrows():
        if row["is_available"] is None:
            continue
        color = "blue" if row["is_available"] else "red"
        popup_html = (
            f"{row['related_hospital']}<br>"
            f"{row['inquiry_end_time']}<br>"
            f"{'受入可(青)' if row['is_available'] else '受入不可(赤)'}<br>"
            f"理由: {row['obstruction_info']}"
        )
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["lon"], row["lat"]],
            },
            "properties": {
                "time": row["inquiry_end_time"].isoformat(),
                "popup": popup_html,
                "style": {
                    "color": color,
                    "fillColor": color,
                    "fillOpacity": 0.8,
                    "radius": 7,
                },
                "icon": "circle",
            },
        })

    period_str = f"PT{int(step_minutes)}M"

    tg = TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period=period_str,
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=10,
        loop_button=True,
        date_options="YYYY-MM-DD HH:mm",
        time_slider_drag_update=True,
    )
    tg.add_to(m)
    return m


def make_connection_map(day, highlight_top10=True):
    """
    現場↔病院 接続マップを作成
    - 現場：オレンジ丸
    - 青線：収容可
    - 赤線：収容不可
    - 緑線：収容不可→最終搬送
    - 病院ピン：Marker（青 / 濃い青 / 赤）
    """

    day = day.dropna(subset=["scene_lat", "scene_lon", "rel_lat", "rel_lon"])
    center_lat = pd.concat([day["scene_lat"], day["rel_lat"]]).mean()
    center_lon = pd.concat([day["scene_lon"], day["rel_lon"]]).mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    Fullscreen().add_to(m)

    # 現場ピン（オレンジ）
    fg_scenes = folium.FeatureGroup(name="現場（オレンジ）", show=True)
    scenes_unique = day[["case_id", "scene_lat", "scene_lon"]].drop_duplicates("case_id")
    for _, r in scenes_unique.iterrows():
        folium.CircleMarker(
            location=[r["scene_lat"], r["scene_lon"]],
            radius=4,
            color="orange",
            fill=True,
            fill_opacity=0.8,
            popup=f"case_id: {r['case_id']}",
        ).add_to(fg_scenes)
    fg_scenes.add_to(m)

    # 受入可（青）
    fg_ok = folium.FeatureGroup(name="受入可の問い合わせ", show=True)
    day_ok = day[day["is_available"] == True]
    for _, r in day_ok.iterrows():
        folium.PolyLine(
            [[r["scene_lat"], r["scene_lon"]], [r["rel_lat"], r["rel_lon"]]],
            color="blue",
            weight=2,
            opacity=0.7,
            tooltip=f"[収容可] {r['case_id']} → {r['related_hospital']}",
        ).add_to(fg_ok)
    fg_ok.add_to(m)

    # 受入不可（赤）
    fg_ng = folium.FeatureGroup(name="受入不可の問い合わせ", show=True)
    day_ng = day[day["is_available"] == False]
    for _, r in day_ng.iterrows():
        folium.PolyLine(
            [[r["scene_lat"], r["scene_lon"]], [r["rel_lat"], r["rel_lon"]]],
            color="red",
            weight=2,
            opacity=0.7,
            tooltip=f"[受入不可] {r['case_id']} → {r['related_hospital']} ({r['obstruction_info']})",
        ).add_to(fg_ng)
    fg_ng.add_to(m)

    # 受入不可 → 最終搬送（緑）
    fg_final = folium.FeatureGroup(name="受入不可→最終搬送", show=False)
    day_ng_final = day_ng.dropna(subset=["final_lat", "final_lon"])
    for _, r in day_ng_final.iterrows():
        folium.PolyLine(
            [[r["scene_lat"], r["scene_lon"]], [r["final_lat"], r["final_lon"]]],
            color="green",
            weight=2,
            opacity=0.7,
            tooltip=f"[最終搬送] {r['case_id']} → {r['hospital_name']}",
        ).add_to(fg_final)
    fg_final.add_to(m)

    # 病院ピン（Marker）
    fg_hosp = folium.FeatureGroup(name="問い合わせ病院ピン", show=True)
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

    thr = None
    if highlight_top10 and len(hosp_stats) > 0:
        thr = hosp_stats["n_total"].quantile(0.9)

    for _, r in hosp_stats.iterrows():
        base_color = "blue" if r["n_ok"] > 0 else "red"

        if (
            highlight_top10
            and thr is not None
            and r["n_total"] >= thr
            and base_color == "blue"
        ):
            marker_color = "darkblue"
        else:
            marker_color = base_color

        popup_html = (
            f"{r['related_hospital']}<br>"
            f"案件数: {int(r['n_total'])}<br>"
            f"収容可: {int(r['n_ok'])} 件<br>"
            f"受入不可: {int(r['n_ng'])} 件<br>"
            f"※ 上位10%件数なら濃い青"
        )

        folium.Marker(
            location=[r["lat"], r["lon"]],
            icon=folium.Icon(color=marker_color, icon="hospital-o", prefix="fa"),
            popup=popup_html,
        ).add_to(fg_hosp)

    fg_hosp.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def folium_to_streamlit(m, height=650):
    """Folium マップを Streamlit に埋め込む"""
    m_html = m._repr_html_()
    html(m_html, height=height)


# ===========================
# Streamlit アプリ本体
# ===========================

st.set_page_config(page_title="救急×病院 可視化ツール", layout="wide")

st.title("🚑 救急・病院 受入状況 可視化 Web アプリ")

st.markdown(
    """
3つのファイルをアップロードすると、  
- **現場↔病院 接続マップ**  
- **病院タイムライン（10分刻み）**  

をブラウザ上で確認できます。
"""
)

st.sidebar.header("1️⃣ データアップロード")

emg_file = st.sidebar.file_uploader("emergency_data ファイル", type=["csv", "xlsx"])
addr_file = st.sidebar.file_uploader("flu_with_address ファイル", type=["csv", "xlsx"])
scene_file = st.sidebar.file_uploader("Book1_for_csis ファイル", type=["csv", "xlsx"])

if not (emg_file and addr_file and scene_file):
    st.info("左のサイドバーから 3 ファイルすべてアップロードしてください。")
    st.stop()

# データ読み込み & 前処理
with st.spinner("データを読み込み・前処理中..."):
    emg = read_any(emg_file)
    addr = read_any(addr_file)
    scene = read_any(scene_file)
    lines = build_lines(emg, addr, scene)

# ---- 日付列の準備 ----
lines["date"] = pd.to_datetime(lines["inquiry_end_time"], errors="coerce").dt.date
date_series = lines["date"].dropna()

if date_series.empty:
    st.error("有効な日付データがありません。アップロードしたファイルを確認してください。")
    st.stop()

min_date = date_series.min()
max_date = date_series.max()

st.sidebar.header("2️⃣ フィルタ条件")

# ★ 期間で選択（開始日〜終了日）
date_range = st.sidebar.date_input(
    "期間（開始日〜終了日）",
    (min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# date_input は 1日だけ選ぶと date 型になるので、タプル/単体両方に対応
if isinstance(date_range, (list, tuple)):
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, date_range

# 選択期間で絞る
mask = (lines["date"] >= start_date) & (lines["date"] <= end_date)
day_base = lines[mask].copy()

# ---- 病院候補（問い合わせ件数の多い順）----
hosp_counts = (
    day_base
    .dropna(subset=["related_hospital"])
    .groupby("related_hospital")["case_id"]
    .nunique()  # 案件数で数える
    .reset_index(name="n_cases")
)

# 件数の多い順にソート
hosp_counts = hosp_counts.sort_values("n_cases", ascending=False)

# ラベル「病院名（◯件）」を作る
hosp_labels = ["（全て）"]
label_to_name = {"（全て）": None}

for _, row in hosp_counts.iterrows():
    label = f"{row['related_hospital']}（{int(row['n_cases'])}件）"
    hosp_labels.append(label)
    label_to_name[label] = row["related_hospital"]

hosp_label_sel = st.sidebar.selectbox("病院（問い合わせ先）", hosp_labels)
# 時間帯
time_options = ["（全て）", "0-6", "6-12", "12-18", "18-24"]
time_sel = st.sidebar.selectbox("時間帯", time_options)

# 症状
cond_options = ["（全て）"] + sorted(day_base["main_condition"].dropna().unique().tolist())
cond_sel = st.sidebar.selectbox("症状", cond_options)

# マップ種別
map_type = st.sidebar.radio("表示するマップ", ["現場↔病院 接続マップ", "病院タイムライン"])

# 実際にフィルタ値を設定
hosp_val = label_to_name.get(hosp_label_sel)  # ラベル→病院名に戻す
time_val = None if time_sel == "（全て）" else time_sel
cond_val = None if cond_sel == "（全て）" else cond_sel

# 共通フィルタ適用
day = day_base.copy()
if hosp_val is not None:
    day = day[day["related_hospital"] == clean_str(hosp_val)]
if time_val is not None:
    day = day[day["time_band"] == time_val]
if cond_val is not None:
    day = day[day["main_condition"] == cond_val]

st.write(f"### 期間: {start_date} 〜 {end_date} / レコード数: {len(day)}")

if day.empty:
    st.warning("この条件に該当するデータがありません。フィルタを緩めてください。")
    st.stop()

# ===========================
# マップの表示
# ===========================
if map_type == "現場↔病院 接続マップ":
    st.subheader("🗺 現場↔病院 接続マップ")
    m = make_connection_map(day)
    folium_to_streamlit(m)

else:  # 病院タイムライン
    st.subheader("⏱ 病院タイムライン（10分刻み）")

    # タイムライン用 df_hosp_day を構築
    df_hosp_day = (
        day
        .dropna(subset=["rel_lat", "rel_lon"])
        .rename(columns={"rel_lat": "lat", "rel_lon": "lon"})
        [["related_hospital", "obstruction_info", "inquiry_end_time", "lat", "lon", "is_available"]]
        .copy()
        .sort_values("inquiry_end_time")
    )

    if df_hosp_day.empty:
        st.warning("この条件に該当する病院タイムラインデータがありません。")
    else:
        step = st.sidebar.slider("タイムラインの刻み（分）", min_value=5, max_value=60, value=10, step=5)
        m = make_hospital_timeline_map(df_hosp_day, step_minutes=step)
        folium_to_streamlit(m)
