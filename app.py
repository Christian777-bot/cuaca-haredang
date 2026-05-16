from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "weather_xgboost.joblib"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"
DATA_PATH = ROOT_DIR / "weather_classification_data.csv"

st.set_page_config(
    page_title="Prediksi Cuaca Pintar",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #eef7ff 0%, #f8fafc 45%, #ffffff 100%);
}
.block-container { padding-top: 1.5rem; max-width: 1240px; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #eaf4ff 100%);
    border-right: 1px solid #dbeafe;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #0f172a !important;
}
section[data-testid="stSidebar"] .stSlider {
    padding-bottom: 8px;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 14px !important;
}

.hero {
    border-radius: 30px;
    padding: 32px;
    color: white;
    background:
      linear-gradient(135deg, rgba(2,132,199,.96), rgba(79,70,229,.94)),
      url('https://images.unsplash.com/photo-1504608524841-42fe6f032b4b?q=80&w=1600&auto=format&fit=crop');
    background-size: cover;
    background-position: center;
    box-shadow: 0 22px 60px rgba(30, 64, 175, .20);
    margin-bottom: 22px;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,.18);
    border: 1px solid rgba(255,255,255,.28);
    padding: 8px 13px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 13px;
    letter-spacing: .04em;
}
.hero-title {
    font-size: clamp(38px, 6vw, 70px);
    font-weight: 950;
    line-height: .98;
    margin-top: 18px;
    letter-spacing: -.05em;
}
.hero-subtitle {
    max-width: 780px;
    font-size: 18px;
    line-height: 1.65;
    color: rgba(255,255,255,.92);
    margin-top: 14px;
}

.card {
    background: rgba(255,255,255,.92);
    border: 1px solid #e2e8f0;
    border-radius: 26px;
    padding: 24px;
    box-shadow: 0 16px 42px rgba(15,23,42,.08);
}
.small-title {
    font-size: 13px;
    font-weight: 900;
    letter-spacing: .08em;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.weather-wrap { display: flex; gap: 18px; align-items: center; margin: 12px 0; }
.weather-icon {
    width: 88px; height: 88px; border-radius: 26px;
    display: grid; place-items: center;
    font-size: 48px;
    background: linear-gradient(135deg, #dbeafe, #f0f9ff);
    border: 1px solid #bfdbfe;
}
.weather-name {
    font-size: clamp(40px, 5vw, 58px);
    font-weight: 950;
    color: #0f172a;
    letter-spacing: -.04em;
    line-height: 1;
}
.pill {
    display: inline-block;
    padding: 8px 13px;
    border-radius: 999px;
    font-weight: 850;
    margin-top: 9px;
}
.high { background: #dcfce7; color: #166534; }
.mid { background: #fef3c7; color: #92400e; }
.low { background: #fee2e2; color: #991b1b; }
.text { color: #334155; line-height: 1.7; font-size: 16px; }
.status {
    margin-top: 15px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 14px 16px;
    color: #0f172a;
    font-weight: 800;
}
.kpi-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
.kpi { background: #fff; border: 1px solid #e2e8f0; border-radius: 20px; padding: 16px; }
.kpi-label { color: #64748b; font-size: 13px; font-weight: 850; }
.kpi-value { color: #0f172a; font-size: 30px; font-weight: 950; margin-top: 4px; }
.kpi-note { color: #64748b; font-size: 12.5px; line-height: 1.45; margin-top: 4px; }
.section-title { font-size: 29px; font-weight: 950; color: #0f172a; margin: 26px 0 12px; letter-spacing: -.03em; }
.tip-card { background: white; border: 1px solid #e2e8f0; border-radius: 22px; padding: 20px; height: 100%; box-shadow: 0 12px 30px rgba(15,23,42,.06); }
.tip-icon { font-size: 28px; margin-bottom: 8px; }
.tip-title { color: #0f172a; font-size: 18px; font-weight: 950; margin-bottom: 6px; }
.tip-text { color: #475569; font-size: 15px; line-height: 1.6; }
.note { background: #fffbeb; color: #78350f; border: 1px solid #fde68a; border-radius: 20px; padding: 16px 18px; line-height: 1.6; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

WEATHER_GUIDE = {
    "Cloudy": {
        "id": "Berawan", "icon": "☁️", "risk": "Risiko rendah sampai sedang",
        "status": "Cukup aman untuk aktivitas luar ruangan",
        "summary": "Langit cenderung tertutup awan. Aktivitas luar ruangan masih memungkinkan, tetapi tetap siapkan antisipasi karena cuaca bisa berubah.",
        "tips": [
            ("☂️", "Bawa payung kecil", "Tidak wajib, tetapi berguna jika cuaca berubah menjadi gerimis."),
            ("🚶", "Aman untuk aktivitas ringan", "Cocok untuk ke kampus, jalan santai, atau aktivitas luar yang singkat."),
            ("👕", "Pakai pakaian nyaman", "Suhu bisa terasa lembap, jadi gunakan pakaian yang tidak terlalu tebal."),
        ],
    },
    "Rainy": {
        "id": "Hujan", "icon": "🌧️", "risk": "Risiko sedang sampai tinggi",
        "status": "Perlu persiapan sebelum keluar rumah",
        "summary": "Kondisi mengarah ke hujan. Sebaiknya siapkan perlengkapan dan kurangi aktivitas luar ruangan yang tidak terlalu penting.",
        "tips": [
            ("☔", "Bawa payung atau jas hujan", "Ini perlengkapan utama agar aktivitas tetap aman dan nyaman."),
            ("🛵", "Hati-hati di jalan", "Jalan bisa licin dan jarak pandang dapat menurun."),
            ("💻", "Lindungi barang elektronik", "Masukkan laptop atau HP ke tas yang aman dari air."),
        ],
    },
    "Sunny": {
        "id": "Cerah", "icon": "☀️", "risk": "Risiko panas dan UV perlu diperhatikan",
        "status": "Baik untuk aktivitas luar ruangan",
        "summary": "Cuaca cenderung cerah. Ini bagus untuk aktivitas luar, tetapi tetap perhatikan paparan matahari dan kebutuhan cairan tubuh.",
        "tips": [
            ("🧴", "Gunakan pelindung matahari", "Topi atau sunscreen membantu mengurangi paparan UV."),
            ("💧", "Bawa air minum", "Tubuh lebih mudah dehidrasi saat cuaca panas."),
            ("🌳", "Cari tempat teduh", "Hindari terlalu lama di bawah matahari langsung."),
        ],
    },
    "Snowy": {
        "id": "Bersalju", "icon": "❄️", "risk": "Risiko dingin dan jalan licin",
        "status": "Butuh perlindungan dari suhu dingin",
        "summary": "Kondisi mengarah ke salju. Gunakan pakaian hangat dan lebih berhati-hati saat bepergian.",
        "tips": [
            ("🧥", "Gunakan jaket tebal", "Tubuh perlu perlindungan ekstra dari suhu dingin."),
            ("🥾", "Pakai alas kaki aman", "Permukaan jalan bisa licin saat bersalju."),
            ("🏠", "Kurangi aktivitas luar", "Keluar rumah seperlunya saja bila kondisi tidak mendukung."),
        ],
    },
}

COLUMN_LABELS = {
    "Temperature": "Suhu (°C)",
    "Humidity": "Kelembapan (%)",
    "Wind Speed": "Kecepatan Angin",
    "Precipitation (%)": "Peluang Hujan (%)",
    "Atmospheric Pressure": "Tekanan Udara",
    "UV Index": "Indeks UV",
    "Visibility (km)": "Jarak Pandang (km)",
    "Cloud Cover": "Kondisi Awan",
    "Season": "Musim",
    "Location": "Lokasi",
}

CATEGORY_LABELS = {
    "overcast": "Mendung Tebal",
    "partly cloudy": "Sebagian Berawan",
    "clear": "Cerah",
    "cloudy": "Berawan",
    "Spring": "Semi",
    "Summer": "Panas",
    "Autumn": "Gugur",
    "Winter": "Dingin",
    "inland": "Dataran Dalam",
    "mountain": "Pegunungan",
    "coastal": "Pesisir",
}

FRIENDLY_LIMITS = {
    "Temperature": (-25.0, 45.0, 0.5),
    "Humidity": (0.0, 100.0, 1.0),
    "Wind Speed": (0.0, 50.0, 0.5),
    "Precipitation (%)": (0.0, 100.0, 1.0),
    "Atmospheric Pressure": (800.0, 1200.0, 1.0),
    "UV Index": (0.0, 14.0, 0.5),
    "Visibility (km)": (0.0, 20.0, 0.5),
}

WEATHER_COLORS = {
    "Berawan": "#64748b",
    "Hujan": "#2563eb",
    "Bersalju": "#38bdf8",
    "Cerah": "#f59e0b",
}

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics_file():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}

@st.cache_data
def load_dataset_preview():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


def display_name(value: str) -> str:
    return CATEGORY_LABELS.get(str(value), str(value).title())


def confidence_level(probability: float):
    if probability >= 0.85:
        return "Keyakinan tinggi", "high", "Prediksi utama sangat kuat."
    if probability >= 0.60:
        return "Keyakinan sedang", "mid", "Masih ada kemungkinan cuaca lain."
    return "Keyakinan rendah", "low", "Perlu membaca peluang kelas lain dengan hati-hati."


def friendly_slider(column: str, metadata: dict) -> float:
    raw = metadata["numeric"][column]
    min_value, max_value, step = FRIENDLY_LIMITS.get(
        column,
        (float(raw["min"]), float(raw["max"]), 1.0),
    )
    default = float(raw.get("mean", min_value))
    default = max(min_value, min(max_value, default))
    return st.slider(
        COLUMN_LABELS.get(column, column),
        min_value=float(min_value),
        max_value=float(max_value),
        value=float(round(default, 1)),
        step=float(step),
        help="Geser nilainya. Ini lebih aman daripada diketik manual, jadi tidak akan muncul error input.",
    )


def build_insights(user_input: dict):
    insights = []
    temp = float(user_input.get("Temperature", 0))
    humidity = float(user_input.get("Humidity", 0))
    wind = float(user_input.get("Wind Speed", 0))
    precipitation = float(user_input.get("Precipitation (%)", 0))
    uv = float(user_input.get("UV Index", 0))
    visibility = float(user_input.get("Visibility (km)", 0))
    cloud = str(user_input.get("Cloud Cover", ""))

    if precipitation >= 60:
        insights.append(("🌧️", "Peluang hujan tinggi", "Nilai peluang hujan cukup besar, jadi model wajar mempertimbangkan kondisi hujan."))
    elif precipitation >= 30:
        insights.append(("🌦️", "Peluang hujan sedang", "Ada sinyal hujan, tetapi belum terlalu dominan."))
    else:
        insights.append(("☀️", "Peluang hujan rendah", "Input peluang hujan rendah, sehingga cuaca kering lebih masuk akal."))

    if humidity >= 75:
        insights.append(("💧", "Kelembapan tinggi", "Udara terasa lembap dan bisa mendukung kondisi berawan atau hujan."))
    elif humidity <= 35:
        insights.append(("🏜️", "Kelembapan rendah", "Udara cenderung kering, sehingga kondisi cerah bisa lebih mungkin."))

    if uv >= 7:
        insights.append(("🧴", "Indeks UV tinggi", "Jika keluar rumah, gunakan pelindung dari sinar matahari."))
    elif uv <= 2:
        insights.append(("🌥️", "Indeks UV rendah", "Paparan matahari relatif rendah, sering terjadi saat cuaca berawan atau mendung."))

    if visibility <= 3:
        insights.append(("👀", "Jarak pandang rendah", "Perlu hati-hati saat berkendara karena pandangan bisa terbatas."))

    if wind >= 15:
        insights.append(("💨", "Angin cukup kencang", "Aktivitas luar ruangan perlu lebih hati-hati, terutama jika membawa payung."))

    if cloud in ["overcast", "cloudy", "partly cloudy"]:
        insights.append(("☁️", "Kondisi awan mendukung", f"Input kondisi awan adalah {display_name(cloud)}, sehingga prediksi berawan atau hujan menjadi masuk akal."))

    if temp >= 30:
        insights.append(("🔥", "Suhu cukup panas", "Perhatikan dehidrasi dan paparan panas saat beraktivitas di luar."))
    elif temp <= 5:
        insights.append(("❄️", "Suhu sangat dingin", "Perlu pakaian hangat dan perlindungan tambahan dari suhu rendah."))

    return insights[:6]


def make_probability_chart(probability_df: pd.DataFrame):
    fig = px.bar(
        probability_df.sort_values("Peluang (%)", ascending=True),
        x="Peluang (%)",
        y="Jenis Cuaca",
        orientation="h",
        text="Label Persen",
        color="Jenis Cuaca",
        color_discrete_map=WEATHER_COLORS,
        range_x=[0, 100],
    )
    fig.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
    fig.update_layout(
        height=380,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=80, t=20, b=20),
        xaxis_title="Peluang menurut model (%)",
        yaxis_title="",
        font=dict(size=14),
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=False),
    )
    return fig


def render_tip_cards(tips):
    cols = st.columns(3)
    for col, (icon, title, text) in zip(cols, tips):
        with col:
            st.markdown(
                f"""
                <div class="tip-card">
                    <div class="tip-icon">{icon}</div>
                    <div class="tip-title">{title}</div>
                    <div class="tip-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_insights(insights):
    for icon, title, text in insights:
        with st.container(border=True):
            col_icon, col_text = st.columns([0.11, 0.89])
            with col_icon:
                st.markdown(f"### {icon}")
            with col_text:
                st.markdown(f"**{title}**")
                st.caption(text)


def main():
    if not MODEL_PATH.exists():
        st.error("Model belum tersedia. Pastikan file `models/weather_xgboost.joblib` sudah ada di GitHub, bukan di luar folder.")
        st.stop()

    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]
    metadata = artifact["feature_metadata"]
    metrics = artifact.get("metrics", {}) or load_metrics_file()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">🌦️ Machine Learning Weather App</div>
            <div class="hero-title">Prediksi Cuaca Pintar</div>
            <div class="hero-subtitle">
                Masukkan kondisi lingkungan, lalu website akan memprediksi tipe cuaca dan memberi rekomendasi praktis
                dengan bahasa yang mudah dipahami orang awam.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("📌 Input Kondisi Cuaca")
        st.caption("Gunakan slider supaya input tidak error dan mudah dipakai.")
        st.divider()

        user_input = {}
        st.subheader("🌡️ Data Numerik")
        for column in metadata["numeric"]:
            user_input[column] = friendly_slider(column, metadata)

        st.subheader("🏷️ Data Kategori")
        for column, options in metadata["categorical"].items():
            user_input[column] = st.selectbox(
                COLUMN_LABELS.get(column, column),
                options=options,
                format_func=display_name,
            )

        st.divider()
        st.info("Interaktif: setiap slider atau pilihan diubah, hasil prediksi langsung ikut berubah.")

    input_df = pd.DataFrame([user_input], columns=metadata["feature_columns"])
    encoded_prediction = int(pipeline.predict(input_df)[0])
    prediction = label_encoder.inverse_transform([encoded_prediction])[0]
    probabilities = pipeline.predict_proba(input_df)[0]

    probability_df = pd.DataFrame({"Weather Type": label_encoder.classes_, "Probability": probabilities})
    probability_df["Jenis Cuaca"] = probability_df["Weather Type"].map(lambda x: WEATHER_GUIDE.get(x, {}).get("id", x))
    probability_df["Peluang (%)"] = (probability_df["Probability"] * 100).round(2)
    probability_df["Label Persen"] = probability_df["Peluang (%)"].map(lambda x: f"{x:.1f}%")
    probability_df = probability_df.sort_values("Probability", ascending=False).reset_index(drop=True)

    top_probability = float(probability_df.iloc[0]["Probability"])
    second_probability = float(probability_df.iloc[1]["Probability"]) if len(probability_df) > 1 else 0.0
    guide = WEATHER_GUIDE.get(prediction, WEATHER_GUIDE["Cloudy"])
    confidence_text, confidence_class, confidence_note = confidence_level(top_probability)

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="card">
                <div class="small-title">Hasil Prediksi Utama</div>
                <div class="weather-wrap">
                    <div class="weather-icon">{guide['icon']}</div>
                    <div>
                        <div class="weather-name">{guide['id']}</div>
                        <div class="pill {confidence_class}">{top_probability * 100:.1f}% • {confidence_text}</div>
                    </div>
                </div>
                <div class="text">{guide['summary']}</div>
                <div class="status">Status aktivitas: {guide['status']}<br>Level risiko: {guide['risk']}<br>Catatan: {confidence_note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        accuracy = float(metrics.get("accuracy", 0))
        f1_macro = float(metrics.get("f1_macro", 0))
        precision = float(metrics.get("precision_macro", 0))
        recall = float(metrics.get("recall_macro", 0))
        st.markdown(
            f"""
            <div class="card">
                <div class="small-title">Performa Model</div>
                <div class="kpi-grid">
                    <div class="kpi"><div class="kpi-label">Akurasi</div><div class="kpi-value">{accuracy * 100:.1f}%</div><div class="kpi-note">Dari 100 prediksi, sekitar {accuracy * 100:.0f} benar.</div></div>
                    <div class="kpi"><div class="kpi-label">F1-Macro</div><div class="kpi-value">{f1_macro * 100:.1f}%</div><div class="kpi-note">Keseimbangan performa untuk semua kelas.</div></div>
                    <div class="kpi"><div class="kpi-label">Precision</div><div class="kpi-value">{precision * 100:.1f}%</div><div class="kpi-note">Ketepatan saat model memilih kelas.</div></div>
                    <div class="kpi"><div class="kpi-label">Recall</div><div class="kpi-value">{recall * 100:.1f}%</div><div class="kpi-note">Kemampuan menemukan kelas yang benar.</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">✅ Rekomendasi Praktis</div>', unsafe_allow_html=True)
    render_tip_cards(guide["tips"])

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Peluang Cuaca", "🧠 Alasan Prediksi", "📈 Evaluasi Model", "📋 Data Input"])

    with tab1:
        chart_col, table_col = st.columns([1.45, .9], gap="large")
        with chart_col:
            st.markdown('<div class="section-title">Peluang Setiap Jenis Cuaca</div>', unsafe_allow_html=True)
            st.caption("Grafik ini menampilkan peluang tiap kelas dalam persen, bukan angka desimal panjang.")
            st.plotly_chart(make_probability_chart(probability_df), use_container_width=True)
        with table_col:
            gap = (top_probability - second_probability) * 100
            if gap >= 25:
                message = "Prediksi utama cukup dominan dibanding kelas lain."
            elif gap >= 10:
                message = "Prediksi utama unggul, tetapi kelas kedua masih perlu diperhatikan."
            else:
                message = "Peluang beberapa kelas berdekatan, jadi hasil perlu dibaca lebih hati-hati."
            st.markdown(
                f"""
                <div class="card">
                    <div class="small-title">Kesimpulan Probabilitas</div>
                    <div class="text">Model paling yakin pada <b>{guide['id']}</b> dengan peluang <b>{top_probability*100:.1f}%</b>.</div>
                    <div class="text" style="margin-top:12px;">Selisih dengan pilihan kedua: <b>{gap:.1f}%</b>.</div>
                    <div class="status">{message}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(probability_df[["Jenis Cuaca", "Peluang (%)"]], hide_index=True, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-title">Kenapa Model Mengarah ke Hasil Ini?</div>', unsafe_allow_html=True)
        st.caption("Penjelasan di bawah dibuat dengan bahasa sederhana, bukan kode mentah.")
        render_insights(build_insights(user_input))

    with tab3:
        st.markdown('<div class="section-title">Evaluasi Model dalam Bahasa Sederhana</div>', unsafe_allow_html=True)
        train_rows = int(metrics.get("train_rows", 0))
        test_rows = int(metrics.get("test_rows", 0))
        model_name = metrics.get("model", "Machine Learning Model")
        st.markdown(
            f"""
            <div class="card">
                <div class="text">
                    Model yang digunakan adalah <b>{model_name}</b>. Model belajar dari <b>{train_rows:,}</b> baris data latih
                    dan diuji pada <b>{test_rows:,}</b> baris data uji. Dengan akurasi sekitar <b>{accuracy*100:.1f}%</b>,
                    model ini sudah cukup baik sebagai alat bantu prediksi awal.
                </div>
                <div class="note" style="margin-top:14px;">
                    Penting: hasil ini bukan pengganti data cuaca resmi real-time. Aplikasi ini cocok untuk demonstrasi machine learning,
                    edukasi, dan estimasi awal berdasarkan pola pada dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        report = metrics.get("classification_report", {})
        rows = []
        for cls in ["Cloudy", "Rainy", "Snowy", "Sunny"]:
            if cls in report:
                rows.append({
                    "Jenis Cuaca": WEATHER_GUIDE.get(cls, {}).get("id", cls),
                    "Precision (%)": round(report[cls].get("precision", 0) * 100, 2),
                    "Recall (%)": round(report[cls].get("recall", 0) * 100, 2),
                    "F1-score (%)": round(report[cls].get("f1-score", 0) * 100, 2),
                    "Jumlah Data Uji": int(report[cls].get("support", 0)),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with tab4:
        st.markdown('<div class="section-title">Data yang Dikirim ke Model</div>', unsafe_allow_html=True)
        readable_input = input_df.rename(columns=COLUMN_LABELS).copy()
        for col in readable_input.columns:
            if readable_input[col].dtype == object:
                readable_input[col] = readable_input[col].map(display_name)
        st.dataframe(readable_input, hide_index=True, use_container_width=True)

        data = load_dataset_preview()
        if not data.empty:
            with st.expander("Lihat contoh dataset yang digunakan"):
                st.dataframe(data.head(12), use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="note" style="margin-top:26px;">
            <b>Kesimpulan:</b> Website ini tidak hanya menampilkan output model, tetapi juga mengubahnya menjadi informasi yang bisa dipakai:
            prediksi cuaca, tingkat keyakinan, rekomendasi aktivitas, alasan sederhana, dan performa model.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
