import streamlit as st
import os
import cv2
import torch
import numpy as np
import timm
import random
import datetime
import uuid
import base64
from io import BytesIO
from PIL import Image
from torchvision import transforms
from insightface.app import FaceAnalysis
from pymongo import MongoClient

# ── SAYFA AYARLARI ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Tespit Platformu", page_icon="🔍", layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 1.5rem !important;
    max-width: 1480px;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }

/* ── Butonlar ── */
.stButton > button {
    background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
    color: white; border-radius: 12px; font-weight: 700; border: none;
    padding: .65rem 1.4rem;
    box-shadow: 0 4px 16px rgba(74,144,226,.4);
    transition: all .22s ease; font-size: .95rem;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #357ABD 0%, #2860a0 100%);
    box-shadow: 0 7px 22px rgba(74,144,226,.55);
    transform: translateY(-2px);
}

/* ── HERO / Upload alanı (KOMPAKT) ── */
.hero-wrap {
    text-align: center;
    padding: 1.2rem 1.5rem .9rem;
    background: linear-gradient(160deg, #f8fafc 0%, #eef2f7 100%);
    border-radius: 18px;
    margin-bottom: 1rem;
    border: 1px solid #e2e8f0;
}
.hero-title {
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    color: #0f172a;
    margin-bottom: .25rem !important;
    letter-spacing: -.01em;
}
.hero-sub {
    color: #64748b;
    font-size: .85rem;
    max-width: 560px;
    margin: 0 auto !important;
    line-height: 1.5;
}

/* ── ANALİZ EKRANI (kompakt) ── */
.analyzing-wrap {
    text-align: center;
    padding: 1.4rem 1.5rem;
    background: linear-gradient(160deg, #f0f9ff 0%, #e0f2fe 100%);
    border-radius: 16px;
    border: 1px solid #bae6fd;
    margin: 1rem 0;
}
.analyzing-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #0c4a6e;
    margin: 0 0 .25rem;
}
.analyzing-sub {
    color: #075985;
    font-size: .82rem;
    margin: 0;
}

/* ── Verdict banner ── */
.verdict-banner {
    border-radius: 18px;
    padding: 1.3rem 1.7rem;
    display: flex; align-items: center; gap: 1.4rem;
    margin-top: .5rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 12px 36px rgba(0,0,0,.15);
    position: relative;
    overflow: hidden;
}
.verdict-banner::before {
    content: ''; position: absolute; top: 0; right: 0;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(255,255,255,.16) 0%, transparent 70%);
    border-radius: 50%;
    transform: translate(70px, -70px);
}
.verdict-real    { background: linear-gradient(135deg, #047857 0%, #10b981 55%, #34d399 100%); }
.verdict-fake    { background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 55%, #f87171 100%); }
.verdict-suspicious { background: linear-gradient(135deg, #9a3412 0%, #ea580c 55%, #fb923c 100%); }
.verdict-unknown { background: linear-gradient(135deg, #1f2937 0%, #4b5563 100%); }
.verdict-banner * { color: white; position: relative; z-index: 1; }
.verdict-icon    {
    font-size: 2.6rem; flex-shrink: 0; line-height: 1;
    filter: drop-shadow(0 4px 10px rgba(0,0,0,.22));
}
.verdict-text-wrap { flex: 1; }
.verdict-title   {
    font-size: 1.15rem; font-weight: 900;
    letter-spacing: .01em; line-height: 1.2;
}
.verdict-sub     {
    font-size: .9rem; opacity: .95; margin-top: .3rem;
    font-weight: 500;
}
.verdict-meta    {
    font-size: .74rem; opacity: .82; margin-top: .45rem;
    font-weight: 500;
}
.verdict-confidence {
    font-size: 2.1rem; font-weight: 900; flex-shrink: 0;
    text-align: right; letter-spacing: -.03em;
    line-height: 1;
}
.verdict-conf-label {
    font-size: .65rem; font-weight: 700; opacity: .85;
    text-align: right; letter-spacing: .1em;
    text-transform: uppercase; margin-top: .22rem;
}

/* ── Sonuç panelleri ── */
.result-panel {
    border-radius: 18px;
    padding: 1.2rem 1.2rem .8rem;
    display: flex; flex-direction: column;
    box-shadow: 0 6px 24px rgba(0,0,0,.05);
}
.panel-real {
    background: linear-gradient(175deg, #f0fdf4 0%, #ffffff 50%);
    border: 2px solid #86efac;
}
.panel-fake {
    background: linear-gradient(175deg, #fef2f2 0%, #ffffff 50%);
    border: 2px solid #fca5a5;
}

.result-panel-header {
    display: flex; align-items: center; gap: .6rem;
    padding-bottom: .8rem; margin-bottom: .8rem;
    border-bottom: 2px solid #e2e8f0;
}
.panel-header-icon { font-size: 1.4rem; }
.panel-title {
    font-size: .98rem; font-weight: 800;
    color: #0f172a; flex: 1;
    letter-spacing: .01em;
}
.panel-count {
    font-size: .8rem; font-weight: 900;
    padding: .3rem .8rem; border-radius: 22px;
    min-width: 38px; text-align: center;
}
.count-real { background: #16a34a; color: white; box-shadow: 0 3px 10px rgba(22,163,74,.35); }
.count-fake { background: #dc2626; color: white; box-shadow: 0 3px 10px rgba(220,38,38,.35); }

.panel-empty-state {
    text-align: center; padding: 2.5rem 1rem;
    color: #94a3b8; font-size: .88rem;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.empty-icon {
    font-size: 2.8rem; display: block;
    margin-bottom: .7rem; opacity: .35;
}
.empty-text { font-weight: 600; color: #64748b; }

/* ── Kişi kartları ── */
.person-card {
    display: flex; align-items: center; gap: 1rem;
    background: white;
    border-radius: 14px;
    padding: .85rem 1rem;
    margin-bottom: .7rem;
    box-shadow: 0 3px 14px rgba(0,0,0,.06);
    border: 1px solid #f1f5f9;
    transition: all .25s ease;
}
.person-card:last-child { margin-bottom: 0; }
.person-card:hover {
    box-shadow: 0 8px 26px rgba(0,0,0,.12);
    transform: translateY(-2px);
}
.card-real { border-left: 5px solid #22c55e; }
.card-fake { border-left: 5px solid #ef4444; }
.card-suspicious { border-left: 5px solid #f97316; }

.face-img {
    width: 92px; height: 92px;
    object-fit: cover; border-radius: 12px; flex-shrink: 0;
    border: 3px solid #fff;
    box-shadow: 0 3px 12px rgba(0,0,0,.12);
}
.card-real .face-img { box-shadow: 0 3px 12px rgba(34,197,94,.28); }
.card-fake .face-img { box-shadow: 0 3px 12px rgba(239,68,68,.28); }
.card-suspicious .face-img { box-shadow: 0 3px 12px rgba(249,115,22,.28); }

.card-info { flex: 1; min-width: 0; }
.card-name {
    font-size: .95rem; font-weight: 800;
    color: #0f172a; margin-bottom: .3rem;
    letter-spacing: -.01em;
}

.status-badge {
    display: inline-block; font-size: .65rem; font-weight: 900;
    padding: .22rem .65rem; border-radius: 22px;
    margin-bottom: .5rem; letter-spacing: .06em;
}
.badge-real { background: #dcfce7; color: #166534; }
.badge-fake { background: #fee2e2; color: #991b1b; }
.badge-suspicious { background: #ffedd5; color: #9a3412; }

.score-label {
    font-size: .65rem; color: #64748b;
    margin-bottom: .22rem; font-weight: 700;
    letter-spacing: .07em; text-transform: uppercase;
}
.score-bar-wrap {
    height: 9px; background: #f1f5f9;
    border-radius: 12px; overflow: hidden;
    margin-bottom: .25rem;
    box-shadow: inset 0 1px 3px rgba(0,0,0,.06);
}
.score-bar-fill {
    height: 100%; border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,.15);
    transition: width .6s ease;
}
.score-row { display: flex; justify-content: space-between; align-items: center; }
.score-pct {
    font-size: .9rem; font-weight: 900;
    color: #1e293b; letter-spacing: -.01em;
}
.frame-count {
    font-size: .7rem; color: #64748b;
    font-weight: 600; background: #f8fafc;
    padding: .18rem .5rem; border-radius: 10px;
}

/* ── Örnek kareler (küçük başlık) ── */
.frames-label {
    font-size: .78rem; font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin: 1.2rem 0 .6rem;
}

/* ── Örnek kare görselleri sınırla ── */
.sample-frames-wrap img {
    max-height: 180px !important;
    object-fit: cover !important;
    width: 100% !important;
    border-radius: 10px;
}
.sample-frames-wrap [data-testid="stImage"] {
    max-height: 180px !important;
    overflow: hidden !important;
}
.sample-frames-wrap [data-testid="stImage"] > img {
    max-height: 180px !important;
    object-fit: cover !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def bgr_to_b64(bgr_img, size=(140, 140)):
    try:
        rgb = cv2.cvtColor(cv2.resize(bgr_img, size), cv2.COLOR_BGR2RGB)
        buf = BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""


def bar_color(score):
    if score >= 0.65:
        return "linear-gradient(90deg, #dc2626, #ef4444)"
    if score >= 0.50:
        return "linear-gradient(90deg, #ea580c, #f97316)"
    return "linear-gradient(90deg, #16a34a, #22c55e)"


def render_card(ident, idx, status):
    score = ident["final_score"]
    pct = score * 100
    b64 = bgr_to_b64(ident["best_crop"])
    bc = bar_color(score)
    n = len(ident["probs"])

    if status == "REAL":
        cc, bb, lbl = "card-real", "badge-real", "GERÇEK"
    elif status == "FAKE":
        cc, bb, lbl = "card-fake", "badge-fake", "SAHTE"
    else:
        cc, bb, lbl = "card-suspicious", "badge-suspicious", "ŞÜPHELİ"

    img_tag = (
        f'<img src="data:image/jpeg;base64,{b64}" class="face-img" />'
        if b64
        else '<div style="width:92px;height:92px;background:#e2e8f0;border-radius:12px;flex-shrink:0;"></div>'
    )

    return f"""
<div class="person-card {cc}">
  {img_tag}
  <div class="card-info">
    <div class="card-name">Kişi {idx + 1}</div>
    <span class="status-badge {bb}">{lbl}</span>
    <div class="score-label">Risk Skoru</div>
    <div class="score-bar-wrap">
      <div class="score-bar-fill" style="width:{pct:.1f}%;background:{bc};"></div>
    </div>
    <div class="score-row">
      <span class="score-pct">%{pct:.1f}</span>
      <span class="frame-count">{n} kare</span>
    </div>
  </div>
</div>"""


def render_verdict(result, identities, fake_count, suspicious_count, filename):
    total = len(identities)
    ts_str = (datetime.datetime.utcnow() + datetime.timedelta(hours=3)).strftime(
        "%d.%m.%Y %H:%M"
    )
    fn = (filename[:32] + "…") if len(filename) > 32 else filename

    if result == "FAKE":
        cls, icon = "verdict-fake", "🚨"
        title = "KESİN MANİPÜLASYON TESPİT EDİLDİ"
        sub = f"Taranan {total} kişiden {fake_count} kişi deepfake olarak işaretlendi."
        cv = identities[0]["final_score"] * 100 if identities else 0
        conf, conf_lbl = f"%{cv:.1f}", "risk skoru"
    elif result == "SUSPICIOUS":
        cls, icon = "verdict-suspicious", "⚠️"
        title = "ŞÜPHELİ MEDYA TESPİT EDİLDİ"
        sub = f"{suspicious_count} kişinin yüzünde anomali / piksellenme tespit edildi."
        cv = identities[0]["final_score"] * 100 if identities else 0
        conf, conf_lbl = f"%{cv:.1f}", "risk skoru"
    elif result == "REAL":
        cls, icon = "verdict-real", "✅"
        title = "ORİJİNAL MEDYA — DOĞRULANDI"
        sub = f"Taranan {total} kişinin tamamı orijinal olarak doğrulandı."
        cv = (1 - identities[0]["final_score"]) * 100 if identities else 100
        conf, conf_lbl = f"%{cv:.1f}", "güven"
    else:
        cls, icon = "verdict-unknown", "❓"
        title = "YÜZ TESPİT EDİLEMEDİ"
        sub = "Analiz için net ve en az 80×80 px büyüklüğünde bir yüz gereklidir."
        conf, conf_lbl = "—", ""

    return f"""
<div class="verdict-banner {cls}">
  <div class="verdict-icon">{icon}</div>
  <div class="verdict-text-wrap">
    <div class="verdict-title">{title}</div>
    <div class="verdict-sub">{sub}</div>
    <div class="verdict-meta">{fn} &nbsp;·&nbsp; {ts_str} &nbsp;·&nbsp; {total} kişi analiz edildi</div>
  </div>
  <div>
    <div class="verdict-confidence">{conf}</div>
    <div class="verdict-conf-label">{conf_lbl}</div>
  </div>
</div>"""


# ── MODEL YÜKLEME ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    fa = FaceAnalysis(
        name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    fa.prepare(ctx_id=0, det_size=(640, 640))

    base = os.path.dirname(os.path.abspath(__file__))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = timm.create_model("efficientnet_b4", pretrained=False, num_classes=1)

    candidates = [
        os.path.join(base, "models", "dfdc_model_v2_best.pth"),
        os.path.join(base, "dfdc_model_v2_best.pth"),
    ]

    loaded_ok = False
    for weights in candidates:
        if not os.path.exists(weights):
            print(f"⏭️  Atlanıyor: {weights}")
            continue
        try:
            raw = torch.load(weights, map_location=dev, weights_only=False)
            if isinstance(raw, dict) and "model_state" in raw:
                state_dict = raw["model_state"]
            elif isinstance(raw, dict) and "state_dict" in raw:
                state_dict = raw["state_dict"]
            else:
                state_dict = raw
            if set(mdl.state_dict().keys()) - set(state_dict.keys()):
                continue
            mdl.load_state_dict(state_dict, strict=True)
            mdl.eval()
            loaded_ok = True
            break
        except Exception as e:
            print(f"❌ Yükleme hatası: {e}")
            continue

    if not loaded_ok:
        print("❌ HİÇBİR MODEL YÜKLENEMEDİ!")

    return fa, mdl.to(dev).eval()


# ── MONGODB ────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_mongo():
    uri = os.environ.get("MONGO_URI", "")
    if not uri:
        return None, False, "MONGO_URI bulunamadı"
    try:
        c = MongoClient(uri, serverSelectionTimeoutMS=5000)
        c.server_info()
        return c["deepfake_db"]["analiz_logs"], True, "Başarılı"
    except Exception as e:
        return None, False, str(e)


# ── YÜKLEMELER ─────────────────────────────────────────────────────────────────
with st.spinner("Yapay zeka modelleri yükleniyor…"):
    face_app, model = load_models()
collection, db_connected, db_error = get_mongo()


def save_sample_frames(frames, base_dir):
    out = os.path.join(base_dir, "history_samples")
    os.makedirs(out, exist_ok=True)
    paths = []
    for fr in frames:
        name = (
            datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            + f"_{uuid.uuid4().hex[:8]}.jpg"
        )
        p = os.path.join(out, name)
        try:
            Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).save(p)
            paths.append(p)
        except Exception:
            pass
    return paths


# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in [
    ("phase", "upload"),
    ("identities", []),
    ("overall_result", None),
    ("fake_count", 0),
    ("suspicious_count", 0),
    ("sampled_frames", []),
    ("analyzed_filename", ""),
    ("tmp_path", None),
    ("is_video", False),
    ("balloons_shown", False),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── SOL MENÜ ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="color:#FF6B00;font-size:1.15rem;font-weight:700;margin-bottom:.4rem;">Proje Künyesi</p>',
        unsafe_allow_html=True,
    )
    st.write("**Üniversite:** Kırıkkale Üniversitesi")
    st.write("**Bölüm:** Bilgisayar Mühendisliği")
    st.write("**Geliştiriciler:** Çağatay Altın & Yavuz Bahri Kadıoğlu")
    st.write("**Proje Sorumlusu:** Dr. Öğr. Üyesi Enes AYAN")
    st.markdown("---")
    st.caption("Bitirme Projesi Kapsamında Geliştirilmiştir.")
    st.markdown("---")
    st.markdown("**Yönetim Paneli**")

    if db_connected:
        st.success("Veritabanı: Bağlı")
    else:
        st.error("Veritabanı: Bağlantı yok")
        st.code(f"HATA:\n{db_error}", language="bash")

    if st.button("Geçmiş Analizler", use_container_width=True):
        if db_connected and collection is not None:
            try:
                kayitlar = list(
                    collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(10)
                )
                if kayitlar:
                    for k in kayitlar:
                        ts = k.get("timestamp")
                        saat = (
                            ts.strftime("%d.%m.%Y %H:%M")
                            if hasattr(ts, "strftime")
                            else "-"
                        )
                        with st.sidebar.expander(
                            f"{saat} | {k.get('filename', '-')} | {k.get('result', '-')}"
                        ):
                            st.write(f"**Karar:** {k.get('result', '-')}")
                            existing = [
                                p
                                for p in k.get("sample_paths", [])
                                if os.path.exists(p)
                            ]
                            if existing:
                                cols = st.columns(len(existing))
                                for i, ip in enumerate(existing):
                                    try:
                                        with cols[i]:
                                            st.image(
                                                ip,
                                                use_column_width=True,
                                                caption=f"Kare {i + 1}",
                                            )
                                    except Exception:
                                        pass
                            else:
                                st.caption("Örnek kare bulunamadı.")
                else:
                    st.sidebar.info("Henüz kayıt yok.")
            except Exception as e:
                st.sidebar.error(f"Hata: {e}")
        else:
            st.sidebar.warning("Veritabanına bağlanılamadı.")

    if st.button("Veritabanını Sıfırla", use_container_width=True):
        if db_connected and collection is not None:
            try:
                n = collection.delete_many({}).deleted_count
                st.sidebar.success(f"{n} kayıt silindi.")
            except Exception as e:
                st.sidebar.error(f"Hata: {e}")
        else:
            st.sidebar.warning("Veritabanına bağlanılamadı.")


# ═══════════════════════════════════════════════════════════════════════════════
#  UPLOAD AŞAMASI
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "upload":
    _, mid, _ = st.columns([1, 3, 1])

    with mid:
        st.markdown(
            """
        <div class="hero-wrap">
          <h1 class="hero-title">Deepfake Tespit Sistemi</h1>
          <p class="hero-sub">
            Video veya resim yükleyin — yapay zeka her yüzü ayrı ayrı analiz edip
            gerçeği sahteden ayırır.
          </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Dosya seç",
            type=["mp4", "avi", "mov", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            tmp_dir = os.path.join(base_dir, "temp_dir")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            is_video = uploaded_file.name.lower().endswith((".mp4", ".avi", ".mov"))

            file_size_kb = len(uploaded_file.getbuffer()) / 1024
            size_str = (
                f"{file_size_kb:.1f} KB"
                if file_size_kb < 1024
                else f"{file_size_kb / 1024:.2f} MB"
            )
            file_type = "Video" if is_video else "Resim"

            st.markdown(
                f"""
            <div style="background:white;border-radius:12px;padding:.85rem 1.1rem;
                        margin:.8rem 0;border:1px solid #e2e8f0;
                        box-shadow:0 2px 10px rgba(0,0,0,.04);
                        display:flex;align-items:center;gap:1rem;">
              <div style="flex:1;">
                <div style="font-weight:700;color:#0f172a;font-size:.9rem;">{uploaded_file.name}</div>
                <div style="color:#64748b;font-size:.76rem;margin-top:.12rem;">
                  {file_type} &nbsp;·&nbsp; {size_str} &nbsp;·&nbsp; Yüklemeye hazır
                </div>
              </div>
              <div style="color:#22c55e;font-weight:700;font-size:.78rem;
                          background:#dcfce7;padding:.3rem .7rem;border-radius:18px;">HAZIR</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if st.button("Analizi Başlat ve Sonucu Gör", use_container_width=True):
                st.session_state.tmp_path = tmp_path
                st.session_state.is_video = is_video
                st.session_state.analyzed_filename = uploaded_file.name
                st.session_state.balloons_shown = False
                st.session_state.phase = "analyzing"
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALİZ AŞAMASI
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "analyzing":
    st.markdown(
        """
    <div class="analyzing-wrap">
      <div class="analyzing-title">Yapay Zeka Analiz Ediyor</div>
      <div class="analyzing-sub">Biyometrik kimlikler çıkarılıyor ve her yüz ayrı ayrı taranıyor…</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    tmp_path = st.session_state.tmp_path
    is_video = st.session_state.is_video
    base_dir = os.path.dirname(os.path.abspath(__file__))

    frames_to_check = []

    if is_video:
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 and not np.isnan(fps) else 30.0
        step = max(1, int(fps))
        fi = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if fi % step == 0 and fr is not None:
                frames_to_check.append(fr)
            fi += 1
        cap.release()

        if not frames_to_check:
            try:
                import av

                cont = av.open(tmp_path)
                fps2 = float(cont.streams.video[0].average_rate or 30)
                step2 = max(1, int(fps2))
                for i, f2 in enumerate(cont.decode(video=0)):
                    if i % step2 == 0:
                        arr = f2.to_ndarray(format="bgr24")
                        if arr is not None and arr.size > 0:
                            frames_to_check.append(arr)
                cont.close()
            except Exception:
                pass
    else:
        fr = cv2.imread(tmp_path)
        if fr is not None:
            frames_to_check.append(fr)

    if frames_to_check:
        tf = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        identities = []
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ph = st.empty()
        pbar = st.progress(0.0)

        for idx, fr in enumerate(frames_to_check):
            # Küçük görselleri büyüt — telefon fotoğraflarında yüz algılama iyileşir
            proc = fr
            scale_factor = 1.0
            min_dim = min(proc.shape[0], proc.shape[1])
            if min_dim < 640:
                scale_factor = 640 / min_dim
                proc = cv2.resize(
                    proc,
                    None,
                    fx=scale_factor,
                    fy=scale_factor,
                    interpolation=cv2.INTER_LINEAR,
                )

            faces = face_app.get(proc)
            if faces:
                for face in faces:
                    if face.det_score < 0.50:
                        continue
                    b = face.bbox.astype(int)
                    # Orijinal koordinatlara dönüştür
                    if scale_factor != 1.0:
                        b = (b / scale_factor).astype(int)
                    CROP = 1.5
                    sz = int(max(b[2] - b[0], b[3] - b[1]) * CROP)
                    cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                    h, w = fr.shape[:2]
                    x1, y1 = max(0, cx - sz // 2), max(0, cy - sz // 2)
                    x2, y2 = min(w, x1 + sz), min(h, y1 + sz)
                    crop = fr[y1:y2, x1:x2]
                    if crop.shape[0] < 40 or crop.shape[1] < 40:
                        continue

                    ph.image(
                        cv2.cvtColor(cv2.resize(crop, (200, 200)), cv2.COLOR_BGR2RGB),
                        caption="Biyometrik Tarama Sürüyor…",
                        width=200,
                    )
                    t = (
                        tf(
                            Image.fromarray(
                                cv2.cvtColor(
                                    cv2.resize(crop, (256, 256)), cv2.COLOR_BGR2RGB
                                )
                            )
                        )
                        .unsqueeze(0)
                        .to(dev)
                    )
                    with torch.no_grad():
                        prob = torch.sigmoid(model(t)).item()

                    emb = face.normed_embedding
                    if emb is not None:
                        matched = False
                        for ident in identities:
                            if np.dot(emb, ident["embedding"]) > 0.45:
                                ident["probs"].append(prob)
                                ident["embedding"] = (ident["embedding"] + emb) / 2
                                ident["embedding"] /= np.linalg.norm(ident["embedding"])
                                matched = True
                                break
                        if not matched:
                            identities.append(
                                {"embedding": emb, "best_crop": crop, "probs": [prob]}
                            )
            pbar.progress((idx + 1) / len(frames_to_check))

        pbar.empty()
        ph.empty()

        SUSPICIOUS_T = 0.50
        FAKE_T = 0.65
        min_kare = 5 if is_video else 1
        fake_count = suspicious_count = 0
        valid = []
        for ident in identities:
            if len(ident["probs"]) >= min_kare:
                sp = sorted(ident["probs"])
                n = len(sp)
                trim = max(1, int(n * 0.1))
                trimmed = sp[trim:-trim] if n > 2 * trim else sp
                ident["final_score"] = float(np.median(trimmed))
                if ident["final_score"] >= FAKE_T:
                    ident["status"] = "FAKE"
                    fake_count += 1
                elif ident["final_score"] >= SUSPICIOUS_T:
                    ident["status"] = "SUSPICIOUS"
                    suspicious_count += 1
                else:
                    ident["status"] = "REAL"
                valid.append(ident)

        identities = sorted(valid, key=lambda x: x["final_score"], reverse=True)

        if identities:
            overall = (
                "FAKE"
                if fake_count > 0
                else ("SUSPICIOUS" if suspicious_count > 0 else "REAL")
            )
            db_p = identities[0]["final_score"]
        else:
            overall = "UNKNOWN"
            db_p = 0.0

        sampled = random.sample(frames_to_check, min(3, len(frames_to_check)))
        saved_paths = save_sample_frames(sampled, base_dir)

        if db_connected and collection is not None and overall != "UNKNOWN":
            try:
                collection.insert_one(
                    {
                        "filename": st.session_state.analyzed_filename,
                        "result": overall,
                        "confidence": f"%{db_p * 100:.2f}",
                        "sample_paths": saved_paths,
                        "timestamp": datetime.datetime.utcnow()
                        + datetime.timedelta(hours=3),
                    }
                )
            except Exception:
                pass

        st.session_state.identities = identities
        st.session_state.overall_result = overall
        st.session_state.fake_count = fake_count
        st.session_state.suspicious_count = suspicious_count
        st.session_state.sampled_frames = [
            cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in sampled
        ]

        try:
            os.remove(tmp_path)
        except Exception:
            pass

        st.session_state.phase = "results"
        st.rerun()
    else:
        st.error("Medya formatı çözümlenemedi. Geçerli bir dosya yükleyin.")
        if st.button("← Geri Dön"):
            st.session_state.phase = "upload"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  SONUÇLAR AŞAMASI
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "results":
    identities = st.session_state.identities
    overall = st.session_state.overall_result
    fake_count = st.session_state.fake_count
    suspicious_count = st.session_state.suspicious_count
    sampled = st.session_state.sampled_frames
    filename = st.session_state.analyzed_filename

    st.markdown(
        render_verdict(overall, identities, fake_count, suspicious_count, filename),
        unsafe_allow_html=True,
    )

    # Balloons sadece ilk girişte (geçmiş analizler butonu tetiklemez)
    if overall == "REAL" and not st.session_state.balloons_shown:
        st.balloons()
        st.session_state.balloons_shown = True

    real_people = [i for i in identities if i["status"] == "REAL"]
    fake_people = [i for i in identities if i["status"] in ("FAKE", "SUSPICIOUS")]

    col_real, col_fake = st.columns(2, gap="large")

    with col_real:
        real_cards = "".join(
            render_card(ident, i, "REAL") for i, ident in enumerate(real_people)
        )
        empty_html = (
            '<div class="panel-empty-state">'
            '<span class="empty-icon"></span>'
            '<div class="empty-text">Gerçek olarak tespit edilen<br>kişi bulunamadı.</div>'
            "</div>"
            if not real_people
            else ""
        )
        st.markdown(
            f"""
        <div class="result-panel panel-real">
          <div class="result-panel-header">
            <span class="panel-header-icon"></span>
            <span class="panel-title">GERÇEK Tespit Edilenler</span>
            <span class="panel-count count-real">{len(real_people)}</span>
          </div>
          <div>
            {real_cards or empty_html}
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_fake:
        fake_cards = "".join(
            render_card(ident, i, ident["status"])
            for i, ident in enumerate(fake_people)
        )
        empty_html = (
            '<div class="panel-empty-state">'
            '<span class="empty-icon"></span>'
            '<div class="empty-text">Sahte veya şüpheli tespit edilen<br>kişi bulunamadı.</div>'
            "</div>"
            if not fake_people
            else ""
        )
        st.markdown(
            f"""
        <div class="result-panel panel-fake">
          <div class="result-panel-header">
            <span class="panel-header-icon"></span>
            <span class="panel-title">SAHTE / ŞÜPHELİ Tespit Edilenler</span>
            <span class="panel-count count-fake">{len(fake_people)}</span>
          </div>
          <div>
            {fake_cards or empty_html}
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Örnek kareler — küçük, sade
    if sampled:
        st.markdown(
            '<div class="frames-label">Analiz Edilen Örnek Kareler</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sample-frames-wrap">', unsafe_allow_html=True)
        fcols = st.columns(len(sampled), gap="medium")
        for i, (fc, fr) in enumerate(zip(fcols, sampled)):
            with fc:
                # ── DÜZELTME: görselleri küçült, max 400px genişlik ──
                h, w = fr.shape[:2]
                max_w = 400
                if w > max_w:
                    scale = max_w / w
                    fr = cv2.resize(
                        fr, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA
                    )
                st.image(fr, use_column_width=True, caption=f"Kare {i + 1}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Reset butonu — sade, başlıksız
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    rc = st.columns([1.5, 2, 1.5])
    with rc[1]:
        if st.button("Yeni Analiz Başlat", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
