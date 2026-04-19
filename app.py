import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import cloudscraper
import plotly.express as px
import plotly.graph_objects as go
import re
import google.generativeai as genai

# ==========================================
# Yapay Zeka (Gemini) Ayarları
# ==========================================
@st.cache_data(ttl=3600)
def generate_ai_analysis(prompt, data_context, api_key):
    if not api_key:
        return "Yapay zeka analizi için API anahtarı gerekli."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        full_prompt = f"{prompt}\n\nİşte veri özeti:\n{data_context}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Yapay zeka analizi oluşturulurken bir hata oluştu: {e}"

# ==========================================
# 1. Web Scraping (İtopya TÜM RAM Verileri)
# ==========================================
@st.cache_data(ttl=3600)
def scrape_itopya_rams(max_pages=15):
    scraper = cloudscraper.create_scraper()
    data = []
    
    page = 1
    while page <= max_pages:
        url = f"https://www.itopya.com/rambellek_k10?ps=100&pg={page}"
        try:
            response = scraper.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                products = soup.find_all("div", class_="product")
                
                if not products:
                    break
                
                for p in products:
                    name_elem = p.find("h2", class_="truncate-text")
                    name = name_elem.get("title", "").strip() if name_elem else ""
                    
                    price_elem = p.find("span", class_="product-price")
                    price = ""
                    eski_fiyat = ""
                    if price_elem:
                        basket_price = price_elem.find("span", class_="product-price-warning")
                        price_strong = price_elem.find("strong")
                        if basket_price and "Sepette" in basket_price.text:
                            price = basket_price.text.replace("Sepette", "").strip()
                            eski_fiyat = price_strong.text.strip() if price_strong else price
                        else:
                            price = price_strong.text.strip() if price_strong else price_elem.text.strip()
                            eski_fiyat = price
                        
                    img_elem = p.select_one(".product-image img")
                    img_url = ""
                    if img_elem:
                        img_url = img_elem.get("data-src", "") or img_elem.get("src", "")
                        if img_url and not img_url.startswith("http"):
                            img_url = "https://www.itopya.com" + img_url
                            
                    link_elem = p.select_one(".product-image a")
                    product_url = ""
                    if link_elem and link_elem.get("href"):
                        product_url = "https://www.itopya.com" + link_elem.get("href")
                            
                    if name and price:
                        data.append({
                            "Model Adı": name,
                            "Fiyat": price,
                            "Eski Fiyat": eski_fiyat,
                            "Ürün Görseli": img_url,
                            "Ürün Linki": product_url
                        })
                page += 1
            else:
                break
        except Exception as e:
            st.warning(f"Sayfa {page} çekilirken bir sorun oluştu: {e}")
            break
            
    return pd.DataFrame(data)

# ==========================================
# 2. Veri Temizleme ve Analiz
# ==========================================
def clean_data(df):
    try:
        df_clean = df.copy()
        if df_clean.empty: return df_clean

        df_clean['Fiyat'] = df_clean['Fiyat'].astype(str).str.replace('TL', '', regex=False).str.strip()
        df_clean['Fiyat'] = df_clean['Fiyat'].str.replace('.', '', regex=False)
        df_clean['Fiyat'] = df_clean['Fiyat'].str.replace(',', '.', regex=False)
        df_clean['Fiyat'] = pd.to_numeric(df_clean['Fiyat'], errors='coerce')
        
        if 'Eski Fiyat' in df_clean.columns:
            df_clean['Eski Fiyat'] = df_clean['Eski Fiyat'].astype(str).str.replace('TL', '', regex=False).str.strip()
            df_clean['Eski Fiyat'] = df_clean['Eski Fiyat'].str.replace('.', '', regex=False)
            df_clean['Eski Fiyat'] = df_clean['Eski Fiyat'].str.replace(',', '.', regex=False)
            df_clean['Eski Fiyat'] = pd.to_numeric(df_clean['Eski Fiyat'], errors='coerce')
        else:
            df_clean['Eski Fiyat'] = df_clean['Fiyat']
        
        def extract_features(name):
            name_upper = name.upper()
            marka = name.split()[0] if name else "Bilinmiyor"
            
            cap_match = re.search(r'(\d+)\s*GB', name_upper)
            kapasite_gb = int(cap_match.group(1)) if cap_match else None
                
            freq_match = re.search(r'(\d+)\s*MHZ', name_upper)
            frekans_mhz = int(freq_match.group(1)) if freq_match else None
            if not frekans_mhz:
                freq_match_alt = re.search(r'(\d{4})', name_upper)
                if freq_match_alt: frekans_mhz = int(freq_match_alt.group(1))
                    
            type_match = re.search(r'(DDR[3-5])', name_upper)
            bellek_turu = type_match.group(1) if type_match else "DDR4"
                
            cl_match = re.search(r'CL\s*(\d+)', name_upper)
            cl_degeri = int(cl_match.group(1)) if cl_match else (16 if bellek_turu == "DDR4" else 36)
            
            kit_match = re.search(r'(2X|DUAL|KIT)', name_upper)
            kit_tipi = "Çift Modül (Dual/Kit)" if kit_match else "Tek Modül (Single)"
                
            return pd.Series([marka, kapasite_gb, frekans_mhz, bellek_turu, cl_degeri, kit_tipi])
            
        df_clean[['Marka', 'Kapasite (GB)', 'Frekans (MHz)', 'Bellek Türü', 'CL Değeri', 'Kit Tipi']] = df_clean['Model Adı'].apply(extract_features)
        df_clean = df_clean.dropna(subset=['Kapasite (GB)', 'Frekans (MHz)', 'Fiyat'])
        
        df_clean = df_clean.drop_duplicates(subset=['Model Adı'])
        
        df_clean['Gerçek Gecikme (ns)'] = ((df_clean['CL Değeri'] * 2000) / df_clean['Frekans (MHz)']).round(2)
        
        final_cols = ["Ürün Görseli", "Marka", "Model Adı", "Kapasite (GB)", "Bellek Türü", "Frekans (MHz)", "CL Değeri", "Gerçek Gecikme (ns)", "Fiyat", "Eski Fiyat", "Kit Tipi", "Ürün Linki"]
        return df_clean[final_cols]
    except Exception as e:
        st.error(f"Hata: {e}")
        return df

# ==========================================
# 3. Dinamik Puanlama Motoru
# ==========================================
def apply_dynamic_scoring(df, w_cap, w_freq, w_lat, dual_bonus):
    dff = df.copy()
    if dff.empty: return dff
    
    # Kullanıcının girdiği ağırlıkları üs olarak kullanarak fiyat makasının F/P oranını ezmesini engelliyoruz (Dynamic Range Protection)
    w_cap_exp = w_cap / 35 if w_cap > 0 else 0.01
    w_freq_exp = w_freq / 30 if w_freq > 0 else 0.01
    w_lat_exp = w_lat / 20 if w_lat > 0 else 0.01
    
    base_score = (dff['Kapasite (GB)'] ** w_cap_exp) * (dff['Frekans (MHz)'] ** w_freq_exp) * ((10 / dff['Gerçek Gecikme (ns)']) ** w_lat_exp)
    
    dff['Performans Skoru'] = base_score
    dff.loc[dff['Kit Tipi'] == "Çift Modül (Dual/Kit)", 'Performans Skoru'] *= (1 + dual_bonus)
    
    perf_max = dff['Performans Skoru'].max()
    dff['Performans Skoru'] = ((dff['Performans Skoru'] / perf_max) * 100).round(2) if perf_max > 0 else 0
    
    dff['Ham F/P'] = dff['Performans Skoru'] / dff['Fiyat']
    fp_max = dff['Ham F/P'].max()
    dff['F/P Skoru'] = ((dff['Ham F/P'] / fp_max) * 100).round(2) if fp_max > 0 else 0
    
    return dff.drop(columns=['Ham F/P'])

# ==========================================
# 4. Kıyaslama Grafikleri (YENİ)
# ==========================================
def render_comparison_charts(data, api_key):
    if data.empty:
        st.info("Kıyaslama yapılacak veri bulunamadı.")
        return
    
    c1, c2 = st.columns(2)
    selected_names = ", ".join(data['Model Adı'].tolist()[:15])
    context = f"Kıyaslanan ürünler: {selected_names}."

    with c1:
        st.markdown("##### 🏆 Performans Puanı Sıralaması")
        top_perf = data.sort_values("Performans Skoru", ascending=True)
        fig1 = px.bar(top_perf, x="Performans Skoru", y="Model Adı", orientation='h', color="Performans Skoru", color_continuous_scale='Viridis')
        st.plotly_chart(fig1, use_container_width=True)
        with st.spinner("Gemini Yorumluyor..."):
            prompt = "Sana sağlanan ürün listesi verilerine göre Performans Puanı sıralamasını yorumla. Grafik görmediğini varsay. En iyi performansı kim veriyor? Sadece verilen veriler üzerinden 2 cümleyle anlat."
            st.info(generate_ai_analysis(prompt, context, api_key))
        
    with c2:
        st.markdown("##### ⚡ Özellik Kıyaslaması (MHz ve GB)")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=data['Model Adı'], y=data['Frekans (MHz)'], name='Frekans (MHz)', marker_color='indianred'))
        fig2.add_trace(go.Bar(x=data['Model Adı'], y=data['Kapasite (GB)'] * 100, name='Kapasite (GB) x100', marker_color='lightsalmon'))
        fig2.update_layout(barmode='group')
        st.plotly_chart(fig2, use_container_width=True)
        with st.spinner("Gemini Yorumluyor..."):
            prompt = "Sağlanan RAM'lerin MHz ve GB verilerine bakarak kapasite ve hız odaklılıklarını yorumla. Grafik görmediğini varsay. Sadece verilen veriler üzerinden 2 cümleyle anlat."
            st.info(generate_ai_analysis(prompt, context, api_key))
        
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### ⏱️ Gecikme (ns) Cezası (Düşük Daha İyi)")
        top_lat = data.sort_values("Gerçek Gecikme (ns)", ascending=False)
        fig3 = px.bar(top_lat, x="Model Adı", y="Gerçek Gecikme (ns)", color="Gerçek Gecikme (ns)", color_continuous_scale='Reds')
        fig3.update_yaxes(autorange="reversed")
        st.plotly_chart(fig3, use_container_width=True)
        with st.spinner("Gemini Yorumluyor..."):
            prompt = "Verilen ürünlerin gecikme (ns) sürelerini yorumla. Grafik görmediğini varsay. Hangi model daha avantajlı? Sadece verilen veriler üzerinden 2 cümleyle anlat."
            st.info(generate_ai_analysis(prompt, context, api_key))
        
    with c4:
        st.markdown("##### 🎯 Fiyat / Performans Analizi")
        fig4 = px.scatter(data, x="Performans Skoru", y="Fiyat", color="Marka", size="Kapasite (GB)", hover_name="Model Adı")
        st.plotly_chart(fig4, use_container_width=True)
        with st.spinner("Gemini Yorumluyor..."):
            prompt = "Sana sağlanan ürün listesi verilerine göre, F/P (fiyat/performans) açısından en mantıklı yatırımı yorumla. Grafik görmediğini varsay. Sadece verilen veriler üzerinden 2 cümleyle anlat."
            st.info(generate_ai_analysis(prompt, context, api_key))

# ==========================================
# 5. Ana Uygulama Gövdesi
# ==========================================
def main():
    st.set_page_config(page_title="AI RAM Asistanı", page_icon="🤖", layout="wide")
    
    if 'step' not in st.session_state: st.session_state.step = 0
    if 'filters' not in st.session_state: st.session_state.filters = {}
    if 'auto_top10' not in st.session_state: st.session_state.auto_top10 = False
    if 'select_all' not in st.session_state: st.session_state.select_all = False
    if 'editor_key' not in st.session_state: st.session_state.editor_key = 0
    if 'initial_top10_set' not in st.session_state: st.session_state.initial_top10_set = False
        
    def set_step(step):
        st.session_state.step = step
        
    def reset_wizard():
        st.session_state.step = 0
        st.session_state.filters = {}
        st.session_state.auto_top10 = False
        st.session_state.select_all = False
        st.session_state.initial_top10_set = False
        st.session_state.editor_key += 1

    with st.sidebar:
        st.title("🔑 API Ayarları")
        api_key = st.text_input("Gemini API Anahtarı", type="password", placeholder="AIzaSy...")
        if not api_key:
            st.warning("⚠️ Yapay zeka analizleri için API anahtarı girmelisiniz.")
        st.divider()

    st.title("🤖 Yapay Zeka Destekli RAM Danışmanı")
    if st.session_state.step < 7:
        st.markdown("İhtiyacınıza en uygun, fiyat/performans canavarı belleği bulmak için adım adım ilerleyin.")
    st.divider()

    raw_df = None
    with st.spinner("İtopya veritabanı taranıyor..."):
        raw_df = scrape_itopya_rams()
        
    if raw_df is None or raw_df.empty:
        st.error("Veri çekilemedi. Lütfen bağlantınızı kontrol edin.")
        return
        
    df = clean_data(raw_df)

    # Ödev gereksinimleri için verileri her çalışmada klasörlere kaydet
    import os
    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(app_dir)
        dataset_dir = os.path.join(project_dir, "Dataset")
        excel_dir = os.path.join(project_dir, "Excel Report")
        
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(excel_dir, exist_ok=True)
        
        df.to_csv(os.path.join(dataset_dir, "itopya_ram_veriseti.csv"), index=False, encoding='utf-8-sig')
        df.to_excel(os.path.join(excel_dir, "ram_analiz_raporu.xlsx"), index=False)
    except Exception as e:
        pass

    # --- SİHİRBAZ EKRANLARI (0-6) ---
    if st.session_state.step == 0:
        st.info("Bu asistan, İtopya'daki tüm RAM modellerini tarayarak bütçenize ve sisteminize en uygun donanımı bilimsel olarak tespit eder.")
        
        if st.button("🚀 Asistanı Başlat", use_container_width=True, type="primary"):
            set_step(1)
            st.rerun()
            
        st.divider()
        st.header("📊 Pazarın Genel Durumu (Tüm İtopya Verisi)")
        
        c1, c2, c3 = st.columns(3)
        summary_context = f"Toplam Ürün: {len(df)}\nOrtalama Fiyat: {df['Fiyat'].mean():.2f} TL\nEn yüksek Frekans: {df['Frekans (MHz)'].max()} MHz\nDDR4 sayısı: {len(df[df['Bellek Türü']=='DDR4'])}, DDR5 sayısı: {len(df[df['Bellek Türü']=='DDR5'])}"

        with c1:
            fig_scatter = px.scatter(df, x="Fiyat", y="Frekans (MHz)", color="Bellek Türü", hover_name="Model Adı", title="Fiyat & Frekans İlişkisi")
            st.plotly_chart(fig_scatter, use_container_width=True)
            with st.spinner("Gemini Yorumluyor..."):
                prompt = "Verilen istatistiklere dayanarak, RAM hızları (MHz) arttıkça fiyatlar nasıl bir eğilim gösteriyor? Grafik görmediğini bilerek sadece veri üzerinden 2 cümleyle yorumla."
                st.info(generate_ai_analysis(prompt, summary_context, api_key))
        with c2:
            fig_box = px.box(df, x="Bellek Türü", y="Fiyat", color="Bellek Türü", title="Nesillere Göre Fiyat Dağılımı")
            st.plotly_chart(fig_box, use_container_width=True)
            with st.spinner("Gemini Yorumluyor..."):
                prompt = "Verilen istatistiklere göre, DDR4 ve DDR5 nesillerinin genel fiyat farklarını özetle. Hangi nesil daha pahalı? Sadece sağlanan verilere dayanarak 2 cümleyle yorumla."
                st.info(generate_ai_analysis(prompt, summary_context, api_key))
        with c3:
            brand_counts = df["Marka"].value_counts().reset_index()
            fig_pie = px.pie(brand_counts, values='count', names='Marka', hole=0.4, title="Marka Dağılımı", color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
            with st.spinner("Gemini Yorumluyor..."):
                prompt = "Sana verilen verilerdeki DDR4 ve DDR5 sayılarına ve ortalama fiyatlara bakarak pazarın rekabet durumunu kısa 2 cümleyle yorumla."
                st.info(generate_ai_analysis(prompt, summary_context, api_key))
            
        st.divider()
        st.header("📋 Tüm RAM Veritabanı")
        st.dataframe(df, use_container_width=True)

    elif st.session_state.step == 1:
        st.subheader("Soru 1: Anakartınız hangi bellek türünü destekliyor?")
        secim = st.radio("DDR Türü", ["Emin değilim / Fark etmez", "DDR4", "DDR5"], horizontal=True)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("İleri ➡️", use_container_width=True):
            if secim != "Emin değilim / Fark etmez": st.session_state.filters['DDR'] = secim
            set_step(2)
            st.rerun()
        if c2.button("Tüm Soruları Atla (Sonuçları Göster)", type="primary", use_container_width=True):
            set_step(7)
            st.rerun()

    elif st.session_state.step == 2:
        st.subheader("Soru 2: Toplam kaç GB kapasite istiyorsunuz?")
        all_caps = sorted(df["Kapasite (GB)"].dropna().unique())
        secim = st.selectbox("Kapasite (GB)", ["Fark etmez"] + [f"{int(c)} GB" for c in all_caps])
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("İleri ➡️", use_container_width=True):
            if secim != "Fark etmez": st.session_state.filters['Kapasite'] = [int(secim.split()[0])]
            set_step(3)
            st.rerun()
        if c2.button("Tüm Soruları Atla (Sonuçları Göster)", type="primary", use_container_width=True):
            set_step(7)
            st.rerun()

    elif st.session_state.step == 3:
        st.subheader("Soru 3: Bütçeniz ne kadar?")
        min_p = float(df["Fiyat"].min())
        max_p = float(df["Fiyat"].max())
        secim = st.slider("Maksimum Fiyat (TL)", min_value=min_p, max_value=max_p, value=max_p, step=250.0)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("İleri ➡️", use_container_width=True):
            if secim < max_p: st.session_state.filters['Max Fiyat'] = secim
            set_step(4)
            st.rerun()
        if c2.button("Tüm Soruları Atla (Sonuçları Göster)", type="primary", use_container_width=True):
            set_step(7)
            st.rerun()

    elif st.session_state.step == 4:
        st.subheader("Soru 4: En az kaç MHz hıza ihtiyacınız var?")
        min_f = int(df["Frekans (MHz)"].min())
        max_f = int(df["Frekans (MHz)"].max())
        secim = st.slider("Minimum Frekans (MHz)", min_value=min_f, max_value=max_f, value=min_f, step=200)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("İleri ➡️", use_container_width=True):
            if secim > min_f: st.session_state.filters['Min MHz'] = secim
            set_step(5)
            st.rerun()
        if c2.button("Tüm Soruları Atla (Sonuçları Göster)", type="primary", use_container_width=True):
            set_step(7)
            st.rerun()

    elif st.session_state.step == 5:
        st.subheader("Soru 5: Maksimum CL (Gecikme) değeri ne olsun?")
        min_cl = int(df["CL Değeri"].min())
        max_cl = int(df["CL Değeri"].max())
        secim = st.slider("Maksimum CL", min_value=min_cl, max_value=max_cl, value=max_cl, step=1)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("İleri ➡️", use_container_width=True):
            if secim < max_cl: st.session_state.filters['Max CL'] = secim
            set_step(6)
            st.rerun()
        if c2.button("Tüm Soruları Atla (Sonuçları Göster)", type="primary", use_container_width=True):
            set_step(7)
            st.rerun()
            
    elif st.session_state.step == 6:
        st.subheader("Soru 6: Modül Tipi Tercihiniz (Tek mi Çift mi?)")
        secim = st.radio("Kit Tipi", ["Fark etmez", "Tek Modül (Single)", "Çift Modül (Dual/Kit)"], horizontal=True)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("Sonuçları Göster 🎉", type="primary", use_container_width=True):
            if secim != "Fark etmez": st.session_state.filters['Kit Tipi'] = [secim]
            set_step(7)
            st.rerun()
        if c2.button("Tüm Soruları Atla (Sonuçları Göster)", type="primary", use_container_width=True):
            set_step(7)
            st.rerun()

    # --- ADIM 7: SONUÇ EKRANI ---
    elif st.session_state.step == 7:
        if not st.session_state.initial_top10_set:
            st.session_state.auto_top10 = True
            st.session_state.initial_top10_set = True

        with st.sidebar:
            st.title("🧠 YZ Öncelik Ayarları")
            st.markdown("Performans puanını hesaplarken hangi özelliklerin sizin için daha önemli olduğunu belirleyin.")
            w_cap = st.slider("Kapasite (GB) Ağırlığı", 0, 100, 35)
            w_freq = st.slider("Frekans (MHz) Ağırlığı", 0, 100, 30)
            w_lat = st.slider("Gecikme (ns) Ağırlığı", 0, 100, 20)
            
            st.info("💡 Çift Modül (Dual Channel) RAM'lere performansı artırdığı için otomatik olarak **%15 Bonus** eklenir.")
            dual_bonus = 0.15
            
            st.divider()
            st.title("🛠️ Filtre Paneli")
            
            all_brands = sorted(df["Marka"].unique())
            all_caps = sorted(df["Kapasite (GB)"].dropna().unique())
            all_kits = sorted(df["Kit Tipi"].unique())
            
            def_caps = st.session_state.filters.get('Kapasite', all_caps)
            def_kits = st.session_state.filters.get('Kit Tipi', all_kits)
            
            min_p, max_p = float(df["Fiyat"].min()), float(df["Fiyat"].max())
            def_max_p = st.session_state.filters.get('Max Fiyat', max_p)
            
            min_f, max_f = int(df["Frekans (MHz)"].min()), int(df["Frekans (MHz)"].max())
            def_min_f = st.session_state.filters.get('Min MHz', min_f)
            
            with st.expander("Marka Seçimi"):
                sel_brands = st.multiselect("Markalar", options=all_brands, default=all_brands)
            
            with st.expander("Bellek Türü ve Kapasite"):
                sel_ddr_types = ["DDR4", "DDR5"]
                if 'DDR' in st.session_state.filters:
                    sel_ddr = st.multiselect("Bellek Türü", options=sel_ddr_types, default=[st.session_state.filters['DDR']])
                else:
                    sel_ddr = st.multiselect("Bellek Türü", options=sel_ddr_types, default=sel_ddr_types)
                sel_caps = st.multiselect("Kapasite (GB)", options=all_caps, default=def_caps)
                
            with st.expander("Kit Tipi"):
                sel_kits = st.multiselect("Kit Tipi", options=all_kits, default=def_kits)
            
            st.subheader("Fiyat ve Hız")
            sel_price = st.slider("Fiyat (TL)", min_p, max_p, (min_p, float(def_max_p)))
            sel_freq = st.slider("Frekans (MHz)", min_f, max_f, (int(def_min_f), max_f))

            st.divider()
            if st.button("🔄 Tüm Filtreleri Sıfırla", use_container_width=True):
                reset_wizard()
                st.rerun()

        # Skoru tüm verisetine uygula
        df_scored = apply_dynamic_scoring(df, w_cap, w_freq, w_lat, dual_bonus)

        filtered_df = df_scored[
            (df_scored["Marka"].isin(sel_brands)) & 
            (df_scored["Bellek Türü"].isin(sel_ddr)) &
            (df_scored["Kapasite (GB)"].isin(sel_caps)) & 
            (df_scored["Kit Tipi"].isin(sel_kits)) &
            (df_scored["Fiyat"] >= sel_price[0]) & (df_scored["Fiyat"] <= sel_price[1]) &
            (df_scored["Frekans (MHz)"] >= sel_freq[0]) & (df_scored["Frekans (MHz)"] <= sel_freq[1])
        ]
        
        if 'Max CL' in st.session_state.filters:
            filtered_df = filtered_df[filtered_df["CL Değeri"] <= st.session_state.filters['Max CL']]

        if filtered_df.empty:
            st.error("Seçimlerinize uygun hiçbir RAM bulunamadı. Lütfen sol menüden filtreleri esnetin.")
            return

        st.success(f"Taleplerinize uygun {len(filtered_df)} adet RAM listeleniyor.")
        st.divider()

        # --- İLK 10 RAM KARTLARI ---
        st.subheader("🌟 Sizin İçin Önerdiklerimiz (En İyi 10 RAM)")
        top_10 = filtered_df.nlargest(10, 'F/P Skoru').reset_index(drop=True)
        
        cols = st.columns(5)
        for idx, row in top_10.iterrows():
            col_idx = idx % 5
            with cols[col_idx]:
                st.image(row["Ürün Görseli"], use_container_width=True)
                st.markdown(f"**{row['Marka']}** {row['Kapasite (GB)']}GB {row['Frekans (MHz)']}MHz")
                if pd.notna(row['Eski Fiyat']) and row['Fiyat'] < row['Eski Fiyat']:
                    st.markdown(f"🔥 **İNDİRİMDE!**")
                    st.markdown(f"💰 ~~{row['Eski Fiyat']} TL~~ **{row['Fiyat']} TL**")
                else:
                    st.markdown(f"💰 **{row['Fiyat']} TL**")
                st.markdown(f"🏆 Puan: **{row['F/P Skoru']}/100**")
                st.markdown(f"[🛒 İncele]({row['Ürün Linki']})")
                st.write("") 
                
        st.divider()

        # --- BİRLEŞTİRİLMİŞ İNTERAKTİF TABLO ---
        st.header("📋 Tüm Ürünler ve Kıyaslama Tablosu")
        st.info("Aşağıdaki listeden istediğiniz ürünlerin başındaki kutucuğu işaretleyerek detaylı kıyaslama yapabilirsiniz.")
        
        df_display = filtered_df.sort_values(by="F/P Skoru", ascending=False).reset_index(drop=True)
        df_display.insert(0, "Kıyasla", False)
        
        if st.session_state.auto_top10:
            df_display.loc[:min(9, len(df_display)-1), "Kıyasla"] = True
        elif st.session_state.select_all:
            df_display["Kıyasla"] = True
            
        edited_df = st.data_editor(
            df_display,
            key=f"data_editor_{st.session_state.editor_key}",
            column_config={
                "Kıyasla": st.column_config.CheckboxColumn("Kıyasla"),
                "Ürün Görseli": None,
                "Ürün Linki": st.column_config.LinkColumn("Satın Al", display_text="🔗 Git"),
                "Fiyat": st.column_config.NumberColumn("Fiyat", format="%.2f ₺"),
                "Gerçek Gecikme (ns)": st.column_config.NumberColumn("Gecikme", format="%.2f ns"),
                "F/P Skoru": st.column_config.ProgressColumn("Puan", format="%.f", min_value=0, max_value=100)
            },
            disabled=["Marka", "Model Adı", "Kapasite (GB)", "Bellek Türü", "Frekans (MHz)", "CL Değeri", "Kit Tipi", "Gerçek Gecikme (ns)", "Fiyat", "F/P Skoru", "Ürün Linki"],
            hide_index=True,
            use_container_width=True,
            height=400
        )
        
        b1, b2, b3 = st.columns(3)
        if b1.button("⭐ İlk 10'u Seç", use_container_width=True):
            st.session_state.auto_top10 = True
            st.session_state.select_all = False
            st.session_state.editor_key += 1
            st.rerun()
            
        if b2.button("✅ Hepsini Seç", use_container_width=True):
            st.session_state.select_all = True
            st.session_state.auto_top10 = False
            st.session_state.editor_key += 1
            st.rerun()
            
        if b3.button("❌ Seçimleri Sıfırla", use_container_width=True):
            st.session_state.auto_top10 = False
            st.session_state.select_all = False
            st.session_state.editor_key += 1
            st.rerun()
        
        # --- DİNAMİK GRAFİK ÇİZİMİ ---
        selected_rows = edited_df[edited_df["Kıyasla"] == True]
        
        st.divider()
        if not selected_rows.empty:
            st.header(f"⚖️ Seçili Ürünlerin Özel Kıyaslaması")
            render_comparison_charts(selected_rows, api_key)
        else:
            st.info("Kıyaslama grafiklerini görmek için yukarıdaki tablodan ürün seçiniz.")

if __name__ == "__main__":
    main()
