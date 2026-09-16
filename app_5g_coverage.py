import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from folium.plugins import HeatMap
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Hệ thống Dự đoán Vùng Phủ Sóng 5G (RSRP AI)",
    page_icon="📶",
    layout="wide"
)

# Nạp model và danh sách trạm phát sóng
@st.cache_resource
def load_assets():
    with open('model_5G.pkl', 'rb') as f:
        data = pickle.load(f)
    try:
        bs_df = pd.read_csv('base_station_data.csv')
    except Exception:
        bs_df = pd.DataFrame()
    return data['model'], data['features'], data['metrics'], bs_df

try:
    model, features, metrics, bs_df = load_assets()
except Exception:
    st.error("Chưa tìm thấy file 'model_5G.pkl'. Vui lòng kiểm tra lại file mô hình!")
    st.stop()

st.title("📶 HỆ THỐNG DỰ ĐOÁN VÙNG PHỦ SÓNG 5G TỪ DỮ LIỆU RSRP")
st.caption(f"Mô hình: XGBoost Regressor | Độ khớp R2: {metrics['r2']*100:.2f}% | Sai số RMSE: {metrics['rmse']:.2f} dBm")

tab1, tab2 = st.tabs(["📍 Chế độ Tọa Độ Đơn Lẻ (Radio Link)", "🗺️ Chế độ Bản Đồ Nhiệt Vùng Phủ 5G"])

# ==============================================================================
# --- TAB 1: DỰ ĐOÁN ĐIỂM ĐƠN LẺ NÂNG CAO (TRỰC QUAN HÓA TIA SÓNG) ---
# ==============================================================================
with tab1:
    st.subheader("🎯 Khảo sát tuyến truyền dẫn vô tuyến (BTS to UE Link)")
    
    col_in1, col_in2 = st.columns([1, 1])
    
    with col_in1:
        st.markdown("##### 📡 Thông số Trạm phát (gNodeB)")
        if not bs_df.empty:
            list_bts = bs_df['ECI'].astype(str).tolist()
            sel_bts = st.selectbox("Chọn trạm phục vụ:", list_bts, index=0, key="t1_bts_select")
            bts_info = bs_df[bs_df['ECI'].astype(str) == sel_bts].iloc[0]
            bts_lat = float(bts_info['LATITUDE (processed)'])
            bts_lon = float(bts_info['LONGITUDE (processed)'])
            bts_h = float(bts_info['Height'])
        else:
            bts_lat, bts_lon, bts_h = 29.5078, 106.7038, 35.0
            
        st.info(f"Tọa độ BTS: **{bts_lat:.5f}, {bts_lon:.5f}** | Chiều cao cột: **{bts_h:.1f} m**")
        in_freq = st.selectbox("Tần số sóng mang (kHz):", [636664, 630000, 427980, 426000], key="t1_freq")

    with col_in2:
        st.markdown("##### 📱 Thông số Máy thu người dùng (UE)")
        c_lat, c_lon = st.columns(2)
        in_lat = c_lat.number_input("Vĩ độ UE:", value=bts_lat + 0.0150, format="%.6f", key="t1_lat")
        in_lon = c_lon.number_input("Kinh độ UE:", value=bts_lon + 0.0180, format="%.6f", key="t1_lon")
        in_alt = st.number_input("Độ cao mặt đất đặt UE (m):", value=220.0, step=5.0, key="t1_alt")
        
        # Tự động tính khoảng cách 3D từ UE đến BTS đã chọn
        d_lat_m = (in_lat - bts_lat) * 111000.0
        d_lon_m = (in_lon - bts_lon) * (111000.0 * np.cos(np.radians(bts_lat)))
        dist_2d = np.sqrt(d_lat_m**2 + d_lon_m**2)
        calc_distance = np.sqrt(dist_2d**2 + (in_alt - bts_h)**2)
        st.success(f"Cự ly không gian 3D thực tế: **{calc_distance:,.1f} m** (~{calc_distance/1000:.2f} km)")

    btn_pred_single = st.button("🚀 PHÂN TÍCH CHẤT LƯỢNG TUYẾN SÓNG", type="primary", use_container_width=True, key="t1_btn_calc")

    if btn_pred_single:
        inp_df = pd.DataFrame([[in_lat, in_lon, calc_distance, in_alt, in_freq]], columns=features)
        rsrp_val = float(model.predict(inp_df)[0])
        
        st.markdown("---")
        # --- BẢNG CHỈ SỐ VÀ THANH ĐO MỨC SÓNG ---
        st.subheader("📊 Kết quả Đánh giá Tuyến truyền sóng")
        res_m1, res_m2, res_m3, res_m4 = st.columns(4)
        
        # Chuẩn hóa giá trị RSRP (-140 dBm -> -44 dBm)
        signal_pct = int(np.clip(((rsrp_val - (-125)) / ((-65) - (-125))) * 100, 0, 100))
        
        res_m1.metric("Công suất thu RSRP", f"{rsrp_val:.2f} dBm")
        res_m2.metric("Chất lượng đường truyền", f"{signal_pct}%")
        
        # Ước tính suy hao không gian tự do FSPL để đối sánh
        freq_mhz = in_freq / 1000.0 if in_freq > 10000 else in_freq
        fspl_est = 32.44 + 20 * np.log10(max(calc_distance/1000.0, 0.05)) + 20 * np.log10(max(freq_mhz, 1.0))
        res_m3.metric("Suy hao FSPL lý thuyết", f"{fspl_est:.1f} dB")
        
        if rsrp_val >= -85:
            res_m4.metric("Trạng thái", "RẤT MẠNH 🟢")
            qual_color = "green"
            qual_text = "Vùng phủ tối ưu (Full vạch sóng) - Đảm bảo QoS cao nhất."
        elif rsrp_val >= -100:
            res_m4.metric("Trạng thái", "TRUNG BÌNH 🟡")
            qual_color = "orange"
            qual_text = "Vùng phủ chấp nhận được - Đủ điều kiện duy trì kết nối ổn định."
        else:
            res_m4.metric("Trạng thái", "LÕM SÓNG 🔴")
            qual_color = "red"
            qual_text = "Vùng rìa ô suy hao lớn - Nguy cơ rớt gói dữ liệu hoặc mất sóng."

        st.progress(signal_pct / 100.0)
        st.caption(f"Đánh giá kỹ thuật: {qual_text}")

        # --- BẢN ĐỒ MÔ PHỎNG ĐƯỜNG TRUYỀN VÔ TUYẾN ---
        st.markdown("##### 🗺️ Trực quan hóa liên kết sóng (Radio Link)")
        mid_lat = (bts_lat + in_lat) / 2.0
        mid_lon = (bts_lon + in_lon) / 2.0
        
        m_link = folium.Map(
            location=[mid_lat, mid_lon], 
            zoom_start=14, 
            tiles="https://mt1.google.com/vt/lyrs=m&hl=vi&x={x}&y={y}&z={z}",
            attr="Google Maps"
        )
        
        # Điểm đặt trạm BTS
        folium.Marker(
            [bts_lat, bts_lon],
            popup=f"<b>Trạm gNodeB</b><br>Chiều cao: {bts_h}m",
            tooltip="Trạm phát sóng gNodeB",
            icon=folium.Icon(color="red", icon="tower-broadcast", prefix="fa")
        ).add_to(m_link)
        
        # Điểm đặt người dùng UE
        folium.Marker(
            [in_lat, in_lon],
            popup=f"<b>Thiết bị thu (UE)</b><br>RSRP: {rsrp_val:.2f} dBm",
            tooltip=f"Vị trí UE ({rsrp_val:.2f} dBm)",
            icon=folium.Icon(color=qual_color, icon="mobile", prefix="fa")
        ).add_to(m_link)
        
        # Đường thẳng tia sóng nối 2 điểm
        folium.PolyLine(
            locations=[[bts_lat, bts_lon], [in_lat, in_lon]],
            color=qual_color,
            weight=4,
            opacity=0.85,
            dash_array='8, 8',
            tooltip=f"Tia sóng vô tuyến | Cự ly: {calc_distance:,.1f}m | RSRP: {rsrp_val:.2f} dBm"
        ).add_to(m_link)
        
        # Vòng tròn bán kính bao quanh trạm
        folium.Circle(
            radius=calc_distance,
            location=[bts_lat, bts_lon],
            color="#3388ff",
            fill=False,
            weight=1.5,
            dash_array='4, 4'
        ).add_to(m_link)
        
        link_map_html = m_link._repr_html_()
        components.html(link_map_html, height=480)

        # --- KHẢ NĂNG ĐÁP ỨNG DỊCH VỤ 5G ---
        st.markdown("##### 📶 Khả năng đáp ứng dịch vụ người dùng")
        srv1, srv2, srv3 = st.columns(3)
        if rsrp_val >= -85:
            srv1.success("✅ **Video Ultra HD 4K/8K, VR/AR:** Mượt mà (Băng thông > 300 Mbps)")
            srv2.success("✅ **Đàm thoại VoNR:** Chất lượng chuẩn HD, trễ < 10ms")
            srv3.success("✅ **Cloud Gaming & Live Stream:** Tối ưu không giật lag")
        elif rsrp_val >= -100:
            srv1.warning("⚠️ **Video Full HD 1080p:** Đạt yêu cầu (Băng thông 30 - 80 Mbps)")
            srv2.success("✅ **Đàm thoại VoNR:** Tín hiệu ổn định")
            srv3.warning("⚠️ **Cloud Gaming:** Đôi khi có hiện tượng tăng độ trễ (Jitter)")
        else:
            srv1.error("❌ **Video Streaming:** Thường xuyên đứng hình tải lại")
            srv2.warning("⚠️ **Đàm thoại:** Dễ bị nhiễu và vỡ tiếng")
            srv3.error("❌ **Cloud Gaming:** Mất gói tin, kết nối chập chờn")

# ==============================================================================
# --- TAB 2: BẢN ĐỒ NHIỆT VÙNG PHỦ SÓNG (GIỮ NGUYÊN CHUẨN HOẠT ĐỘNG) ---
# ==============================================================================
with tab2:
    st.subheader("🗺️ Bản đồ nhiệt (Heatmap) lan truyền sóng theo trạm gNodeB")
    
    mode_bts = st.radio(
        "Lựa chọn nguồn trạm BTS:",
        ["📡 Trạm thực tế từ Dataset", "⛰️/🌊 Tự do chọn địa hình (Núi, Biển, Đô thị)"],
        horizontal=True,
        key="tab2_mode"
    )
    
    col_bs1, col_bs2, col_bs3 = st.columns([2, 1, 1])
    
    if mode_bts == "📡 Trạm thực tế từ Dataset" and not bs_df.empty:
        with col_bs1:
            list_bts = bs_df['ECI'].astype(str).tolist()
            selected_bts_id = st.selectbox("Chọn mã trạm ECI:", list_bts, index=0, key="tab2_bts_eci")
            bts_row = bs_df[bs_df['ECI'].astype(str) == selected_bts_id].iloc[0]
            center_lat = float(bts_row['LATITUDE (processed)'])
            center_lon = float(bts_row['LONGITUDE (processed)'])
            bts_height = float(bts_row['Height'])
            default_ue_alt = 220
    else:
        with col_bs1:
            preset_loc = st.selectbox(
                "Chọn khu vực địa hình mẫu:",
                [
                    "🌊 Ven biển / Mặt biển (Vũng Tàu - Độ cao 5m)",
                    "⛰️ Vùng đồi núi cao (Đà Lạt - Độ cao 1500m)",
                    "🏙️ Đô thị mật độ cao (TP. Hồ Chí Minh - Độ cao 15m)",
                    "📍 Nhập tọa độ thủ công tùy ý"
                ],
                key="tab2_preset"
            )
            
            if "Vũng Tàu" in preset_loc:
                center_lat, center_lon, bts_height, default_ue_alt = 10.3460, 107.0843, 45.0, 5
            elif "Đà Lạt" in preset_loc:
                center_lat, center_lon, bts_height, default_ue_alt = 11.9404, 108.4583, 30.0, 1500
            elif "Hồ Chí Minh" in preset_loc:
                center_lat, center_lon, bts_height, default_ue_alt = 10.7769, 106.7009, 35.0, 15
            else:
                c_sub1, c_sub2, c_sub3 = st.columns(3)
                center_lat = c_sub1.number_input("Vĩ độ (Lat):", value=10.8010, format="%.4f", key="tab2_custom_lat")
                center_lon = c_sub2.number_input("Kinh độ (Lon):", value=106.7110, format="%.4f", key="tab2_custom_lon")
                bts_height = c_sub3.number_input("Chiều cao cột anten (m):", value=30.0, key="tab2_custom_h")
                default_ue_alt = 20

    with col_bs2:
        radius_km = st.slider("Bán kính khảo sát (km):", min_value=1, max_value=15, value=5, step=1, key="tab2_radius")
        grid_alt = st.slider("Độ cao UE đo sóng (m):", min_value=0, max_value=2000, value=default_ue_alt, step=10, key="tab2_alt")

    with col_bs3:
        grid_freq = st.selectbox("Tần số khảo sát (kHz):", [636664, 630000, 427980], index=0, key="tab2_freq")
        st.write("")
        btn_map = st.button("🌐 VẼ VÙNG PHỦ SÓNG", type="primary", use_container_width=True, key="tab2_btn_map")

    if btn_map:
        with st.spinner("Đang tính toán suy hao theo địa hình mới bằng AI..."):
            num_radial_steps = 30
            num_angular_steps = 36
            radii = np.linspace(200, radius_km * 1000, num_radial_steps)
            angles = np.linspace(0, 2 * np.pi, num_angular_steps, endpoint=False)
            
            flat_lat = []
            flat_lon = []
            flat_dist = []
            
            for r in radii:
                for theta in angles:
                    d_lat = (r * np.cos(theta)) / 111000.0
                    d_lon = (r * np.sin(theta)) / (111000.0 * np.cos(np.radians(center_lat)))
                    flat_lat.append(center_lat + d_lat)
                    flat_lon.append(center_lon + d_lon)
                    dist_3d = np.sqrt(r**2 + (grid_alt - bts_height)**2)
                    flat_dist.append(dist_3d)
                    
            flat_lat = np.array(flat_lat)
            flat_lon = np.array(flat_lon)
            flat_dist = np.array(flat_dist)
            
            batch_df = pd.DataFrame({
                'Latitude': flat_lat,
                'Longitude': flat_lon,
                'Distance': flat_dist,
                'Altitude': np.full_like(flat_lat, grid_alt),
                'Frequency': np.full_like(flat_lat, grid_freq)
            })[features]
            
            pred_rsrp = model.predict(batch_df)
            
            norm_weight = np.clip((pred_rsrp - (-125.0)) / ((-65.0) - (-125.0)), 0.05, 1.0)
            heat_data = [[flat_lat[i], flat_lon[i], float(norm_weight[i])] for i in range(len(flat_lat))]
            
            m = folium.Map(
                location=[center_lat, center_lon], 
                zoom_start=12, 
                tiles="https://mt1.google.com/vt/lyrs=m&hl=vi&x={x}&y={y}&z={z}",
                attr="Google Maps"
            )
            
            # Cắm mốc khẳng định chủ quyền Hoàng Sa & Trường Sa của Việt Nam
            folium.Marker(
                [16.5367, 111.6094],
                popup="<b>Quần đảo Hoàng Sa</b><br>Thuộc chủ quyền thành phố Đà Nẵng, Việt Nam",
                tooltip="Quần đảo Hoàng Sa (Việt Nam)",
                icon=folium.Icon(color="red", icon="flag", prefix="fa")
            ).add_to(m)
            
            folium.Marker(
                [8.6444, 111.9194],
                popup="<b>Quần đảo Trường Sa</b><br>Thuộc chủ quyền tỉnh Khánh Hòa, Việt Nam",
                tooltip="Quần đảo Trường Sa (Việt Nam)",
                icon=folium.Icon(color="red", icon="flag", prefix="fa")
            ).add_to(m)
            HeatMap(heat_data, radius=22, blur=18, min_opacity=0.3, max_zoom=13).add_to(m)
            folium.Marker(
                [center_lat, center_lon],
                popup=f"Trạm gNodeB (Cột cao: {bts_height}m | Vị trí: {center_lat:.4f}, {center_lon:.4f})",
                tooltip="Trạm phát sóng gNodeB",
                icon=folium.Icon(color="red", icon="tower-broadcast", prefix="fa")
            ).add_to(m)
            
            map_html = m._repr_html_()
            components.html(map_html, height=550)
            
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Mức RSRP cao nhất", f"{np.max(pred_rsrp):.1f} dBm")
            c_m2.metric("Mức RSRP trung bình", f"{np.mean(pred_rsrp):.1f} dBm")
            c_m3.metric("Mức RSRP thấp nhất rìa ô", f"{np.min(pred_rsrp):.1f} dBm")
            # --- KHỐI THÔNG BÁO DÀNH CHO NGƯỜI DÙNG PHỔ THÔNG ---
            st.markdown("---")
            st.subheader("📢 Đánh giá chất lượng sóng thực tế cho khu vực")
            
            avg_rsrp = np.mean(pred_rsrp)
            max_rsrp = np.max(pred_rsrp)
            min_rsrp = np.min(pred_rsrp)
            
            # Tự động phân tích tình trạng phủ sóng theo giá trị trung bình
            if avg_rsrp >= -90:
                st.success(f"""
                🟢 **KẾT LUẬN: VÙNG PHỦ SÓNG RẤT MẠNH (ĐẠT CHUẨN CAO)**
                * **Trải nghiệm thực tế:** Điện thoại luôn hiển thị **đầy đủ 4 - 5 vạch sóng 5G**.
                * **Tốc độ mạng:** Tải video 4K/8K tức thì, gọi video không trễ, chơi game mượt mà.
                * **Khuyến nghị:** Vùng phủ tối ưu, không cần lắp thêm trạm phát phụ.
                """)
            elif avg_rsrp >= -105:
                st.warning(f"""
                🟡 **KẾT LUẬN: VÙNG PHỦ SÓNG Ở MỨC TRUNG BÌNH - KHÁ**
                * **Trải nghiệm thực tế:** Điện thoại hiển thị khoảng **2 - 3 vạch sóng**. 
                * **Tốc độ mạng:** Ngoài đường lướt web, xem YouTube Full HD bình thường. Tuy nhiên khi đi sâu vào trong nhà cao tầng, tầng hầm hoặc góc khuất có thể bị tụt tốc độ.
                * **Khuyến nghị:** Cân nhắc nâng độ cao anten phát hoặc chỉnh góc ngửa anten (tilt) để sóng vươn xa hơn.
                """)
            else:
                st.error(f"""
                🔴 **KẾT LUẬN: VÙNG PHỦ SÓNG YẾU / LÕM SÓNG NHIỀU NƠI**
                * **Trải nghiệm thực tế:** Điện thoại chỉ bắt được **0 - 1 vạch sóng**, chập chờn hoặc tự nhảy về mạng 4G.
                * **Tốc độ mạng:** Tải trang chậm, dễ bị đứng hình khi gọi video.
                * **Khuyến nghị:** Cự ly khảo sát đang quá xa trạm hoặc vật cản nhiều, cần quy hoạch lắp đặt thêm trạm phát phụ (Small Cell) tại khu vực này.
                """)

            # Bảng quy đổi màu sắc và thông số ra vạch sóng đời thực
            with st.expander("📖 Hướng dẫn đọc bản đồ & thông số (Dành cho người mới)", expanded=True):
                st.markdown("""
                | Màu sắc trên bản đồ | Mức sóng RSRP | Số vạch sóng trên điện thoại | Trải nghiệm người dùng |
                | :--- | :--- | :--- | :--- |
                | **🔴 Đỏ (Tâm trạm)** | **>-85 dBm** | ⭐⭐⭐⭐⭐ (5 vạch) | Cực nhanh, đứng ngay gần trạm phát. |
                | **🟡 Vàng / Xanh lục** | **-85 đến -100 dBm** | ⭐⭐⭐ (3 - 4 vạch) | Dùng tốt mọi nhu cầu lướt web, video, mạng xã hội. |
                | **🔵 Xanh lam / Tím** | **-100 đến -115 dBm**| ⭐ (1 - 2 vạch) | Mạng bắt đầu chậm, vào nhà kín dễ mất kết nối. |
                | **⚪ Mất màu (Ngoài rìa)**| **<-115 dBm** | ❌ (Không có sóng 5G) | Vùng lõm sóng, mất kết nối hoặc rớt về sóng 4G. |

                > **Mẹo dễ nhớ:** Đơn vị công suất thu **dBm luôn là số âm**. Con số càng nhỏ (càng gần số 0, ví dụ `-84 dBm`) thì sóng càng khỏe; số càng âm sâu (ví dụ `-120 dBm`) thì sóng càng yếu.
                """)