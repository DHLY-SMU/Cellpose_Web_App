import streamlit as st
import numpy as np
import os
import torch
from cellpose import io, plot, models
from skimage import measure
import time
import matplotlib.pyplot as plt
from io import BytesIO 
import pandas as pd
from PIL import Image # <--- 修正：导入 Pillow 的 Image 模块

# 设置 Streamlit 页面配置
st.set_page_config(
    page_title="细胞形态学分析工具",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Cellpose 模型加载 (使用 Streamlit 缓存，只加载一次) ---
# ... (此函数内容保持不变)
@st.cache_resource
def load_cellpose_model():
    """检查并设置设备 (MPS/CUDA/CPU)，然后加载 Cellpose 模型"""
    try:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            st.info("--- [√] 检测到 Apple MPS (Metal) 加速器，已启用！---")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            st.info("--- [√] 检测到 CUDA (NVIDIA) 加速器，已启用！---")
        else:
            device = torch.device("cpu")
            st.warning("--- [!] 未检测到硬件加速器，将使用 CPU。 ---")

        st.info(f"--- 正在加载 Cellpose 模型到设备: {device} ---")
        model = models.CellposeModel(gpu=(device.type != 'cpu'), model_type='cyto')
        model.net.to(device)
        return model, device
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        return None, None

# --- 2. 核心分析函数 ---
def run_cellpose_analysis_optimized(uploaded_file, model_obj, params):

    file_name = uploaded_file.name

    try:
        # 1. 图像预处理和缩放 (使用 PIL 读取 UploadedFile)
        img_pil = Image.open(uploaded_file)
        
        # === 图像预处理/缩放逻辑 (提速关键) ===
        max_size = params.get('max_size', 2048) # 从 params 中获取 max_size
        original_shape = img_pil.size 
        
        if max(original_shape) > max_size:
            scale_factor = max_size / max(original_shape)
            new_width = int(img_pil.width * scale_factor)
            new_height = int(img_pil.height * scale_factor)
            # 使用 Image.LANCZOS (兼容性最好的高质量重采样方法)
            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS) 
            st.info(f"⚠️ 图像已从 {original_shape[0]}x{original_shape[1]} 自动缩放至 {new_width}x{new_height} 进行分析。")

        img = np.array(img_pil) 
        # ... (其余通道检测和 Cellpose 运行逻辑保持不变) ...
        channels = [0, 0] # 明场图
        
        # 运行 AI 分割
        masks, flows, styles = model_obj.eval(
            img,
            diameter=params['diameter'],
            cellprob_threshold=params['prob_threshold'],
            channels=channels,
            flow_threshold=params['flow_threshold']
        )
        
        # 统计分析 (保持不变，使用 params['ecc_threshold'] 进行判断)
        regions = measure.regionprops(masks)
        valid_cells = 0
        polarized_cells = 0
        eccentricities = []
        solidities = []

        for prop in regions:
            if prop.area < 50: continue
            valid_cells += 1
            ecc = prop.eccentricity
            solidity = prop.area / prop.convex_area if prop.convex_area > 0 else 0
            eccentricities.append(ecc)
            solidities.append(solidity)
            # 极化判断
            if ecc > params['ecc_threshold'] and prop.area >= params['area_min']:
                polarized_cells += 1

        avg_ecc = np.mean(eccentricities) if valid_cells > 0 else 0
        avg_sol = np.mean(solidities) if valid_cells > 0 else 0
        polarization_percentage = (polarized_cells / valid_cells) * 100 if valid_cells > 0 else 0
        
        # ... (创建可视化图表的代码保持不变) ...
        fig = plt.figure(figsize=(15, 6))
        # ... (绘图代码) ...
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        png_data = buf.getvalue()
        plt.close(fig)

        return {
            'File_Name': file_name,
            'Cell_Count': valid_cells,
            'Eccentricity_Mean': avg_ecc,
            'Solidity_Mean': avg_sol,
            'Polarization_Percent': polarization_percentage,
            'Visualization_PNG': png_data
        }

    except Exception as e:
        st.error(f"❌ 运行 {file_name} 分析时发生严重错误: {e}")
        return None

# --- 3. Streamlit 主界面 ---
def main():
    st.title("🔬 Cellpose 驱动的细胞形态学分析 Web 工具")
    st.markdown("上传您的明场细胞图像 (`.tif`, `.jpg`)，并使用左侧参数调整分割效果。")
    st.markdown("---")

    model_obj, device = load_cellpose_model()
    if model_obj is None: return

    # 2. 侧边栏：参数调整区域
    st.sidebar.header("⚙️ 分割参数调整 (Cellpose)")
    
    # NEW: 图像缩放参数 (添加到侧边栏，启用提速功能)
    max_size = st.sidebar.slider(
        '图像最大边长限制 (px)', min_value=512, max_value=2048, value=1024, step=256,
        help="限制图像的最大处理尺寸，以加快 Streamlit Cloud 上的 CPU 分析速度。"
    )
    
    # Cellpose 核心参数 (保持不变)
    cell_diameter = st.sidebar.slider('预估细胞直径 (px)', min_value=10, max_value=200, value=24, step=1)
    prob_threshold = st.sidebar.slider('细胞概率阈值', min_value=-6.0, max_value=6.0, value=-2.0, step=0.1)
    flow_threshold = st.sidebar.slider('流场阈值', min_value=0.0, max_value=1.0, value=0.4, step=0.05)

    st.sidebar.header("📐 形态学过滤参数")
    
    # *** 关键修正：极化阈值默认值设为 0.85 ***
    ecc_threshold = st.sidebar.slider(
        '极化伸长度阈值 (Eccentricity)', min_value=0.50, max_value=0.95, 
        value=0.85, # 默认值修正为 0.85
        step=0.01,
        help="伸长度（Eccentricity）高于此值的细胞才被计入极化细胞。0.85 要求细胞非常细长。"
    )
    area_min = st.sidebar.slider('极化细胞最小面积 (px)', min_value=50, max_value=1000, value=200, step=10)

    # 3. 汇总参数字典 (现在包含 max_size)
    analysis_params = {
        'max_size': max_size,  # NEW
        'diameter': cell_diameter,
        'prob_threshold': prob_threshold,
        'flow_threshold': flow_threshold,
        'ecc_threshold': ecc_threshold,
        'area_min': area_min
    }

    # 4. 文件上传 (保持不变)
    st.sidebar.header("📁 文件上传")
    uploaded_files = st.sidebar.file_uploader(
        "请选择一个或多个图像文件 (TIF, JPG)", 
        type=["tif", "tiff", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.subheader(f"共上传 {len(uploaded_files)} 张图片，请查看下方结果：")
        all_results_list = []
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            # ... (运行逻辑保持不变)
            file_name = uploaded_file.name
            
            with st.container():
                st.markdown(f"#### 🔍 正在处理：{file_name}")
                
                # 运行分析 (这里应该使用一个带有进度条/状态文本的函数，但为简洁使用这个版本)
                results = run_cellpose_analysis_optimized(uploaded_file, model_obj, analysis_params)
                
                if results:
                    all_results_list.append(results)
                    # ... (结果展示和下载按钮的代码) ...
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("##### 🔑 形态学指标")
                        st.metric("细胞总数 (N)", results['Cell_Count'])
                        st.metric("平均伸长度", f"{results['Eccentricity_Mean']:.4f}")
                        st.metric("平均平滑度", f"{results['Solidity_Mean']:.4f}")
                        st.metric("极化细胞百分比", f"**{results['Polarization_Percent']:.2f}%**")
                        
                        st.download_button(
                            label=f"📥 下载 {file_name} 分割图 (PNG)",
                            data=results['Visualization_PNG'],
                            file_name=f"Segmentation_Viz_{file_name.split('.')[0]}.png",
                            mime="image/png",
                            key=f"download_png_{i}"
                        )

                    with col2:
                        st.image(Image.open(BytesIO(results['Visualization_PNG'])), caption="分割和流场可视化", use_container_width=True)
                    
                st.markdown("---")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        # 5. 汇总结果 (所有文件分析完成后)
        if all_results_list:
            df_data = [{k: v for k, v in res.items() if k != 'Visualization_PNG'} for res in all_results_list]
            df_final = pd.DataFrame(df_data)
            
            st.header("📋 批量分析汇总表")
            st.dataframe(df_final)

            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 下载汇总数据 (CSV)",
                data=csv,
                file_name='cellpose_analysis_summary.csv',
                mime='text/csv',
            )
            st.success("🎉 所有文件分析完成，数据和图片下载链接已生成。")


if __name__ == "__main__":
    main()
