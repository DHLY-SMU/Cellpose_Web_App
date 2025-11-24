import streamlit as st
import numpy as np
import os
import torch
from cellpose import io, plot, models
from skimage import measure
import time
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import pandas as pd

# 设置 Streamlit 页面配置
st.set_page_config(
    page_title="细胞形态学分析工具",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Cellpose 模型加载 (使用 Streamlit 缓存，只加载一次) ---
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
        # 1. 使用 PIL 读取 UploadedFile，并转换为 NumPy 数组
        img_pil = Image.open(uploaded_file)
        img = np.array(img_pil) 

        # 确定通道数
        if img.ndim == 2:
            channels = [0, 0]
        elif img.ndim == 3:
            channels = [0, 0]
        else:
            st.error(f"图像 {file_name} 维度不正确。")
            return None
        
        # 运行 AI 分割
        masks, flows, styles = model_obj.eval(
            img,
            diameter=params['diameter'],
            cellprob_threshold=params['prob_threshold'],
            channels=channels,
            flow_threshold=params['flow_threshold']
        )
        
        # 统计分析
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
        
        # --- 创建可视化结果图 ---
        fig = plt.figure(figsize=(15, 6))

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(plot.mask_rgb(masks))
        ax1.set_title(f'1. Final Segmentation (n={valid_cells})', fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(flows[0])
        ax2.set_title("2. Vector Flows (流场)", fontweight='bold')
        ax2.axis('off')

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(flows[2], cmap='inferno')
        ax3.set_title(f"3. Probability Heatmap", fontweight='bold')
        ax3.axis('off')

        plt.tight_layout()
        
        # 关键：将 Matplotlib Figure 转换为 PNG 字节流用于下载
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        png_data = buf.getvalue()
        plt.close(fig) # 释放内存

        return {
            'File_Name': file_name,
            'Cell_Count': valid_cells,
            'Eccentricity_Mean': avg_ecc,
            'Solidity_Mean': avg_sol,
            'Polarization_Percent': polarization_percentage,
            'Visualization_PNG': png_data # 存储 PNG 字节数据
        }

    except Exception as e:
        st.error(f"❌ 运行 {file_name} 分析时发生严重错误: {e}")
        return None

# --- 3. Streamlit 主界面 ---
def main():
    st.title("🔬 Cellpose 驱动的细胞形态学分析 Web 工具")
    st.markdown("上传您的明场细胞图像 (`.tif`, `.jpg`)，并使用左侧参数调整分割效果。")
    st.markdown("---")

    # 1. 加载模型
    model_obj, device = load_cellpose_model()
    if model_obj is None:
        st.error("无法加载 Cellpose 模型。")
        return

    # 2. 侧边栏：参数调整区域
    st.sidebar.header("⚙️ 分割参数调整 (Cellpose)")
    
    # Cellpose 核心参数
    cell_diameter = st.sidebar.slider(
        '预估细胞直径 (px)', min_value=10, max_value=200, value=24, step=1,
        help="调整预估细胞大小。过小容易过分割，过大容易欠分割。"
    )
    prob_threshold = st.sidebar.slider(
        '细胞概率阈值', min_value=-6.0, max_value=6.0, value=-2.0, step=0.1,
        help="负值更激进，分割出更多细胞；正值更保守，只分割高置信度的区域。"
    )
    flow_threshold = st.sidebar.slider(
        '流场阈值', min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help="较低值（如0.1）能分割更小的细胞，但可能增加噪点。较高值（如0.8）只分割形态清晰的细胞。"
    )

    st.sidebar.header("📐 形态学过滤参数")
    
    ecc_threshold = st.sidebar.slider(
        '极化伸长度阈值 (Eccentricity)', min_value=0.50, max_value=0.95, value=0.70, step=0.01,
        help="伸长度（Eccentricity）高于此值的细胞才被计入极化细胞。"
    )
    area_min = st.sidebar.slider(
        '极化细胞最小面积 (px)', min_value=50, max_value=1000, value=200, step=10,
        help="面积小于此值的细胞不计入极化统计，用于排除微小碎片。"
    )

    # 3. 汇总参数字典
    analysis_params = {
        'diameter': cell_diameter,
        'prob_threshold': prob_threshold,
        'flow_threshold': flow_threshold,
        'ecc_threshold': ecc_threshold,
        'area_min': area_min
    }

    # 4. 文件上传（支持多文件）
    st.sidebar.header("📁 文件上传")
    uploaded_files = st.sidebar.file_uploader(
        "请选择一个或多个图像文件 (TIF, JPG)", 
        type=["tif", "tiff", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.subheader(f"共上传 {len(uploaded_files)} 张图片，请查看下方结果：")
        all_results_list = []
        
        # 批量处理进度条
        progress_bar = st.progress(0)
        
        # 遍历所有上传的文件
        for i, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            
            with st.container():
                st.markdown(f"#### 🔍 正在处理：{file_name}")
                
                # 运行分析
                results = run_cellpose_analysis_optimized(uploaded_file, model_obj, analysis_params)
                
                if results:
                    all_results_list.append(results)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    # 结果展示 (左侧)
                    with col1:
                        st.markdown("##### 🔑 形态学指标")
                        st.metric("细胞总数 (N)", results['Cell_Count'])
                        st.metric("平均伸长度", f"{results['Eccentricity_Mean']:.4f}")
                        st.metric("平均平滑度", f"{results['Solidity_Mean']:.4f}")
                        st.metric("极化细胞百分比", f"**{results['Polarization_Percent']:.2f}%**")
                        
                        # 下载可视化图片
                        st.download_button(
                            label=f"📥 下载 {file_name} 分割图 (PNG)",
                            data=results['Visualization_PNG'],
                            file_name=f"Segmentation_Viz_{file_name.split('.')[0]}.png",
                            mime="image/png",
                            key=f"download_png_{i}" # 必须有唯一的 key
                        )

                    # 可视化结果展示 (右侧)
                    with col2:
                        # 从 PNG 字节数据重新加载 Figure 对象，以便 Streamlit 显示
                        st.image(Image.open(BytesIO(results['Visualization_PNG'])), caption="分割和流场可视化", use_container_width=True)
                    
                st.markdown("---")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        # 5. 汇总结果 (所有文件分析完成后)
        if all_results_list:
            
            # 移除 Visualization_PNG 字段以创建数据框
            df_data = [{k: v for k, v in res.items() if k != 'Visualization_PNG'} for res in all_results_list]
            df_final = pd.DataFrame(df_data)
            
            st.header("📋 批量分析汇总表")
            st.dataframe(df_final)

            # 提供 CSV 下载按钮 (数据下载)
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