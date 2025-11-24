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
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            st.warning("--- [!] Streamlit Cloud 环境默认使用 CPU。速度较慢。 ---")

        st.info(f"--- 正在加载 Cellpose 模型到设备: {device} ---")
        model = models.CellposeModel(gpu=(device.type != 'cpu'), model_type='cyto')
        model.net.to(device)
        return model, device
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        return None, None

# --- 2. 核心分析函数 ---
def run_cellpose_analysis_optimized(uploaded_file, model_obj, params, progress_bar, status_text_container):

    file_name = uploaded_file.name
    
    # 模拟进度条的更新函数
    def update_progress(percent, message):
        progress_bar.progress(percent)
        status_text_container.text(f"🚀 进度: {message}")

    try:
        # 0. 初始化
        update_progress(0, "正在读取图像文件...")
        
        # 1. 使用 PIL 读取 UploadedFile
        img_pil = Image.open(uploaded_file)
        
        # 图像预处理/缩放逻辑
        max_size = params['max_size']
        original_shape = img_pil.size
        
        if max(original_shape) > max_size:
            scale_factor = max_size / max(original_shape)
            new_width = int(img_pil.width * scale_factor)
            new_height = int(img_pil.height * scale_factor)
            img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            st.info(f"⚠️ **为加快速度，图像已从 {original_shape[0]}x{original_shape[1]} 自动缩放至 {new_width}x{new_height} 进行分析。**")

        img = np.array(img_pil) 

        # 确定通道数
        if img.ndim == 2:
            channels = [0, 0]
        elif img.ndim == 3:
            channels = [0, 0]
        else:
            st.error(f"图像 {file_name} 维度不正确。")
            return None
        
        # 2. 运行 AI 分割 (耗时最长的步骤)
        update_progress(10, "正在进行 Cellpose AI 分割...")
        start_eval_time = time.time()
        
        masks, flows, styles = model_obj.eval(
            img,
            diameter=params['diameter'],
            cellprob_threshold=params['prob_threshold'],
            channels=channels,
            flow_threshold=params['flow_threshold']
        )
        
        eval_time = time.time() - start_eval_time
        update_progress(80, f"AI 分割完成！耗时 {eval_time:.2f} 秒。")
        
        # 3. 统计分析
        update_progress(85, "正在进行形态学统计...")
        
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

            if ecc > params['ecc_threshold'] and prop.area >= params['area_min']:
                polarized_cells += 1

        avg_ecc = np.mean(eccentricities) if valid_cells > 0 else 0
        avg_sol = np.mean(solidities) if valid_cells > 0 else 0
        polarization_percentage = (polarized_cells / valid_cells) * 100 if valid_cells > 0 else 0

        # 4. 创建可视化结果图
        update_progress(95, "正在生成可视化结果...")
        
        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(plot.mask_rgb(masks))
        ax1.set_title(f'1. Final Segmentation (n={valid_cells})', fontweight='bold')
        ax1.axis('off')
        # ... (省略 ax2, ax3 的绘图代码，与之前一致) ...
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(flows[0])
        ax2.set_title("2. Vector Flows (流场)", fontweight='bold')
        ax2.axis('off')

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(flows[2], cmap='inferno')
        ax3.set_title(f"3. Probability Heatmap", fontweight='bold')
        ax3.axis('off')
        
        plt.tight_layout()
        
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        png_data = buf.getvalue()
        plt.close(fig)
        
        update_progress(100, f"分析完成！总计算耗时: {eval_time:.2f} 秒。")

        return {
            'File_Name': file_name,
            'Cell_Count': valid_cells,
            'Eccentricity_Mean': avg_ecc,
            'Solidity_Mean': avg_sol,
            'Polarization_Percent': polarization_percentage,
            'Visualization_PNG': png_data,
            'Eval_Time': eval_time # 返回计算时间
        }

    except Exception as e:
        status_text_container.text(f"❌ 分析失败: {e}")
        st.error(f"❌ 运行 {file_name} 分析时发生严重错误: {e}")
        return None

# --- 3. Streamlit 主界面 ---
def main():
    st.title("🔬 Cellpose 驱动的细胞形态学分析 Web 工具")
    st.markdown("上传您的明场细胞图像 (`.tif`, `.jpg`)，并使用左侧参数调整分割效果。")
    
    # NEW: 隐私声明
    st.warning("🔒 **隐私保障声明:** 我们不会在服务器上保存您上传的任何图片和分析结果。所有数据都在内存中处理，页面关闭或刷新后即刻清除。")
    st.markdown("---")

    # 1. 加载模型
    model_obj, device = load_cellpose_model()
    if model_obj is None:
        st.error("无法加载 Cellpose 模型。")
        return

    # 2. 侧边栏：参数调整区域 (保持不变)
    st.sidebar.header("⚙️ 分割参数调整 (Cellpose)")
    max_size = st.sidebar.slider(
        '图像最大边长限制 (px)', min_value=512, max_value=2048, value=1024, step=256,
        help="限制图像的最大处理尺寸，以加快 Streamlit Cloud 上的 CPU 分析速度。减小此值可显著提速，但会损失细节。"
    )
    cell_diameter = st.sidebar.slider('预估细胞直径 (px)', min_value=10, max_value=200, value=24, step=1)
    prob_threshold = st.sidebar.slider('细胞概率阈值', min_value=-6.0, max_value=6.0, value=-2.0, step=0.1)
    flow_threshold = st.sidebar.slider('流场阈值', min_value=0.0, max_value=1.0, value=0.4, step=0.05)

    st.sidebar.header("📐 形态学过滤参数")
    ecc_threshold = st.sidebar.slider('极化伸长度阈值 (Eccentricity)', min_value=0.50, max_value=0.95, value=0.70, step=0.01)
    area_min = st.sidebar.slider('极化细胞最小面积 (px)', min_value=50, max_value=1000, value=200, step=10)

    analysis_params = {
        'max_size': max_size,
        'diameter': cell_diameter,
        'prob_threshold': prob_threshold,
        'flow_threshold': flow_threshold,
        'ecc_threshold': ecc_threshold,
        'area_min': area_min
    }

    # 4. 文件上传
    st.sidebar.header("📁 文件上传")
    uploaded_files = st.sidebar.file_uploader(
        "请选择一个或多个图像文件 (TIF, JPG)", 
        type=["tif", "tiff", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.subheader(f"共上传 {len(uploaded_files)} 张图片，请查看下方结果：")
        all_results_list = []
        
        # 顶部的总进度条 (用于批量文件进度)
        total_progress_bar = st.progress(0, text="批量处理进度：0%")
        
        for i, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            
            with st.container():
                st.markdown(f"#### 🔍 正在处理：{file_name}")
                
                # 为单张图片创建进度条和状态文本容器
                single_progress_bar = st.progress(0)
                status_text_container = st.empty()
                
                results = run_cellpose_analysis_optimized(uploaded_file, model_obj, analysis_params, single_progress_bar, status_text_container)
                
                # 隐藏单张图片的进度条和状态文本
                single_progress_bar.empty()
                status_text_container.empty()
                
                if results:
                    st.success(f"✅ 分析完成！计算耗时: {results['Eval_Time']:.2f} 秒")
                    all_results_list.append(results)
                    
                    # ... (结果展示代码保持不变) ...
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
            
            # 更新总进度条
            total_progress = (i + 1) / len(uploaded_files)
            total_progress_bar.progress(total_progress, text=f"批量处理进度：{i + 1}/{len(uploaded_files)} 文件已完成")

        # 5. 汇总结果 (所有文件分析完成后)
        if all_results_list:
            df_data = [{k: v for k, v in res.items() if k not in ['Visualization_PNG', 'Eval_Time']} for res in all_results_list]
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
            total_progress_bar.empty() # 清除总进度条
            st.success("🎉 所有文件分析完成，数据和图片下载链接已生成。")


if __name__ == "__main__":
    main()
