# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 16:54:25 2025

@author: 27862
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Callable, Tuple, List
import time

# 设置页面配置
st.set_page_config(
    page_title="自相关函数模拟器",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-header">自相关函数模拟器</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">模拟随机过程并计算其自相关函数 | 支持自定义随机过程函数</p>', unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 模拟参数")
    
    # 随机过程选择
    process_type = st.selectbox(
        "选择随机过程类型",
        ["布朗运动", "几何布朗运动", "OU过程", "自定义过程"],
        help="选择要模拟的随机过程类型"
    )
    
    # 基本参数
    col1, col2 = st.columns(2)
    with col1:
        sigma = st.slider("波动率 (σ)", 0.1, 2.0, 1.0, 0.1)
    with col2:
        T = st.slider("时间范围 (T)", 0.1, 5.0, 1.0, 0.1)
    
    # 模拟参数
    n_paths = st.slider("模拟路径数", 10, 10000, 1000, 10)
    n_steps = st.slider("时间步数", 10, 1000, 100, 10)
    dt = T / n_steps
    
    # 随机种子
    use_seed = st.checkbox("使用随机种子")
    if use_seed:
        seed = st.number_input("随机种子", 0, 10000, 42)
        np.random.seed(seed)
    
    # 自相关函数参数
    st.header("📊 自相关函数设置")
    grid_size = st.slider("网格分辨率", 10, 100, 30, 5)
    
    # 自定义过程参数
    if process_type == "自定义过程":
        st.header("✏️ 自定义过程定义")
        custom_code = st.text_area(
            "输入自定义过程函数 (使用t, sigma, dt, n_steps参数)",
            '''def custom_process(t, sigma, dt, n_steps):
    # 自定义随机过程实现
    # 返回: 时间数组和时间序列数组
    times = np.linspace(0, t, n_steps)
    # 示例: 带漂移的布朗运动
    mu = 0.1  # 漂移率
    dW = np.random.normal(0, np.sqrt(dt), n_steps-1)
    process = np.zeros(n_steps)
    for i in range(1, n_steps):
        process[i] = process[i-1] + mu*dt + sigma*dW[i-1]
    return times, process''',
            height=200
        )
    
    st.header("🎨 可视化设置")
    col1, col2 = st.columns(2)
    with col1:
        color_scheme = st.selectbox("颜色方案", ["Viridis", "Plasma", "Rainbow", "Jet"])
    with col2:
        opacity = st.slider("曲面透明度", 0.1, 1.0, 0.8, 0.1)

# 随机过程函数定义
def brownian_motion(t: float, sigma: float, dt: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """标准布朗运动"""
    times = np.linspace(0, t, n_steps)
    dW = np.random.normal(0, np.sqrt(dt), n_steps-1)
    W = np.zeros(n_steps)
    W[1:] = np.cumsum(dW)
    return times, W

def geometric_brownian_motion(t: float, sigma: float, dt: float, n_steps: int, 
                              mu: float = 0.1, S0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """几何布朗运动"""
    times = np.linspace(0, t, n_steps)
    dW = np.random.normal(0, np.sqrt(dt), n_steps-1)
    S = np.zeros(n_steps)
    S[0] = S0
    for i in range(1, n_steps):
        S[i] = S[i-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW[i-1])
    return times, S

def ou_process(t: float, sigma: float, dt: float, n_steps: int, 
               theta: float = 1.0, mu: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Ornstein-Uhlenbeck过程"""
    times = np.linspace(0, t, n_steps)
    dW = np.random.normal(0, np.sqrt(dt), n_steps-1)
    X = np.zeros(n_steps)
    X[0] = 0.0
    for i in range(1, n_steps):
        X[i] = X[i-1] + theta*(mu - X[i-1])*dt + sigma*dW[i-1]
    return times, X

# 模拟函数
def simulate_process(process_type: str, t: float, sigma: float, dt: float, 
                     n_steps: int, n_paths: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """模拟多条路径"""
    paths = []
    for _ in range(n_paths):
        if process_type == "布朗运动":
            times, values = brownian_motion(t, sigma, dt, n_steps)
        elif process_type == "几何布朗运动":
            times, values = geometric_brownian_motion(t, sigma, dt, n_steps)
        elif process_type == "OU过程":
            times, values = ou_process(t, sigma, dt, n_steps)
        elif process_type == "自定义过程":
            # 动态执行自定义代码
            try:
                exec(custom_code, globals())
                times, values = custom_process(t, sigma, dt, n_steps)
            except Exception as e:
                st.error(f"自定义过程错误: {e}")
                # 回退到布朗运动
                times, values = brownian_motion(t, sigma, dt, n_steps)
        else:
            times, values = brownian_motion(t, sigma, dt, n_steps)
        paths.append((times, values))
    return paths

# 计算自相关函数
def compute_autocorrelation(paths: List[Tuple[np.ndarray, np.ndarray]], 
                           grid_size: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算自相关函数R(s,t)"""
    # 获取时间网格
    times = paths[0][0]
    t_max = times[-1]
    s_grid = np.linspace(0, t_max, grid_size)
    t_grid = np.linspace(0, t_max, grid_size)
    
    # 创建网格
    S, T = np.meshgrid(s_grid, t_grid)
    R = np.zeros((grid_size, grid_size))
    
    # 为每个网格点计算自相关
    for i in range(grid_size):
        for j in range(grid_size):
            s_val = s_grid[i]
            t_val = t_grid[j]
            
            # 找到最接近的时间索引
            s_idx = np.argmin(np.abs(times - s_val))
            t_idx = np.argmin(np.abs(times - t_val))
            
            # 计算所有路径在该时间点的自相关
            autocorrs = []
            for times_arr, values in paths:
                if s_idx < len(values) and t_idx < len(values):
                    autocorrs.append(values[s_idx] * values[t_idx])
            
            R[i, j] = np.mean(autocorrs) if autocorrs else 0
    
    return S, T, R

# 计算理论自相关函数（布朗运动）
def theoretical_autocorrelation(S: np.ndarray, T: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """布朗运动的理论自相关函数: R(s,t) = sigma^2 * min(s,t)"""
    return sigma**2 * np.minimum(S, T)

# 主应用
def main():
    # 创建两列布局
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("📈 随机过程模拟")
        
        # 模拟按钮
        if st.button("🚀 开始模拟", use_container_width=True):
            with st.spinner("正在模拟..."):
                start_time = time.time()
                
                # 模拟路径
                paths = simulate_process(process_type, T, sigma, dt, n_steps, n_paths)
                
                # 计算自相关函数
                S, T_grid, R = compute_autocorrelation(paths, grid_size)
                
                # 计算理论值（如果是布朗运动）
                if process_type == "布朗运动":
                    R_theoretical = theoretical_autocorrelation(S, T_grid, sigma)
                
                end_time = time.time()
                st.success(f"模拟完成！耗时 {end_time-start_time:.2f} 秒")
                
                # 存储到session state
                st.session_state.paths = paths
                st.session_state.S = S
                st.session_state.T = T_grid
                st.session_state.R = R
                if process_type == "布朗运动":
                    st.session_state.R_theoretical = R_theoretical
                st.session_state.process_type = process_type
    
    with col2:
        st.header("📊 统计信息")
        
        if 'paths' in st.session_state:
            # 计算基本统计量
            all_values = np.concatenate([values for _, values in st.session_state.paths])
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            min_val = np.min(all_values)
            max_val = np.max(all_values)
            
            # 显示统计卡片
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("均值", f"{mean_val:.4f}")
                st.metric("标准差", f"{std_val:.4f}")
            with col_b:
                st.metric("最小值", f"{min_val:.4f}")
                st.metric("最大值", f"{max_val:.4f}")
            
            # 路径数量信息
            st.info(f"模拟路径数: {len(st.session_state.paths)}")
    
    # 显示结果
    if 'paths' in st.session_state:
        # 创建标签页
        tab1, tab2, tab3 = st.tabs(["路径可视化", "自相关函数3D图", "自相关函数热图"])
        
        with tab1:
            st.subheader(f"{st.session_state.process_type} 模拟路径")
            
            # 选择要显示的路径数量
            max_show = min(20, len(st.session_state.paths))
            show_paths = st.slider("显示路径数", 1, max_show, min(5, max_show))
            
            # 创建路径图
            fig_paths = go.Figure()
            
            # 添加路径
            for i, (times, values) in enumerate(st.session_state.paths[:show_paths]):
                fig_paths.add_trace(go.Scatter(
                    x=times,
                    y=values,
                    mode='lines',
                    name=f'路径 {i+1}',
                    line=dict(width=1 if show_paths > 10 else 2),
                    opacity=0.7 if show_paths > 5 else 0.9
                ))
            
            # 添加均值路径
            if len(st.session_state.paths) > 1:
                all_times = st.session_state.paths[0][0]  # 假设所有路径时间点相同
                mean_path = np.mean([values for _, values in st.session_state.paths], axis=0)
                fig_paths.add_trace(go.Scatter(
                    x=all_times,
                    y=mean_path,
                    mode='lines',
                    name='均值路径',
                    line=dict(color='black', width=3, dash='dash')
                ))
            
            # 更新布局
            fig_paths.update_layout(
                title=f"{st.session_state.process_type} 模拟路径 (显示 {show_paths} 条)",
                xaxis_title="时间",
                yaxis_title="值",
                hovermode='x unified',
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_paths, use_container_width=True)
        
        with tab2:
            st.subheader("自相关函数 3D 可视化")
            
            # 创建3D图
            fig_3d = go.Figure(data=[
                go.Surface(
                    z=st.session_state.R,
                    x=st.session_state.S[0, :],
                    y=st.session_state.T[:, 0],
                    colorscale=color_scheme.lower(),
                    opacity=opacity,
                    name='模拟自相关'
                )
            ])
            
            # 如果是布朗运动，添加理论曲面
            if st.session_state.process_type == "布朗运动" and 'R_theoretical' in st.session_state:
                fig_3d.add_trace(go.Surface(
                    z=st.session_state.R_theoretical,
                    x=st.session_state.S[0, :],
                    y=st.session_state.T[:, 0],
                    colorscale='Greys',
                    opacity=0.3,
                    showscale=False,
                    name='理论自相关'
                ))
            
            # 更新3D图布局
            fig_3d.update_layout(
                title=f"{st.session_state.process_type} 自相关函数 R(s,t)",
                scene=dict(
                    xaxis_title="s",
                    yaxis_title="t",
                    zaxis_title="R(s,t)",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2)
                    )
                ),
                height=600,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with tab3:
            st.subheader("自相关函数热图")
            
            # 创建热图
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=st.session_state.R,
                x=st.session_state.S[0, :],
                y=st.session_state.T[:, 0],
                colorscale=color_scheme.lower(),
                colorbar=dict(title="R(s,t)")
            ))
            
            # 更新热图布局
            fig_heatmap.update_layout(
                title=f"{st.session_state.process_type} 自相关函数热图",
                xaxis_title="s",
                yaxis_title="t",
                height=500,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # 添加理论值对比（如果是布朗运动）
            if st.session_state.process_type == "布朗运动" and 'R_theoretical' in st.session_state:
                st.subheader("模拟 vs 理论 对比")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    # 计算误差
                    error = np.abs(st.session_state.R - st.session_state.R_theoretical)
                    mean_error = np.mean(error)
                    max_error = np.max(error)
                    
                    st.metric("平均绝对误差", f"{mean_error:.6f}")
                    st.metric("最大绝对误差", f"{max_error:.6f}")
                
                with col_b:
                    # 误差热图
                    fig_error = go.Figure(data=go.Heatmap(
                        z=error,
                        x=st.session_state.S[0, :],
                        y=st.session_state.T[:, 0],
                        colorscale='Reds',
                        colorbar=dict(title="绝对误差")
                    ))
                    
                    fig_error.update_layout(
                        title="模拟与理论值的绝对误差",
                        xaxis_title="s",
                        yaxis_title="t",
                        height=300
                    )
                    
                    st.plotly_chart(fig_error, use_container_width=True)
        
        # 下载数据选项
        st.divider()
        st.subheader("📥 数据导出")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("导出自相关函数数据"):
                # 创建DataFrame
                df_data = {
                    's': st.session_state.S.flatten(),
                    't': st.session_state.T.flatten(),
                    'R_simulated': st.session_state.R.flatten()
                }
                
                if 'R_theoretical' in st.session_state:
                    df_data['R_theoretical'] = st.session_state.R_theoretical.flatten()
                
                df = pd.DataFrame(df_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="下载CSV",
                    data=csv,
                    file_name=f"autocorrelation_{st.session_state.process_type}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("导出模拟路径数据"):
                # 导出第一条路径作为示例
                times, values = st.session_state.paths[0]
                df_path = pd.DataFrame({
                    'time': times,
                    'value': values
                })
                csv_path = df_path.to_csv(index=False)
                
                st.download_button(
                    label="下载路径数据",
                    data=csv_path,
                    file_name=f"path_{st.session_state.process_type}.csv",
                    mime="text/csv"
                )
    
    else:
        # 初始状态显示说明
        st.info("👈 请在左侧配置模拟参数，然后点击'开始模拟'按钮")
        
        # 显示示例图
        st.subheader("示例展示")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Wiener_process_animated.gif/400px-Wiener_process_animated.gif", 
                    caption="布朗运动示例", use_column_width=True)
        
        with col2:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/3d_heatmap.png/400px-3d_heatmap.png", 
                    caption="自相关函数3D图示例", use_column_width=True)
        
        # 功能说明
        st.subheader("功能说明")
        st.markdown("""
        1. **布朗运动**: 标准维纳过程，用于模拟随机游走
        2. **几何布朗运动**: 常用于金融资产价格建模
        3. **OU过程**: Ornstein-Uhlenbeck过程，均值回复过程
        4. **自定义过程**: 支持用户自定义随机过程函数
        
        自相关函数 R(s,t) 表示随机过程在时间 s 和 t 的值之间的相关性。
        对于布朗运动，理论自相关函数为 R(s,t) = σ² × min(s,t)。
        """)

# 运行主应用
if __name__ == "__main__":
    main()
