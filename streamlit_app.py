import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from reliability.Fitters import Fit_Weibull_2P, Fit_Lognormal_2P, Fit_Exponential_1P
from lifelines import KaplanMeierFitter
import io

# Page Config
st.set_page_config(
    page_title="Failure Rate Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Fira+Code:wght@400;500&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0d1117;
        color: #f3f4f6;
    }
    /* Style Streamlit Download Button */
    div.stDownloadButton button, [data-testid="stBaseButton-secondary"] {
        color: #000000 !important;
        background-color: #10b981 !important;
        border: 1px solid #10b981 !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stDownloadButton button:hover, [data-testid="stBaseButton-secondary"]:hover {
        background-color: #059669 !important;
        border-color: #059669 !important;
        color: #000000 !important;
        transform: translateY(-1px) !important;
    }
    div.stDownloadButton button:active, [data-testid="stBaseButton-secondary"]:active {
        background-color: #047857 !important;
        border-color: #047857 !important;
        color: #000000 !important;
    }
    .main-title {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(to right, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #9ca3af;
        text-align: center;
        font-weight: 300;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stCodeBlock {
        background-color: #030712 !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Failure Rate Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Weibull, Log-Normal, Exponential, Kaplan-Meier Reliability Analysis</div>', unsafe_allow_html=True)

# Helper function for KM lookup
def get_km_value(target_time, times, values):
    if target_time < times[0]:
        return 0.0
    elif target_time > times[-1]:
        return values[-1] 
    else:
        idx = np.searchsorted(times, target_time, side='right') - 1
        return values[idx]

# Sidebar for file uploaders and method selection
with st.sidebar:
    st.header("⚙️ Data Import & Controls")
    
    file_cdf = st.file_uploader(
        "1. 누적고장률 데이터 CSV (01_Eva_Leak_CDF_Rate.csv)", 
        type=["csv"],
        help="Upload the field cumulative failure rate data file."
    )
    
    file_pdf = st.file_uploader(
        "2. 사용개월별 고장/생존수량 CSV (02_Eva_Leak_PDF_Predict_Data.csv)", 
        type=["csv"],
        help="Upload the usage month failure and survival quantity data file."
    )
    
    st.markdown("---")
    
    method = st.selectbox(
        "3. 분석 방법 선택 (Analysis Method)",
        ["Weibull", "Log-Normal", "Exponential", "Kaplan-Meier"],
        help="Select the reliability fitting distribution model."
    )

if file_cdf is not None and file_pdf is not None:
    try:
        # Preprocess Data
        df_cdf = pd.read_csv(file_cdf)
        df_pdf = pd.read_csv(file_pdf)
        
        # 1. Parse CDF Field Data
        x_time = []
        y_ratio = []
        df_cdf_temp = df_cdf.copy()
        df_cdf_temp.set_index('PROD_M', inplace=True)
        for use in df_cdf_temp.index:
            for prod in df_cdf_temp.columns:
                val = df_cdf_temp.loc[use, prod]
                if pd.notna(val) and val != 0:
                    x_time.append(float(use))
                    y_ratio.append(float(val))
                    
        # 2. Parse PDF Unpacked Data
        df_pdf['End'] = df_pdf['End'].replace(['', ' ', None], np.nan)
        df_pdf['Start'] = pd.to_numeric(df_pdf['Start'], errors='coerce')
        df_pdf['End'] = pd.to_numeric(df_pdf['End'], errors='coerce')
        df_pdf['Fail_Count'] = pd.to_numeric(df_pdf['Fail_Count'], errors='coerce').fillna(0).astype(int)
        
        df_fails = df_pdf[(df_pdf['Fail_Count'] > 0) & (df_pdf['End'].notna())].copy()
        df_right = df_pdf[(df_pdf['Fail_Count'] > 0) & (df_pdf['End'].isna())].copy()
        
        fail_starts = df_fails['Start'].values
        fail_ends = df_fails['End'].values
        fail_counts = df_fails['Fail_Count'].values
        midpoints = (fail_starts + fail_ends) / 2.0
        failures_unpacked = np.repeat(midpoints, fail_counts)
        
        right_starts = df_right['Start'].values
        right_counts = df_right['Fail_Count'].values
        right_censored_final = np.repeat(right_starts, right_counts)
        
        timeline = np.linspace(1, 240, 240)
        num_failures = len(failures_unpacked)
        num_censored = len(right_censored_final)
        
        summary_text = ""
        F_t = None
        F_lower = None
        F_upper = None
        km_time = None
        
        # --- Fitting Models ---
        if method == 'Weibull':
            fit = Fit_Weibull_2P(
                failures=failures_unpacked,
                right_censored=right_censored_final,
                show_probability_plot=False,
                CI=0.95
            )
            shape_param = fit.beta
            scale_param = fit.alpha
            beta_lower = fit.beta_lower
            beta_upper = fit.beta_upper
            alpha_lower = fit.alpha_lower
            alpha_upper = fit.alpha_upper
            
            F_t = (1 - np.exp(- (timeline / scale_param) ** shape_param)) * 100
            F_lower = (1 - np.exp(- (timeline / alpha_lower) ** beta_upper)) * 100
            F_upper = (1 - np.exp(- (timeline / alpha_upper) ** beta_lower)) * 100
            
            summary_text = (
                f"--------------------------------------------------\n"
                f"   최종 분석 결과 (Weibull)\n"
                f"--------------------------------------------------\n"
                f" - 고장 데이터: {num_failures} 개\n"
                f" - 관측 중단 데이터: {num_censored} 개\n"
                f"-------------------------------------------------\n"
                f"형상모수 (Shape, β) : {shape_param:.4f}\n"
                f"척도모수 (Scale, η) : {scale_param:.4f}\n"
                f"--------------------------------------------------\n"
                f"형상모수 95% CI    : {beta_lower:.4f} ~ {beta_upper:.4f}\n"
                f"척도모수 95% CI    : {alpha_lower:.4f} ~ {alpha_upper:.4f}\n"
                f"--------------------------------------------------\n"
                f"▶ 12개월(1년) 누적고장률: {F_t[11]:.4f} %\n"
                f"▶ 60개월(5년) 누적고장률: {F_t[59]:.4f} %\n"
                f"▶ 120개월(10년) 누적고장률: {F_t[119]:.4f} %\n"
                f"▶ 240개월(20년) 누적고장률: {F_t[239]:.4f} %"
            )
            
        elif method == 'Log-Normal':
            failures_ln = failures_unpacked[failures_unpacked > 0]
            right_ln = right_censored_final[right_censored_final > 0]
            
            fit = Fit_Lognormal_2P(
                failures=failures_ln,
                right_censored=right_ln,
                show_probability_plot=False,
                CI=0.95
            )
            mu_param = fit.mu
            sigma_param = fit.sigma
            mu_lower = fit.mu_lower
            mu_upper = fit.mu_upper
            sigma_lower = fit.sigma_lower
            sigma_upper = fit.sigma_upper
            
            F_t = norm.cdf((np.log(timeline) - mu_param) / sigma_param) * 100
            F_lower = norm.cdf((np.log(timeline) - mu_upper) / sigma_upper) * 100
            F_upper = norm.cdf((np.log(timeline) - mu_lower) / sigma_lower) * 100
            
            summary_text = (
                f"--------------------------------------------------\n"
                f"   최종 분석 결과 (Log-Normal)\n"
                f"--------------------------------------------------\n"
                f" - 고장 데이터: {len(failures_ln)} 개\n"
                f" - 관측 중단 데이터: {len(right_ln)} 개\n"
                f"-------------------------------------------------\n"
                f"모수 1 (Mu, μ)    : {mu_param:.4f}\n"
                f"모수 2 (Sigma, σ) : {sigma_param:.4f}\n"
                f"--------------------------------------------------\n"
                f"▶ 12개월(1년) 누적고장률: {F_t[11]:.4f} %\n"
                f"▶ 60개월(5년) 누적고장률: {F_t[59]:.4f} %\n"
                f"▶ 120개월(10년) 누적고장률: {F_t[119]:.4f} %\n"
                f"▶ 240개월(20년) 누적고장률: {F_t[239]:.4f} %"
            )
            
        elif method == 'Exponential':
            fit = Fit_Exponential_1P(
                failures=failures_unpacked,
                right_censored=right_censored_final,
                show_probability_plot=False,
                CI=0.95
            )
            lambda_param = fit.Lambda
            mttf_param = 1 / lambda_param
            lambda_lower = fit.Lambda_lower
            lambda_upper = fit.Lambda_upper
            
            F_t = (1 - np.exp(- lambda_param * timeline)) * 100
            F_lower = (1 - np.exp(- lambda_lower * timeline)) * 100
            F_upper = (1 - np.exp(- lambda_upper * timeline)) * 100
            
            summary_text = (
                f"--------------------------------------------------\n"
                f"   최종 분석 결과 (Exponential)\n"
                f"--------------------------------------------------\n"
                f" - 고장 데이터: {num_failures} 개\n"
                f" - 관측 중단 데이터: {num_censored} 개\n"
                f"-------------------------------------------------\n"
                f"고장률 (Lambda, λ) : {lambda_param:.6f}\n"
                f"평균수명 (MTTF, 1/λ): {mttf_param:.4f}\n"
                f"--------------------------------------------------\n"
                f"고장률 95% CI      : {lambda_lower:.6f} ~ {lambda_upper:.6f}\n"
                f"--------------------------------------------------\n"
                f"▶ 12개월(1년) 누적고장률: {F_t[11]:.4f} %\n"
                f"▶ 60개월(5년) 누적고장률: {F_t[59]:.4f} %\n"
                f"▶ 120개월(10년) 누적고장률: {F_t[119]:.4f} %\n"
                f"▶ 240개월(20년) 누적고장률: {F_t[239]:.4f} %"
            )
            
        elif method == 'Kaplan-Meier':
            T = np.concatenate((failures_unpacked, right_censored_final))
            E = np.concatenate((np.ones(len(failures_unpacked)), np.zeros(len(right_censored_final))))
            
            kmf = KaplanMeierFitter()
            kmf.fit(T, event_observed=E, label='Kaplan-Meier Estimate')
            
            km_survival_df = kmf.survival_function_
            km_ci_df = kmf.confidence_interval_
            
            km_time = km_survival_df.index.values
            km_survival = km_survival_df['Kaplan-Meier Estimate'].values
            km_survival_lower = km_ci_df.iloc[:, 0].values
            km_survival_upper = km_ci_df.iloc[:, 1].values
            
            F_t = (1 - km_survival) * 100
            F_upper = (1 - km_survival_lower) * 100
            F_lower = (1 - km_survival_upper) * 100
            
            val_12 = get_km_value(12, km_time, F_t)
            val_60 = get_km_value(60, km_time, F_t)
            val_120 = get_km_value(120, km_time, F_t)
            val_240 = get_km_value(240, km_time, F_t)
            
            summary_text = (
                f"   Kaplan-Meier 분석 결과\n"
                f"--------------------------------------------------\n"
                f"▶ 12개월(1년) 누적고장률 추정치: {val_12:.4f} %\n"
                f"▶ 60개월(5년) 누적고장률 추정치: {val_60:.4f} %\n"
                f"▶ 120개월(10년) 누적고장률 추정치: {val_120:.4f} %\n"
                f"▶ 240개월(20년) 누적고장률 추정치: {val_240:.4f} %"
            )
            
        # --- Dual Plot Generation ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calculate Y-axis top limit dynamically
        if method == 'Kaplan-Meier':
            max_y_fitted = np.max(F_t[km_time <= 120])
            x_time_arr = np.array(x_time)
            y_ratio_arr = np.array(y_ratio)
            mask_scatter = (x_time_arr <= 120)
            max_y_scatter = np.max(y_ratio_arr[mask_scatter]) if len(y_ratio_arr[mask_scatter]) > 0 else 0
            top_limit = max(1.0, max(max_y_fitted, max_y_scatter) * 1.15)
            
            # Left Plot (Linear Scale)
            axes[0].step(km_time, F_t, where='post', label='Kaplan-Meier F(t)', color='blue', linewidth=2)
            axes[0].scatter(x_time, y_ratio, color='red', alpha=0.3, s=15, label='Field Failure Rate')
            axes[0].fill_between(km_time, F_lower, F_upper, step='post', color='blue', alpha=0.15, label='95% Confidence Interval')
            axes[0].set_title('Kaplan-Meier Failure Rate F(t) - Linear Scale', fontsize=12, fontweight='bold')
            axes[0].set_xlim(0, 120)
            axes[0].set_ylim(0, top_limit)
            
            # Right Plot (Log-Log Scale)
            mask = (km_time > 0) & (F_t > 0)
            axes[1].step(km_time[mask], F_t[mask], where='post', label='F(t)', color='red', linewidth=2)
            axes[1].fill_between(km_time[mask], F_lower[mask], F_upper[mask], step='post', color='red', alpha=0.15, label='95% CI')
            axes[1].set_title('Kaplan-Meier Failure Rate F(t) - Log-Log Scale', fontsize=12, fontweight='bold')
            axes[1].set_xscale('log')
            axes[1].set_yscale('log')
            axes[1].set_xlim(1, 120)
            
        else:
            max_y_fitted = np.max(F_t[timeline <= 120])
            x_time_arr = np.array(x_time)
            y_ratio_arr = np.array(y_ratio)
            mask_scatter = (x_time_arr <= 120)
            max_y_scatter = np.max(y_ratio_arr[mask_scatter]) if len(y_ratio_arr[mask_scatter]) > 0 else 0
            top_limit = max(1.0, max(max_y_fitted, max_y_scatter) * 1.15)
            
            # Left Plot (Linear Scale)
            label_fit = f'F(t)'
            if method == 'Weibull':
                label_fit = f'F(t) (β={fit.beta:.2f}, η={fit.alpha:.1f})'
            elif method == 'Log-Normal':
                label_fit = f'F(t) (μ={fit.mu:.2f}, σ={fit.sigma:.2f})'
            elif method == 'Exponential':
                label_fit = f'F(t) (λ={fit.Lambda:.5f})'
                
            axes[0].plot(timeline, F_t, label=label_fit, color='blue', linewidth=2)
            axes[0].scatter(x_time, y_ratio, color='red', alpha=0.2, s=10, label='Field Failure Rate')
            axes[0].fill_between(timeline, F_lower, F_upper, color='blue', alpha=0.15, label='95% Confidence Interval')
            axes[0].set_title(f'{method} Failure Rate F(t) - Linear Scale', fontsize=12, fontweight='bold')
            axes[0].set_xlim(0, 120)
            axes[0].set_ylim(0, top_limit)
            
            # Right Plot (Log-Log Scale)
            axes[1].plot(timeline, F_t, label='F(t)', color='red', linewidth=2)
            axes[1].fill_between(timeline, F_lower, F_upper, color='red', alpha=0.15, label='95% CI')
            axes[1].set_title(f'{method} Failure Rate F(t) - Log-Log Scale', fontsize=12, fontweight='bold')
            axes[1].set_xscale('log')
            axes[1].set_yscale('log')
            axes[1].set_xlim(1, 120)
            
        # Common configurations for axes[0] and axes[1]
        for ax in axes:
            ax.set_xlabel('Service Month (t)', fontsize=10)
            ax.set_ylabel('Cumulative Failure Rate F(t) %', fontsize=10)
            ax.legend(fontsize=9, loc='upper left')
            
        axes[0].set_xticks(np.arange(0, 121, 10))
        axes[0].grid(True, which='major', linestyle='-', alpha=0.7)
        axes[1].grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        
        # Display Graphs
        st.pyplot(fig)
        
        # --- Save Prediction CSV Segment ---
        st.markdown("### 💾 분석 결과 데이터 저장 (CSV Export)")
        
        # Prepare CSV Contents
        times_list = timeline if method != 'Kaplan-Meier' else km_time
        csv_df = pd.DataFrame({
            'Time(Month)': times_list,
            'Failure_Rate(%)': F_t,
            'Lower_CI(%)': F_lower,
            'Upper_CI(%)': F_upper
        })
        
        # Generate CSV to byte buffer
        csv_buffer = io.StringIO()
        csv_df.to_csv(csv_buffer, index=False)
        csv_string = "\uFEFF" + csv_buffer.getvalue() # Add BOM for Excel Korean support
        
        # Input filename and download button
        filename_input = st.text_input(
            "저장할 파일명 입력 (.csv)", 
            value=f"{method}_Predict_Results.csv",
            help="Type your desired filename."
        )
        
        st.download_button(
            label="💾 CSV 파일로 저장",
            data=csv_string,
            file_name=filename_input,
            mime="text/csv",
            help="Click to open save dialog and download."
        )
        
        # --- Console Output Segment ---
        st.markdown("### 📄 분석 결과 상세 로그 (Console Log)")
        st.code(summary_text, language="text")
        
    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
else:
    st.info("👈 왼쪽 사이드바에서 두 개의 CSV 데이터 파일을 업로드하여 분석을 시작하세요.")
