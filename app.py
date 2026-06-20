import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to run headless
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from scipy.stats import norm
from reliability.Fitters import Fit_Weibull_2P, Fit_Lognormal_2P, Fit_Exponential_1P
from lifelines import KaplanMeierFitter

# 웹 브라우저 주소창에 직접 http://127.0.0.1:5000/ 입력하여 접속

app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_data(cdf_path, pdf_path):
    # 1. Read files
    df_cdf = pd.read_csv(cdf_path)
    df_pdf = pd.read_csv(pdf_path)
    
    # 2. Extract x_time, y_ratio from CDF
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
                
    # 3. Unpack failures and censored data from PDF
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
    
    return x_time, y_ratio, failures_unpacked, right_censored_final

def get_km_value(target_time, times, values):
    if target_time < times[0]:
        return 0.0
    elif target_time > times[-1]:
        return values[-1] 
    else:
        idx = np.searchsorted(times, target_time, side='right') - 1
        return values[idx]

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check files in request
        if 'file_cdf' not in request.files or 'file_pdf' not in request.files:
            return jsonify({'status': 'error', 'message': 'Both CSV files must be selected.'})
            
        file_cdf = request.files['file_cdf']
        file_pdf = request.files['file_pdf']
        method = request.form.get('method', 'Weibull')
        
        if file_cdf.filename == '' or file_pdf.filename == '':
            return jsonify({'status': 'error', 'message': 'Invalid file selection.'})
            
        # Save temp files
        cdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_cdf.csv')
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_pdf.csv')
        file_cdf.save(cdf_path)
        file_pdf.save(pdf_path)
        
        # Process files
        x_time, y_ratio, failures_unpacked, right_censored_final = process_data(cdf_path, pdf_path)
        
        # Timeline for models (1 to 240 months)
        timeline = np.linspace(1, 240, 240)
        
        summary_text = ""
        plot_base64 = ""
        
        # Variables for plotting
        F_t = None
        F_lower = None
        F_upper = None
        km_time = None
        
        num_failures = len(failures_unpacked)
        num_censored = len(right_censored_final)
        
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
            # Remove 0s for Log-Normal compliance
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
            
        else:
            return jsonify({'status': 'error', 'message': 'Unknown analysis method.'})

        # --- Matplotlib Dual Plot Generation ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Determine top limit for autoscale on Y-axis
        if method == 'Kaplan-Meier':
            max_y_fitted = np.max(F_t[km_time <= 120])
            # Filter scatter points within 120 months
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
        
        # Set major grid and ticks for linear plot
        axes[0].set_xticks(np.arange(0, 121, 10))
        axes[0].grid(True, which='major', linestyle='-', alpha=0.7)
        axes[1].grid(True, which="both", ls="-", alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Cleanup temp files
        try:
            os.remove(cdf_path)
            os.remove(pdf_path)
        except Exception:
            pass
            
        return jsonify({
            'status': 'success',
            'summary': summary_text,
            'plot': plot_base64,
            'analysis_data': {
                'time': timeline.tolist() if method != 'Kaplan-Meier' else km_time.tolist(),
                'failure_rate': F_t.tolist(),
                'lower_ci': F_lower.tolist() if F_lower is not None else [],
                'upper_ci': F_upper.tolist() if F_upper is not None else []
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Analysis error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
