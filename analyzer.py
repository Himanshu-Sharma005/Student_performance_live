"""
Data analysis, preprocessing, and visualization utilities for Student Performance Analyzer.
"""
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns


def load_csv_data(filepath_or_buffer):
    """
    Load a CSV file with automatic encoding fallback and validation.
    Returns (DataFrame, error_message).
    """
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    df = None
    last_err = None

    for enc in encodings:
        try:
            if hasattr(filepath_or_buffer, 'seek'):
                filepath_or_buffer.seek(0)
            df = pd.read_csv(filepath_or_buffer, encoding=enc)
            break
        except pd.errors.EmptyDataError:
            return None, "The uploaded CSV file is empty (contains no data)."
        except Exception as e:
            last_err = str(e)

    if df is None:
        return None, f"Failed to parse CSV file: {last_err}"

    if df.empty:
        return None, "The uploaded CSV file is empty (0 rows)."

    if len(df.columns) == 0:
        return None, "The uploaded CSV file has no columns."

    # Strip whitespace from column names for cleanliness
    df.columns = [str(c).strip() for c in df.columns]
    return df, None


def detect_columns(df):
    """
    Intelligently auto-detect column mappings for student datasets.
    """
    cols = list(df.columns)
    cols_lower = [c.lower() for c in cols]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    student_col = None
    score_col = None
    attendance_col = None
    subject_cols = []

    # 1. Detect student identifier / name column
    name_keywords = ['name', 'student_name', 'studentname', 'full_name', 'fullname', 
                     'student_id', 'studentid', 'student', 'roll_no', 'rollno', 'id']
    for kw in name_keywords:
        for orig, low in zip(cols, cols_lower):
            if kw == low or (kw in low and not any(k in low for k in ['score', 'marks', 'grade'])):
                student_col = orig
                break
        if student_col:
            break
    if not student_col and categorical_cols:
        student_col = categorical_cols[0]

    # 2. Detect attendance column
    att_keywords = ['attendance_percentage', 'attendance_percent', 'attendance', 'att_pct', 'att', 
                    'present_pct', 'present_days', 'presence', 'present', 'attendance_rate']
    for kw in att_keywords:
        for orig, low in zip(cols, cols_lower):
            if kw in low and orig in numeric_cols:
                attendance_col = orig
                break
        if attendance_col:
            break

    # 3. Detect main score column
    # Prioritize overall/aggregate score columns
    overall_score_keywords = ['avg_score', 'average_score', 'avgscore', 'total_score', 'totalscore', 
                              'final_score', 'overall_score', 'percentage', 'gpa', 'final_grade', 'aggregate']
    for kw in overall_score_keywords:
        for orig, low in zip(cols, cols_lower):
            if kw in low and orig in numeric_cols and orig != attendance_col:
                score_col = orig
                break
        if score_col:
            break

    # 4. Detect subject columns
    subject_keywords = ['math', 'science', 'english', 'history', 'physics', 'chemistry', 
                        'biology', 'computer', 'social', 'art', 'geography', 'hindi', 'music',
                        'literature', 'algebra', 'calculus', 'geometry', 'economics']
    for orig, low in zip(cols, cols_lower):
        if orig in numeric_cols and orig not in [student_col, attendance_col, score_col]:
            if any(sk in low for sk in subject_keywords) or 'score' in low or 'mark' in low:
                subject_cols.append(orig)

    # If score_col wasn't explicitly found, but generic 'score' or 'marks' exists and isn't in subjects:
    if not score_col:
        for orig, low in zip(cols, cols_lower):
            if orig in numeric_cols and orig not in [student_col, attendance_col] and orig not in subject_cols:
                if 'score' in low or 'mark' in low or 'total' in low:
                    score_col = orig
                    break

    # If no subjects identified by keyword, collect other numerical columns that look like score metrics (0-100 range)
    if not subject_cols and numeric_cols:
        for c in numeric_cols:
            if c not in [student_col, attendance_col, score_col]:
                # Check value range if possible
                col_max = df[c].max()
                col_min = df[c].min()
                if pd.notna(col_max) and col_max <= 100 and pd.notna(col_min) and col_min >= 0:
                    subject_cols.append(c)

    # If score_col is still None but subject_cols exist, score can be computed as average of subjects
    return {
        'student_col': student_col,
        'score_col': score_col,
        'attendance_col': attendance_col,
        'subject_cols': subject_cols,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }


def preprocess_data(df, remove_duplicates=False, missing_strategy='none', custom_numeric_cols=None):
    """
    Pipeline to clean and preprocess DataFrame.
    Returns: (cleaned_df, summary_dict, applied_steps_list)
    """
    initial_rows, initial_cols = df.shape
    initial_missing_total = int(df.isna().sum().sum())
    initial_missing_by_col = df.isna().sum().to_dict()
    initial_duplicates = int(df.duplicated().sum())

    cleaned_df = df.copy()
    steps_applied = []

    # 1. Drop completely empty rows and columns
    empty_rows_before = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(how='all')
    empty_rows_dropped = empty_rows_before - len(cleaned_df)
    if empty_rows_dropped > 0:
        steps_applied.append(f"Removed {empty_rows_dropped} completely empty row(s).")

    empty_cols_before = len(cleaned_df.columns)
    cleaned_df = cleaned_df.dropna(axis=1, how='all')
    empty_cols_dropped = empty_cols_before - len(cleaned_df.columns)
    if empty_cols_dropped > 0:
        steps_applied.append(f"Removed {empty_cols_dropped} completely empty column(s).")

    # 2. Type conversions for selected/apparent numeric columns
    numeric_converted = []
    convert_targets = custom_numeric_cols or []
    if not convert_targets:
        # Detect object and string columns that can mostly be converted to numeric
        for col in cleaned_df.select_dtypes(include=['object', 'string']).columns:
            non_null = cleaned_df[col].dropna()
            if len(non_null) > 0:
                converted = pd.to_numeric(non_null, errors='coerce')
                # If over 80% convert successfully, convert column
                if converted.notna().mean() >= 0.8:
                    convert_targets.append(col)

    for col in convert_targets:
        if col in cleaned_df.columns and not pd.api.types.is_numeric_dtype(cleaned_df[col]):
            before_invalid = cleaned_df[col].isna().sum()
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            after_invalid = cleaned_df[col].isna().sum()
            new_invalid = after_invalid - before_invalid
            numeric_converted.append(col)
            steps_applied.append(f"Converted '{col}' to numeric (detected {new_invalid} non-numeric entries).")

    # 3. Handle duplicates
    duplicates_removed = 0
    if remove_duplicates:
        duplicates_removed = int(cleaned_df.duplicated().sum())
        if duplicates_removed > 0:
            cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
            steps_applied.append(f"Removed {duplicates_removed} duplicate row(s).")
        else:
            steps_applied.append("Deduplication checked: No duplicate rows found.")
    else:
        steps_applied.append(f"Deduplication skipped: {initial_duplicates} duplicate row(s) retained as requested.")

    # 4. Handle missing values in numerical columns
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    missing_handled_count = 0

    if missing_strategy == 'mean':
        for col in numeric_columns:
            null_count = cleaned_df[col].isna().sum()
            if null_count > 0:
                mean_val = float(np.nanmean(cleaned_df[col]))
                cleaned_df[col] = cleaned_df[col].fillna(round(mean_val, 2))
                missing_handled_count += null_count
                steps_applied.append(f"Imputed {null_count} missing value(s) in '{col}' using mean ({mean_val:.2f}).")
    elif missing_strategy == 'median':
        for col in numeric_columns:
            null_count = cleaned_df[col].isna().sum()
            if null_count > 0:
                median_val = float(np.nanmedian(cleaned_df[col]))
                cleaned_df[col] = cleaned_df[col].fillna(round(median_val, 2))
                missing_handled_count += null_count
                steps_applied.append(f"Imputed {null_count} missing value(s) in '{col}' using median ({median_val:.2f}).")
    elif missing_strategy == 'drop':
        rows_before_drop = len(cleaned_df)
        cleaned_df = cleaned_df.dropna().reset_index(drop=True)
        dropped_rows = rows_before_drop - len(cleaned_df)
        if dropped_rows > 0:
            steps_applied.append(f"Dropped {dropped_rows} row(s) containing missing values.")
    else:
        steps_applied.append(f"Missing values handling: Kept {initial_missing_total} missing value(s) unchanged as requested.")

    final_rows, final_cols = cleaned_df.shape
    final_missing_total = int(cleaned_df.isna().sum().sum())
    final_missing_by_col = cleaned_df.isna().sum().to_dict()
    final_duplicates = int(cleaned_df.duplicated().sum())

    summary = {
        'initial_rows': initial_rows,
        'initial_cols': initial_cols,
        'final_rows': final_rows,
        'final_cols': final_cols,
        'initial_missing': initial_missing_total,
        'final_missing': final_missing_total,
        'initial_duplicates': initial_duplicates,
        'final_duplicates': final_duplicates,
        'duplicates_removed': duplicates_removed,
        'empty_rows_dropped': empty_rows_dropped,
        'empty_cols_dropped': empty_cols_dropped,
        'missing_strategy': missing_strategy,
        'remove_duplicates_flag': remove_duplicates,
        'initial_missing_by_col': initial_missing_by_col,
        'final_missing_by_col': final_missing_by_col,
        'numeric_columns': numeric_columns,
        'categorical_columns': cleaned_df.select_dtypes(exclude=[np.number]).columns.tolist(),
        'steps_applied': steps_applied
    }

    return cleaned_df, summary


def calculate_statistics(df, mapping=None):
    """
    Calculate general and student-specific statistical KPIs.
    """
    if mapping is None:
        mapping = detect_columns(df)

    student_col = mapping.get('student_col')
    score_col = mapping.get('score_col')
    attendance_col = mapping.get('attendance_col')
    subject_cols = mapping.get('subject_cols', [])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # If score_col is not set but subject_cols are present, compute a dynamic average score series
    score_series = None
    if score_col and score_col in df.columns and pd.api.types.is_numeric_dtype(df[score_col]):
        score_series = df[score_col]
    elif subject_cols:
        valid_subs = [s for s in subject_cols if s in df.columns and pd.api.types.is_numeric_dtype(df[s])]
        if valid_subs:
            score_series = df[valid_subs].mean(axis=1)

    attendance_series = None
    if attendance_col and attendance_col in df.columns and pd.api.types.is_numeric_dtype(df[attendance_col]):
        attendance_series = df[attendance_col]

    stats = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'numerical_columns_count': len(numeric_cols),
        'categorical_columns_count': len(categorical_cols),
        'total_missing_values': int(df.isna().sum().sum()),
        'total_duplicate_rows': int(df.duplicated().sum()),
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'student_col': student_col,
        'score_col': score_col,
        'attendance_col': attendance_col,
        'subject_cols': subject_cols,
    }

    # Student specific stats
    if score_series is not None and len(score_series.dropna()) > 0:
        valid_scores = score_series.dropna()
        stats['has_scores'] = True
        stats['avg_score'] = round(float(valid_scores.mean()), 2)
        stats['max_score'] = round(float(valid_scores.max()), 2)
        stats['min_score'] = round(float(valid_scores.min()), 2)
        stats['median_score'] = round(float(valid_scores.median()), 2)
        # Pass percentage (>= 50 threshold)
        pass_count = int((valid_scores >= 50).sum())
        stats['pass_count'] = pass_count
        stats['pass_percentage'] = round((pass_count / len(valid_scores)) * 100, 2)
    else:
        stats['has_scores'] = False

    if attendance_series is not None and len(attendance_series.dropna()) > 0:
        valid_att = attendance_series.dropna()
        stats['has_attendance'] = True
        stats['avg_attendance'] = round(float(valid_att.mean()), 2)
        stats['max_attendance'] = round(float(valid_att.max()), 2)
        stats['min_attendance'] = round(float(valid_att.min()), 2)
    else:
        stats['has_attendance'] = False

    # Student count
    if student_col and student_col in df.columns:
        stats['student_count'] = int(df[student_col].nunique())
    else:
        stats['student_count'] = len(df)

    # At-risk count (attendance < 75 and score < 60)
    if stats.get('has_scores') and stats.get('has_attendance'):
        at_risk_mask = (attendance_series < 75) & (score_series < 60)
        stats['at_risk_count'] = int(at_risk_mask.sum())
        stats['attendance_score_corr'] = round(float(attendance_series.corr(score_series)), 4)
    else:
        stats['at_risk_count'] = None
        stats['attendance_score_corr'] = None

    # Subject-wise statistics
    subject_stats = {}
    for sub in subject_cols:
        if sub in df.columns and pd.api.types.is_numeric_dtype(df[sub]):
            sub_clean = df[sub].dropna()
            if len(sub_clean) > 0:
                subject_stats[sub] = {
                    'mean': round(float(sub_clean.mean()), 2),
                    'max': round(float(sub_clean.max()), 2),
                    'min': round(float(sub_clean.min()), 2),
                    'std': round(float(sub_clean.std()), 2)
                }
    stats['subject_stats'] = subject_stats

    # Descriptive statistics dataframe summary (mean, std, min, 25%, 50%, 75%, max)
    if numeric_cols:
        desc_df = df[numeric_cols].describe().round(2).T
        desc_df = desc_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        stats['descriptive_stats'] = desc_df.to_dict(orient='index')
    else:
        stats['descriptive_stats'] = {}

    return stats


def _fig_to_base64(fig):
    """Convert Matplotlib figure to base64 PNG data URI and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"


def generate_visualizations(df, mapping=None):
    """
    Generate student performance & analytical charts using Matplotlib and Seaborn.
    Returns a dictionary of base64 image strings.
    """
    if mapping is None:
        mapping = detect_columns(df)

    student_col = mapping.get('student_col')
    score_col = mapping.get('score_col')
    attendance_col = mapping.get('attendance_col')
    subject_cols = mapping.get('subject_cols', [])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    charts = {}

    # Setup aesthetic styling
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = sns.color_palette("deep")

    # Derive score series if needed
    score_series = None
    score_label = "Marks / Score"
    if score_col and score_col in df.columns and pd.api.types.is_numeric_dtype(df[score_col]):
        score_series = df[score_col].dropna()
        score_label = score_col
    elif subject_cols:
        valid_subs = [s for s in subject_cols if s in df.columns and pd.api.types.is_numeric_dtype(df[s])]
        if valid_subs:
            score_series = df[valid_subs].mean(axis=1).dropna()
            score_label = "Average Score (Subjects Mean)"

    # -------------------------------------------------------------
    # 1. Subject-wise average marks (Bar chart)
    # -------------------------------------------------------------
    valid_subs = [s for s in subject_cols if s in df.columns and pd.api.types.is_numeric_dtype(df[s])]
    if len(valid_subs) >= 2:
        try:
            sub_means = df[valid_subs].mean().round(2)
            clean_labels = [s.replace('_Score', '').replace('_', ' ').title() for s in valid_subs]

            fig, ax = plt.subplots(figsize=(8, 4.8))
            bars = ax.bar(clean_labels, sub_means.values, color=sns.color_palette("mako", len(valid_subs)), edgecolor='#2b2d42', linewidth=0.8, width=0.55)
            
            # Value labels above bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1e293b')

            ax.set_title("Subject-wise Average Marks", fontsize=14, fontweight='bold', pad=12, color='#0f172a')
            ax.set_xlabel("Subjects", fontsize=11, labelpad=8)
            ax.set_ylabel("Average Score", fontsize=11, labelpad=8)
            ax.set_ylim(0, max(sub_means.max() * 1.15, 100))
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            charts['subject_averages'] = _fig_to_base64(fig)
        except Exception as e:
            plt.close('all')

    # -------------------------------------------------------------
    # 2. Student score distribution (Histogram + KDE)
    # -------------------------------------------------------------
    if score_series is not None and len(score_series) > 1:
        try:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            sns.histplot(score_series, kde=True, color='#2563eb', edgecolor='#1e40af', bins=15, ax=ax, alpha=0.6)
            
            # Add mean and passing threshold lines
            mean_score = score_series.mean()
            ax.axvline(mean_score, color='#dc2626', linestyle='--', linewidth=2, label=f'Mean Score: {mean_score:.1f}')
            ax.axvline(50, color='#16a34a', linestyle=':', linewidth=2, label='Passing Threshold: 50')

            ax.set_title(f"Score Distribution ({score_label})", fontsize=14, fontweight='bold', pad=12, color='#0f172a')
            ax.set_xlabel("Score / Marks", fontsize=11, labelpad=8)
            ax.set_ylabel("Student Count", fontsize=11, labelpad=8)
            ax.legend(frameon=True, facecolor='white', loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.5)
            charts['score_distribution'] = _fig_to_base64(fig)
        except Exception as e:
            plt.close('all')

    # -------------------------------------------------------------
    # 3. Attendance analysis (Histogram / Bar chart)
    # -------------------------------------------------------------
    if attendance_col and attendance_col in df.columns and pd.api.types.is_numeric_dtype(df[attendance_col]):
        att_series = df[attendance_col].dropna()
        if len(att_series) > 1:
            try:
                fig, ax = plt.subplots(figsize=(8, 4.8))
                sns.histplot(att_series, kde=True, color='#0d9488', edgecolor='#115e59', bins=12, ax=ax, alpha=0.6)
                
                # 75% required attendance line
                avg_att = att_series.mean()
                ax.axvline(avg_att, color='#2563eb', linestyle='--', linewidth=2, label=f'Average: {avg_att:.1f}%')
                ax.axvline(75, color='#e11d48', linestyle='-', linewidth=2, label='75% Minimum Required')

                ax.set_title("Student Attendance Percentage Distribution", fontsize=14, fontweight='bold', pad=12, color='#0f172a')
                ax.set_xlabel("Attendance (%)", fontsize=11, labelpad=8)
                ax.set_ylabel("Student Count", fontsize=11, labelpad=8)
                ax.legend(frameon=True, facecolor='white', loc='upper left')
                ax.grid(True, linestyle='--', alpha=0.5)
                charts['attendance_analysis'] = _fig_to_base64(fig)
            except Exception as e:
                plt.close('all')

    # -------------------------------------------------------------
    # 4. Attendance vs Academic Performance (Scatter Plot)
    # -------------------------------------------------------------
    if attendance_col and attendance_col in df.columns and score_series is not None:
        try:
            valid_idx = df[attendance_col].dropna().index.intersection(score_series.dropna().index)
            if len(valid_idx) > 2:
                x = df.loc[valid_idx, attendance_col]
                y = score_series.loc[valid_idx]

                fig, ax = plt.subplots(figsize=(8, 4.8))
                
                # Check if Pass_Fail column or Grade exists for hue
                hue_col = None
                for c in ['Pass_Fail', 'pass_fail', 'Status', 'Grade']:
                    if c in df.columns:
                        hue_col = c
                        break

                if hue_col:
                    sns.scatterplot(x=x, y=y, hue=df.loc[valid_idx, hue_col], palette="Set2", s=65, alpha=0.85, ax=ax, edgecolor='#334155')
                else:
                    sns.scatterplot(x=x, y=y, color='#6366f1', s=65, alpha=0.85, ax=ax, edgecolor='#312e81')

                # Regression trendline using NumPy
                try:
                    slope, intercept = np.polyfit(x, y, 1)
                    trend_x = np.linspace(x.min(), x.max(), 100)
                    ax.plot(trend_x, slope * trend_x + intercept, color='#dc2626', linestyle='--', linewidth=2,
                            label=f'Trendline (m={slope:.2f})')
                except Exception:
                    pass

                corr = x.corr(y)
                ax.set_title(f"Attendance vs. Academic Performance (r = {corr:.3f})", fontsize=14, fontweight='bold', pad=12, color='#0f172a')
                ax.set_xlabel("Attendance (%)", fontsize=11, labelpad=8)
                ax.set_ylabel("Score / Marks", fontsize=11, labelpad=8)
                ax.legend(frameon=True, facecolor='white')
                ax.grid(True, linestyle='--', alpha=0.5)
                charts['attendance_vs_score'] = _fig_to_base64(fig)
        except Exception as e:
            plt.close('all')

    # -------------------------------------------------------------
    # 5. Top-performing students (Bar chart)
    # -------------------------------------------------------------
    if score_series is not None and student_col and student_col in df.columns:
        try:
            plot_df = pd.DataFrame({
                'Student': df[student_col].astype(str),
                'Score': score_series
            }).dropna().sort_values(by='Score', ascending=False).head(10)

            if len(plot_df) > 1:
                fig, ax = plt.subplots(figsize=(8, 4.8))
                # Horizontal bar chart for clean name readability
                bars = ax.barh(plot_df['Student'][::-1], plot_df['Score'][::-1], color=sns.color_palette("viridis", len(plot_df)), edgecolor='#334155', height=0.6)
                
                for bar in bars:
                    width = bar.get_width()
                    ax.annotate(f'{width:.1f}',
                                xy=(width, bar.get_y() + bar.get_height() / 2),
                                xytext=(5, 0), textcoords="offset points",
                                ha='left', va='center', fontsize=9.5, fontweight='bold', color='#1e293b')

                ax.set_title("Top 10 Performing Students", fontsize=14, fontweight='bold', pad=12, color='#0f172a')
                ax.set_xlabel("Score / Marks", fontsize=11, labelpad=8)
                ax.set_xlim(0, max(plot_df['Score'].max() * 1.15, 100))
                ax.grid(axis='x', linestyle='--', alpha=0.5)
                charts['top_students'] = _fig_to_base64(fig)
        except Exception as e:
            plt.close('all')

    # -------------------------------------------------------------
    # 6. Correlation heatmap (Seaborn heatmap)
    # -------------------------------------------------------------
    if len(numeric_cols) >= 2:
        try:
            # Filter numeric columns for correlation (exclude ID columns if they look like simple sequences)
            corr_cols = [c for c in numeric_cols if not c.lower().endswith('_id') and c.lower() != 'id']
            if len(corr_cols) < 2:
                corr_cols = numeric_cols

            corr_matrix = df[corr_cols].corr()

            fig, ax = plt.subplots(figsize=(8, 5.2))
            clean_labels = [c.replace('_', ' ').title() for c in corr_cols]
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                        center=0, square=True, linewidths=0.8, cbar_kws={"shrink": 0.8},
                        xticklabels=clean_labels, yticklabels=clean_labels, ax=ax)

            ax.set_title("Correlation Matrix (Numerical Features)", fontsize=14, fontweight='bold', pad=12, color='#0f172a')
            plt.xticks(rotation=35, ha='right', fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            charts['correlation_heatmap'] = _fig_to_base64(fig)
        except Exception as e:
            plt.close('all')

    # -------------------------------------------------------------
    # Generic Dataset Visualizations Fallback
    # (If dataset is not student-related, e.g. generic numerical/categorical data)
    # -------------------------------------------------------------
    if 'score_distribution' not in charts and numeric_cols:
        try:
            # Distribution of first numeric column
            target_col = numeric_cols[0]
            fig, ax = plt.subplots(figsize=(8, 4.8))
            sns.histplot(df[target_col].dropna(), kde=True, color='#3b82f6', ax=ax)
            ax.set_title(f"Distribution of {target_col}", fontsize=14, fontweight='bold', pad=12)
            ax.grid(True, linestyle='--', alpha=0.5)
            charts['generic_distribution'] = _fig_to_base64(fig)
        except Exception:
            plt.close('all')

    return charts
