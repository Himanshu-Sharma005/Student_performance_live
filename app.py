import io
import os
import uuid
import tempfile
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, send_file, jsonify
)
from werkzeug.utils import secure_filename
import analyzer

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "student-performance-analyzer-secret-2026")

# 10 MB upload limit
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Directory for uploads: use local uploads/ if writable; otherwise fallback to temp dir (e.g. on Vercel/serverless)
SAMPLE_FILE = os.path.join(BASE_DIR, 'sample_students.csv')
local_upload_dir = os.path.join(BASE_DIR, 'uploads')
try:
    os.makedirs(local_upload_dir, exist_ok=True)
    test_path = os.path.join(local_upload_dir, '.write_test')
    with open(test_path, 'w') as f:
        f.write('ok')
    os.remove(test_path)
    UPLOAD_FOLDER = local_upload_dir
except Exception:
    UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'spa_uploads')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)



def get_session_file():
    """Retrieve the path to the current user's uploaded CSV file."""
    file_id = session.get('file_id')
    if not file_id:
        return None
    filepath = os.path.join(UPLOAD_FOLDER, f"{file_id}.csv")
    if os.path.exists(filepath):
        return filepath
    return None


@app.route('/')
def index():
    filepath = get_session_file()
    if not filepath:
        # No dataset loaded yet: render upload homepage
        return render_template('index.html', has_data=False)

    # Load raw data from session file
    df_raw, err = analyzer.load_csv_data(filepath)
    if err or df_raw is None:
        flash(err or "Failed to read data file.", "danger")
        session.pop('file_id', None)
        return render_template('index.html', has_data=False)

    # Retrieve preprocessing options from session (or set defaults)
    remove_duplicates = session.get('remove_duplicates', True)
    missing_strategy = session.get('missing_strategy', 'none')

    # Preprocess data
    df_cleaned, preproc_summary = analyzer.preprocess_data(
        df_raw,
        remove_duplicates=remove_duplicates,
        missing_strategy=missing_strategy
    )

    # Column mapping (custom or auto-detected)
    custom_mapping = session.get('column_mapping')
    detected_mapping = analyzer.detect_columns(df_cleaned)
    if not custom_mapping:
        mapping = detected_mapping
    else:
        # Merge with detected to ensure all keys exist
        mapping = {**detected_mapping, **custom_mapping}

    # Calculate statistics & KPIs
    stats = analyzer.calculate_statistics(df_cleaned, mapping)

    # Generate charts
    charts = analyzer.generate_visualizations(df_cleaned, mapping)

    # Data preview: first 10 rows
    preview_df = df_cleaned.head(10)
    preview_data = {
        'columns': df_cleaned.columns.tolist(),
        'dtypes': {c: str(df_cleaned[c].dtype) for c in df_cleaned.columns},
        'rows': preview_df.values.tolist(),
        'total_rows': len(df_cleaned),
        'total_cols': len(df_cleaned.columns)
    }

    filename = session.get('original_filename', 'uploaded_dataset.csv')

    return render_template(
        'index.html',
        has_data=True,
        filename=filename,
        preproc_summary=preproc_summary,
        remove_duplicates=remove_duplicates,
        missing_strategy=missing_strategy,
        mapping=mapping,
        stats=stats,
        charts=charts,
        preview=preview_data
    )


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'csv_file' not in request.files:
        flash("No file part in upload request.", "warning")
        return redirect(url_for('index'))

    file = request.files['csv_file']
    if not file or file.filename == '':
        flash("No file selected. Please choose a CSV file.", "warning")
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    if not filename.lower().endswith('.csv'):
        flash("Invalid file format. Please upload a valid CSV file (.csv).", "danger")
        return redirect(url_for('index'))

    # Read and validate with pandas
    try:
        df, err = analyzer.load_csv_data(file)
        if err or df is None:
            flash(err or "Unable to parse CSV file. Please verify file integrity.", "danger")
            return redirect(url_for('index'))
    except Exception as e:
        flash(f"Error parsing CSV: {str(e)}", "danger")
        return redirect(url_for('index'))

    # Save to session file
    file_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_FOLDER, f"{file_id}.csv")
    df.to_csv(filepath, index=False)

    # Clean up previous session file if present
    old_file = get_session_file()
    if old_file and os.path.exists(old_file):
        try:
            os.remove(old_file)
        except Exception:
            pass

    session['file_id'] = file_id
    session['original_filename'] = filename
    session['remove_duplicates'] = True
    session['missing_strategy'] = 'none'
    session.pop('column_mapping', None)  # Reset mapping on new file upload

    flash(f"Successfully uploaded '{filename}' with {len(df)} rows and {len(df.columns)} columns.", "success")
    return redirect(url_for('index'))


@app.route('/load-sample')
def load_sample():
    if not os.path.exists(SAMPLE_FILE):
        flash("Sample dataset file not found on server.", "danger")
        return redirect(url_for('index'))

    df, err = analyzer.load_csv_data(SAMPLE_FILE)
    if err or df is None:
        flash("Failed to load sample dataset.", "danger")
        return redirect(url_for('index'))

    file_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_FOLDER, f"{file_id}.csv")
    df.to_csv(filepath, index=False)

    session['file_id'] = file_id
    session['original_filename'] = 'sample_students.csv'
    session['remove_duplicates'] = True
    session['missing_strategy'] = 'none'
    session.pop('column_mapping', None)

    flash("Sample student dataset loaded successfully! Explore the preview, KPIs, and visualizations below.", "success")
    return redirect(url_for('index'))


@app.route('/preprocess', methods=['POST'])
def update_preprocessing():
    filepath = get_session_file()
    if not filepath:
        flash("No active dataset to preprocess. Please upload a file.", "warning")
        return redirect(url_for('index'))

    remove_dups = request.form.get('remove_duplicates') == 'on'
    missing_strat = request.form.get('missing_strategy', 'none')

    session['remove_duplicates'] = remove_dups
    session['missing_strategy'] = missing_strat

    flash("Preprocessing settings updated and applied successfully.", "info")
    return redirect(url_for('index') + "#preprocessing-section")


@app.route('/update-columns', methods=['POST'])
def update_columns():
    filepath = get_session_file()
    if not filepath:
        flash("No active dataset.", "warning")
        return redirect(url_for('index'))

    student_col = request.form.get('student_col') or None
    score_col = request.form.get('score_col') or None
    attendance_col = request.form.get('attendance_col') or None
    subject_cols = request.form.getlist('subject_cols')

    session['column_mapping'] = {
        'student_col': student_col,
        'score_col': score_col,
        'attendance_col': attendance_col,
        'subject_cols': subject_cols
    }

    flash("Column mappings updated successfully. Analytics and charts have been refreshed.", "success")
    return redirect(url_for('index') + "#analytics-section")


@app.route('/download')
def download_cleaned_data():
    filepath = get_session_file()
    if not filepath:
        flash("No dataset available for download.", "warning")
        return redirect(url_for('index'))

    df_raw, err = analyzer.load_csv_data(filepath)
    if err or df_raw is None:
        flash("Error preparing file for download.", "danger")
        return redirect(url_for('index'))

    remove_duplicates = session.get('remove_duplicates', True)
    missing_strategy = session.get('missing_strategy', 'none')

    df_cleaned, _ = analyzer.preprocess_data(
        df_raw,
        remove_duplicates=remove_duplicates,
        missing_strategy=missing_strategy
    )

    out_filename = f"cleaned_{session.get('original_filename', 'students_data.csv')}"
    buffer = io.BytesIO()
    df_cleaned.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=out_filename,
        mimetype='text/csv'
    )


@app.route('/reset')
def reset():
    filepath = get_session_file()
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception:
            pass

    session.clear()
    flash("Session reset. You can now upload a new dataset.", "info")
    return redirect(url_for('index'))


@app.errorhandler(413)
def request_entity_too_large(error):
    flash("Uploaded file is too large! The maximum allowed file size is 10 MB.", "danger")
    return redirect(url_for('index')), 413


@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html', has_data=False, error_msg="Page not found (404)"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', has_data=False, error_msg="Internal server error (500). Please try again."), 500


if __name__ == '__main__':
    # Run development server
    app.run(host='127.0.0.1', port=5000, debug=True)
