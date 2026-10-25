# Student Performance Analyzer

An interactive Python and Flask-based web application and analytical dashboard that enables users to upload student performance CSV files, preprocess data using Pandas and NumPy, calculate academic and statistical KPIs, and view interactive data visualizations generated with Matplotlib and Seaborn.

---

## Key Features

- **Interactive CSV Upload**:
  - Drag-and-drop zone and file picker for CSV datasets.
  - Supports CSVs with varied column names and structures.
  - File validation (empty files, non-CSV formats, size limit of 10 MB).
  - One-click **"Try Sample Data"** option for instant demonstration.
  - Option to upload a new CSV or reset at any time.

- **Transparent Data Preprocessing Pipeline**:
  - Automatically drops completely empty rows and columns.
  - Detects duplicate rows with a user option to deduplicate.
  - Identifies missing values per column with user-configurable imputation (leave unchanged, fill with mean, fill with median, or drop).
  - Detects numeric and categorical columns and converts compatible text columns to numeric.
  - Displays a before-and-after preprocessing summary and action audit trail.

- **Smart Column Detection & Custom Mapping**:
  - Automatically maps Student ID/Name, Overall Score, Attendance, and Subject columns.
  - Interactive **Column Settings Modal** allowing users to customize column mappings if header names differ.
  - Graceful fallback for generic datasets (shows numerical distributions and correlation heatmaps).

- **Academic & Dataset KPI Cards**:
  - Dataset metrics: Total records, Total columns, Numerical columns count, Categorical columns count, Missing values count, Duplicate rows count.
  - Academic KPIs: Average Score, Highest Score, Lowest Score, Median Score, Average Attendance, Total Unique Students, Pass Rate (%), and At-Risk Students Count (Attendance < 75% and Score < 60%).

- **Polished Data Visualizations**:
  1. **Subject-wise Average Marks**: Bar chart comparing average scores across subjects (e.g., Math, Science, English, History).
  2. **Student Score Distribution**: Histogram with KDE density curve, class mean line, and passing threshold.
  3. **Attendance Analysis**: Attendance percentage distribution with a 75% minimum required benchmark line.
  4. **Attendance vs. Academic Performance**: Scatter plot with color coding and a linear trendline.
  5. **Top-Performing Students**: Horizontal bar chart highlighting the top 10 scoring students.
  6. **Correlation Matrix Heatmap**: Seaborn heatmap displaying correlations between all numerical features.
  - Chart Zoom modal for high-resolution inspection.

- **Export & Usability**:
  - One-click **Download Cleaned CSV** button to export preprocessed data.
  - Built-in **Light & Dark Theme Toggle** with persistent preference storage.
  - Responsive layout built with Bootstrap 5.3 and custom modern CSS.

---

## Technology Stack

- **Backend**: Python 3.11+, Flask 3.1
- **Data Manipulation & Computation**: Pandas 3.0, NumPy 2.5
- **Data Visualization**: Matplotlib 3.11, Seaborn 0.13
- **Frontend**: HTML5, CSS3, Bootstrap 5.3, Bootstrap Icons
- **Environment**: Virtual Environment (`.venv`)

---

## Project Structure

```text
student-performance-analyzer/
├── app.py                             # Main Flask application and route handlers
├── analyzer.py                        # Preprocessing, statistical analytics, and chart generation
├── test_app.py                        # Automated unit and integration tests
├── requirements.txt                   # Application dependencies
├── README.md                          # Project documentation
├── sample_students.csv                # Sample student dataset for testing & demo
├── students.csv                       # Original synthetic dataset with duplicates
├── Student_Performance_Analyzer.ipynb # Original Jupyter Notebook analysis (preserved)
├── summary_report.html                # Original static HTML summary report (preserved)
├── templates/
│   └── index.html                     # Responsive dashboard template (upload + analytics views)
├── static/
│   ├── css/
│   │   └── style.css                  # Custom styling for light/dark theme, cards, dropzone
│   └── images/                        # Static assets directory
└── uploads/                           # Temporary user session uploads (ignored in Git)
```

---

## Setup & Installation Instructions (Windows)

### 1. Clone the Repository
```cmd
git clone https://github.com/Himanshu-Sharma005/student-performance-analyzer.git
cd student-performance-analyzer
```

### 2. Create a Virtual Environment
```cmd
python -m venv .venv
```

### 3. Activate the Virtual Environment
**Windows Command Prompt (cmd.exe):**
```cmd
.venv\Scripts\activate.bat
```

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```cmd
python -m pip install -r requirements.txt
```

### 5. Run the Automated Tests (Optional)
```cmd
python -m unittest test_app.py -v
```

### 6. Start the Flask Application
```cmd
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## How to Use the Application

1. **Upload a Dataset**:
   - On the homepage, drag and drop a `.csv` file into the upload area or click **Browse Files**.
   - Alternatively, click **Try Sample Data** to immediately load `sample_students.csv`.
2. **Review Preprocessing**:
   - The **Data Preprocessing Pipeline Summary** displays the row and column counts before and after cleaning, detected duplicates, and missing values.
   - Click **Adjust Preprocessing Options** to toggle deduplication or change the missing value strategy (Leave unchanged, Mean, Median, or Drop rows).
3. **Customize Columns (If Needed)**:
   - If your dataset uses custom column names, click **Column Settings** in the top navigation bar.
   - Map your columns for Student Name/ID, Overall Score, Attendance, and Subjects.
4. **Explore Visualizations**:
   - Scroll through the 6 generated charts. Click **Zoom** or click on any chart to view it in full resolution.
5. **Download Cleaned Data**:
   - Click **Download Cleaned CSV** in the navigation bar to export the processed dataset.
6. **Upload Another Dataset**:
   - Click **New Dataset** to reset the session and upload another CSV file.

---

## Example CSV Format

The application accepts any CSV structure. For full academic features, column headers similar to the following are recommended:

| Student_ID | Name | Age | Gender | Attendance_Percentage | Math_Score | Science_Score | English_Score | History_Score | Avg_Score | Grade | Pass_Fail |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1001 | Elijah Martin | 18 | Female | 93 | 54 | 73 | 78 | 70 | 68.75 | C | Pass |
| 1002 | Lucas Hernandez | 17 | Female | 62 | 78 | 29 | 69 | 56 | 58.00 | D | Pass |
| 1003 | Charlotte Miller | 16 | Female | 71 | 84 | 100 | 74 | 86 | 86.00 | A | Pass |

*Note: Datasets with different column names (such as `RollNo`, `Candidate_Name`, `Marks_Physics`, `Presence_Pct`) will be automatically detected and can also be adjusted via the Column Settings modal.*

---

## Original Jupyter Notebook Workflow

The original data science workflow and SQLite analysis are preserved in:
- `Student_Performance_Analyzer.ipynb`
- `summary_report.html`

To run the notebook:
```cmd
jupyter notebook Student_Performance_Analyzer.ipynb
```
