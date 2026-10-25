import os
import io
import unittest
from app import app
import pandas as pd

class StudentAnalyzerAppTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def test_01_homepage_initial(self):
        """Test initial homepage without dataset loaded."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Upload Student Performance CSV", response.data)
        self.assertIn(b"Drag and drop your CSV file here", response.data)

    def test_02_load_sample(self):
        """Test loading sample dataset."""
        response = self.client.get('/load-sample', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"sample_students.csv", response.data)
        self.assertIn(b"Total Records", response.data)
        self.assertIn(b"Average Score", response.data)
        self.assertIn(b"Subject-wise Average Marks", response.data)
        self.assertIn(b"Correlation Matrix Heatmap", response.data)

    def test_03_preprocessing_options(self):
        """Test updating preprocessing options (e.g. mean imputation)."""
        # First load sample
        self.client.get('/load-sample', follow_redirects=True)
        # Apply preprocessing
        response = self.client.post('/preprocess', data={
            'remove_duplicates': 'on',
            'missing_strategy': 'mean'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Preprocessing settings updated", response.data)

    def test_04_download_cleaned_data(self):
        """Test downloading cleaned CSV."""
        self.client.get('/load-sample', follow_redirects=True)
        response = self.client.get('/download')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['Content-Type'], 'text/csv; charset=utf-8')
        # Check that downloaded content is valid CSV
        csv_text = response.data.decode('utf-8')
        self.assertIn("Student_ID", csv_text)
        self.assertIn("Name", csv_text)

    def test_05_upload_valid_custom_csv(self):
        """Test uploading a CSV with different column names and structure."""
        custom_csv = (
            "RollNo,StudentFullName,ExamScore,PresenceRate\n"
            "101,John Doe,85.5,92.0\n"
            "102,Jane Smith,78.0,88.5\n"
            "103,Alex Brown,45.0,60.0\n"
            "104,Emily Davis,95.0,98.0\n"
        )
        data = {
            'csv_file': (io.BytesIO(custom_csv.encode('utf-8')), 'test_custom_students.csv')
        }
        response = self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"test_custom_students.csv", response.data)
        self.assertIn(b"John Doe", response.data)
        self.assertIn(b"ExamScore", response.data)

    def test_06_upload_invalid_file_type(self):
        """Test uploading a non-CSV file (should be rejected gracefully)."""
        data = {
            'csv_file': (io.BytesIO(b"Fake image data"), 'document.pdf')
        }
        response = self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Invalid file format", response.data)

    def test_07_upload_empty_csv(self):
        """Test uploading an empty CSV file."""
        data = {
            'csv_file': (io.BytesIO(b""), 'empty.csv')
        }
        response = self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"empty", response.data)

    def test_08_reset(self):
        """Test resetting session."""
        self.client.get('/load-sample', follow_redirects=True)
        response = self.client.get('/reset', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Upload Student Performance CSV", response.data)

if __name__ == '__main__':
    unittest.main()
