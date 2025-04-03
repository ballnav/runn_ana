import os
import csv
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# Import your processing function from count4
from count8 import process_video

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index13.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save the uploaded video
        video_path = 'uploaded_video.mp4'
        file.save(video_path)

        # Process the video
        process_video(video_path)

        # Read the analysis results from count2.csv
        analysis_results = []
        with open('count3.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                analysis_results.append(row)

        # Read the summary data from gait_summary.csv
        summary_data = {}
        try:
            with open('gait_summary.csv', mode='r') as summary_file:
                csv_reader = csv.DictReader(summary_file)
                summary_data = next(csv_reader)  # Get the first (and only) row
        except (FileNotFoundError, StopIteration):
            pass  # Handle missing summary file gracefully

        # Return both the detailed analysis and summary data
        return jsonify({
            'analysis_results': analysis_results,
            'summary_data': summary_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Use 500 instead of 1500 for server errors


if __name__ == '__main__':
    app.run(debug=True)