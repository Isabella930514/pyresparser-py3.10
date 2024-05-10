from flask import Flask, render_template, request, jsonify, url_for
from data_preparation import web_scrapper, job_skill_graph, indeed_job_recommendation
from werkzeug.utils import secure_filename
from pyresparser import resume_parser_main
import os

app = Flask(__name__)

RESUME_UPLOAD_FOLDER = './pyresparser/resumes'
app.config['RESUME_UPLOAD_FOLDER'] = RESUME_UPLOAD_FOLDER
app.config['SENTENSE_UPLOAD_FOLDER'] = './data_preparation/datasets'


@app.route("/")
@app.route("/home/")
def home():
    return render_template("home.html")


@app.route("/job_information_crawling/")
def job_information_crawling():
    return render_template("job_information_crawling.html")


@app.route('/search_jobs_info/', methods=['POST'])
def search_jobs():
    job_information = request.get_data(as_text=True)
    # ---- Scrape from web or read from local saved -----
    # ---------------------------------------------------
    jobs_info = web_scrapper.get_jobs_info(job_information)
    # ---------------------------------------------------
    # -------- job and skills graph construction ----------
    # ---------------------------------------------------
    # job_skill_graph.job_skill_graph_def(jobs_info)
    return jsonify(jobs_info)


@app.route("/resume_information_extraction/")
def resume_information_extraction():
    return render_template("resume_information_extraction.html")


@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    # check file in request
    uploaded_files = request.files.getlist('files[]')

    if not uploaded_files:
        return jsonify({"error": "No files uploaded"}), 400

    file_list = []
    for file in uploaded_files:
        if file:
            filename = secure_filename(file.filename)
            resume_upload_folder = app.config['RESUME_UPLOAD_FOLDER']
            if not os.path.exists(resume_upload_folder):
                os.makedirs(resume_upload_folder)
            save_path = os.path.join(resume_upload_folder, filename)
            file.save(save_path)
            file_list.append(filename)
    resume_list = resume_parser_main.resume_parser(file_list, app.config['RESUME_UPLOAD_FOLDER'])

    return jsonify(resume_list)


@app.route("/information_conversion/")
def information_conversion():
    return render_template("information_conversion.html")


@app.route('/convert_and_augment', methods=['POST'])
def convert_and_augment():
    # check file in request
    file = request.files['file']
    augment = request.form['augment']
    prediction = request.form['prediction']
    model = request.form['model']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['SENTENSE_UPLOAD_FOLDER'], filename)
        file.save(save_path)

        if augment == 'false' and prediction == 'false':
            indeed_job_recommendation.convert_kg(model, save_path, False, False)
        elif augment == 'true' and prediction == 'false':
            indeed_job_recommendation.convert_kg(model, save_path, True, False)
        elif augment == 'false' and prediction == 'true':
            indeed_job_recommendation.convert_kg(model, save_path, False, True)
        elif augment == 'true' and prediction == 'true':
            indeed_job_recommendation.convert_kg(model, save_path, True, True)
        return jsonify({"success": True})

    return jsonify({"error": "File processing failed"}), 500


@app.route('/network.html')
def network_rebel():
    return render_template('network.html')


if __name__ == '__main__':
    app.run(debug=False)
