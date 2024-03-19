from flask import Flask, render_template, request, jsonify
from data_preparation import web_scrapper, job_skill_graph

app = Flask(__name__)


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
    job_skill_graph.job_skill_graph_def(jobs_info)
    return jsonify({"message": "Job search initiated successfully."})


if __name__ == '__main__':
    app.run(debug=False)
