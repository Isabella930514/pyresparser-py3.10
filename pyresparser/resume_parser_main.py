from pyresparser import ResumeParser
import json
import os


def resume_parser(resume_name_list, path):
    resume_list = []
    print('start to deal with resumes...')
    for resume in resume_name_list:
        data = ResumeParser(f'{path}/{resume}').get_extracted_data()
        resume_list.append(data)
    # Write all resume info to a json file so it can be re-used later
    filename = 'resume_jobs_info.json'
    file_path = './pyresparser/outputs'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    filepath = os.path.join(file_path, filename)

    with open(filepath, 'w') as fp:
        json.dump(resume_list, fp)

    return resume_list
