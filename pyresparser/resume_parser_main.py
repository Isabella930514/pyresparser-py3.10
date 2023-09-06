from pyresparser import ResumeParser
from data_preparation import config
import json

resume_list = []
print('start to deal with resumes...')
for resume in config.RESUME_NAME_LIST:
    data = ResumeParser(f'C:/Users/cjv2124/pyresparser-py3.10/pyresparser/resumes/{resume}').get_extracted_data()
    resume_list.append(data)
    # Write all resume info to a json file so it can be re-used later
with open(config.RESUME_INFO_JSON_FILE, 'w') as fp:
    json.dump(resume_list, fp)