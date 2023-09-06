# Saved file for each job url
JOBS_LINKS_JSON_FILE = r'C:/Users/cjv2124/pyresparser-py3.10/data_preparation/indeed_jobs_links.json'
# Saved file for each job info
JOBS_INFO_JSON_FILE = r'C:/Users/cjv2124/pyresparser-py3.10/data_preparation/indeed_jobs_info.json'
# Saved file for recommended jobs
RECOMMENDED_JOBS_FILE = r'./data_preparation/recommended_jobs'
# Path to webdriver exe
WEBDRIVER_PATH = r'./data_preparation/chromedriver_win32/chromedriver.exe'
# Cities to search: 6 largest Canadian cities
JOB_LOCATIONS = ['Auckland', 'Wellington']
# Seach "data scientist" OR "data+engineer" OR "data+analyst" with quotation marks
# JOB_SEARCH_WORDS = 'data+scientist+or+data+engineer+or+data+analyst+or+developer+' \
#                    'or+AI+engineer+or+business+intelligence+or+SQL+developer'
JOB_SEARCH_WORDS = 'data+or+scientist'
# To avoid same job posted multiple times, we only look back for 30 days
DAY_RANGE = 30
# Path to sample resume
SAMPLE_RESUME_PDF = r'./data_preparation/PWang_resume.pdf'
# Resume names
RESUME_NAME_LIST = ['OmkarResume.pdf', 'Isabel.pdf', 'Shun.pdf', 'Kevin.pdf']
# Saved file for each resume info
RESUME_INFO_JSON_FILE = r'C:/Users/cjv2124/pyresparser-py3.10/data_preparation/resume_jobs_info.json'
# Skills Data Sources
SKILLS_DATA = r'C:/Users/cjv2124/pyresparser-py3.10/pyresparser/skills.csv'