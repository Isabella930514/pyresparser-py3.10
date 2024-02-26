# Saved file for each job url
JOBS_LINKS_JSON_FILE = r'C:/Users/cjv2124/pyresparser-py3.10/data_preparation/indeed_jobs_links.json'
# Saved file for each job info
JOBS_INFO_JSON_FILE = r'C:/Users/cjv2124/pyresparser-py3.10/data_preparation/indeed_jobs_info.json'
# Saved file for recommended jobs
RECOMMENDED_JOBS_FILE = r'./data_preparation/recommended_jobs'
# Path to webdriver exe
WEBDRIVER_PATH = r'./data_preparation/chromedriver_win32/chromedriver.exe'
# Cities to search: 6 largest Canadian cities
JOB_LOCATIONS = ['Auckland']
# Seach "data scientist" OR "data+engineer" OR "data+analyst" with quotation marks
# JOB_SEARCH_WORDS = 'data+scientist+or+data+engineer+or+data+analyst+or+developer+' \
#                    'or+AI+engineer+or+business+intelligence+or+SQL+developer'
JOB_SEARCH_WORDS = 'data+or+scientist'
# To avoid same job posted multiple times, we only look back for 30 days
DAY_RANGE = 30
# Path to sample resume
SAMPLE_RESUME_PDF = r'./data_preparation/Isabel.txt'
# Resume names
RESUME_NAME_LIST = ['Isabel.pdf']
# Saved file for each resume info
RESUME_INFO_JSON_FILE = r'C:/Users/cjv2124/pyresparser-py3.10/pyresparser/resume_jobs_info.json'
# Skills Data Sources
SKILLS_DATA = r'C:/Users/cjv2124/pyresparser-py3.10/pyresparser/skills.csv'
# TEXT-2-TRIPLE TEXT
TEXT_FILE = r'C:/Users/cjv2124/pyresparser-py3.10/data_preparation/wiki_sentences_v2.csv'
# Extractor Type -> 'REBEL', 'SPACY'
EXTRACTOR_TYPE = 'REBEL'
# if extract neighbours from wikipedia
IF_NEIGHBOUR = True
# SET the number of expanded-node and max neighours
EXPEND_NUM = 10
MAX_NEIGH = 10
# wiki link
endpoint_url = "https://query.wikidata.org/sparql"