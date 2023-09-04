from pyresparser import ResumeParser
data = ResumeParser('./OmkarResume.pdf').get_extracted_data()
for key, value in data.items():
    print(f"{key}\t{value}")