"""
Created on Wed 2023 6

@author: Isabel Wang

"""

from collections import Counter
from data_preparation import config
from textblob import TextBlob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


def calculate_final_score(group):
    """
    give different weights for job-skill relationship calculation

    """
    weights = {
        'Weight': 0.2,
        'Senti': 0.5,
        'tfidf': 0.3
    }

    final_score = 0
    for index, row in group.iterrows():
        final_score += weights.get(row['level_0'], 0) * row.get(row['level_0'], 0)

    return final_score

def job_skill_graph_def(job_info_list):
    """
        we have three methods to calculate relationship between job and skills:
            1: frequency
            2: sentiment
            3: TF-IDF
        Capture different aspects of similarity from different angles by combining multiple different measures (with weighted averaging).

        """
    # get all job titles
    job_title_list = []
    for item in job_info_list:
        job_title_list.append(item['title'])

    # skills data source
    with open(config.SKILLS_DATA, 'r') as skill_file:
        skills = skill_file.readlines()
        if str(skills).startswith("[") and str(skills).endswith("]"):
            new_str = str(skills)[2:-2]
        skills_list = new_str.strip().split(',')

    for job in set(job_title_list):
        job_description_list = []
        for job_info in job_info_list:
            if job_info['title'] == job:
                job_description_list.append(job_info['desc'])
        job_description = '.'.join(job_description_list)

        # convert lower cases
        job_description_lower = job_description.lower()
        skills_list_lower = [skill.lower() for skill in skills_list]

        # calculate freq
        skill_freq = Counter()
        words = job_description_lower.split()

        for word in words:
            if word in skills_list_lower:
                skill_freq[word] += 1
        # Get all keys into a list
        keys_list = list(skill_freq.keys())

        if keys_list == []:
            continue

        # analyse job-skill relations from sentiment and TF-IDF perceptions
        df_senti = senti_skills(job, job_description, keys_list)
        df_tfidf = TF_IDF_skills(job, job_description, keys_list)

        # calculate weights
        total_count = sum(skill_freq.values())
        skill_weight = {skill: round((count / total_count), 3) for skill, count in skill_freq.items()}

        # create a DataFrame
        df = pd.DataFrame(list(skill_weight.items()), columns=['Skill', 'Weight'])

        # If the CSV file exists, read it into a DataFrame
        # If the file doesn't exist, create a new DataFrame
        try:
            df_existing = pd.read_csv("job-skill.csv")
        except FileNotFoundError:
            df_existing = pd.DataFrame()

        df.insert(1, 'Job', job)
        df = df[['Job', 'Weight', 'Skill']]
        df = df.sort_values(by='Weight', ascending=False)

        # ***final job-skills calcualtion***
        merged_df = pd.concat([df, df_senti, df_tfidf], keys=['Weight', 'Senti', 'tfidf']).reset_index()

        final_scores = merged_df.groupby(['Job', 'Skill']).apply(calculate_final_score).reset_index()
        final_scores.rename(columns={final_scores.columns[-1]: 'Final_Score'}, inplace=True)
        final_scores = final_scores[['Job', 'Final_Score', 'Skill']]
        final_scores = final_scores.sort_values(by='Final_Score', ascending=False)

        # Append the new data to the existing DataFrame
        df_combined = pd.concat([df_existing, final_scores], ignore_index=True)

        # Write the combined DataFrame back to the CSV file
        df_combined.to_csv("job-skill.csv", index=False)

    print("done")


def senti_skills(job, job_description, skills_list):
    """
    Analyse skills' sentiments for each job
    """
    blob = TextBlob(job_description)

    # Analyze sentences that mention specific skills
    skill_sentiment = {}
    for skill in skills_list:
        sentiment_list = []
        print(f"Sentence containing '{skill}':")
        for sentence in blob.sentences:
            if skill in sentence.string.lower():
                sentiment_list.append(sentence.sentiment.polarity)
        avg_sentiment = sum(sentiment_list)/len(sentiment_list)
        skill_sentiment[skill] = round(avg_sentiment, 3)

    df_senti = pd.DataFrame(list(skill_sentiment.items()), columns=['Skill', 'Senti'])
    df_senti.insert(1, 'Job', job)
    df_senti = df_senti[['Job', 'Senti', 'Skill']]
    df_senti = df_senti.sort_values(by='Senti', ascending=False)

    return df_senti

def TF_IDF_skills(job, job_description, skills_list):
    # separate the job des into multiple sentences
    nltk.download('punkt')
    sentences = nltk.sent_tokenize(job_description)

    # initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # calculate TF-IDF value
    tfidf_matrix = vectorizer.fit_transform(sentences)

    keyword_tfidf_scores = {}
    for keyword in skills_list:
        # if skill in vectorizer
        idx = vectorizer.vocabulary_.get(keyword)
        if idx is not None:
            # if has, get TF-IDF value
            keyword_tfidf_scores[keyword] = round(tfidf_matrix.getcol(idx).sum(), 3)

    df_tfidf = pd.DataFrame(list(keyword_tfidf_scores.items()), columns=['Skill', 'tfidf'])
    df_tfidf.insert(1, 'Job', job)
    df_tfidf = df_tfidf[['Job', 'tfidf', 'Skill']]
    df_tfidf = df_tfidf.sort_values(by='tfidf', ascending=False)

    return df_tfidf
