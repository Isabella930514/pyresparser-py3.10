# -*- coding: utf-8 -*-
"""
Created on Wed 2023

@author: Isabel Wang

Scrape jobs from indeed.nz
"""

import random, json
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
import time
import os
import data_preparation
from data_preparation import config

# Number of jobs show on each result page
page_record_limit = 20


def information_address(job_information):
    job_info_dict = {}
    job_list = job_information.strip("{").strip("}").split(',')
    for info in job_list:
        type_list = info.strip().split(':')
        job_info_dict[type_list[0]] = type_list[1]
    cleaned_dict = {key.strip('"'): value.strip('"') for key, value in job_info_dict.items()}

    return cleaned_dict


def get_jobs_info(job_information):
    """
    Scrape from web or read from saved file
    Input:
        search_location - search job in a certain city. Input from commond line.
    Output:
        jobs_info - a list that has info of each job i.e. link, location, title, company, salary, desc
    """

    job_info_dict = information_address(job_information)
    home_dir = os.path.expanduser("~")
    filepath = os.path.join(home_dir, "indeed_jobs_info.json")
    exists = os.path.isfile(filepath)
    if exists:
        with open(filepath, 'r') as fp:
            jobs_info = json.load(fp)
    else:
        jobs_info = web_scrape(job_info_dict)
    return jobs_info


def web_scrape(job_info_dict):
    """
    Scrape jobs from indeed.nz
    When scraping web, be kind and patient
    Input:
        search_location - search job in a certain city. Input from commond line.
    Output:
        jobs_info - a list that has info of each job i.e. link, location, title, company, salary, desc
    """
    # urls of all jobs
    job_links = []
    # Record time for web scraping
    start = time.time()  # start time

    # *** Disable all JS plugins on the site ***
    chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_experimental_option("prefs", {
    #     "profile.managed_default_content_settings.javascript": 2
    # })

    # Launch webdriver
    driver = webdriver.Chrome(options=chrome_options)
    job_locations = [job_info_dict["city"]]

    # *** Extract all job urls ***
    target_job = job_info_dict["jobTitle"].replace(" ", "+")
    for location in job_locations:
        if job_info_dict["site"] == "seek":
            '''seek searching'''
            url = 'https://www.seek.co.nz/' + target_job + '-jobs/in-' + location
        else:
            '''indeed searching'''
            url = 'https://nz.indeed.com/jobs?q=' + target_job + '&l=' + location + '&limit=' + str(
                page_record_limit) + '&fromage=' + str(data_preparation.config.DAY_RANGE)

        # Set timeout
        driver.set_page_load_timeout(150)
        webdriver.DesiredCapabilities.CHROME["unexpectedAlertBehaviour"] = "accept"
        driver.get(url)
        # Be kind and don't hit indeed server so hard
        time.sleep(3)
        for i in range(config.NUM_PAGE):
            if job_info_dict["site"] == "indeed":
                try:
                    # For each job on the page find its url
                    for job_each in driver.find_elements(By.XPATH, '//*[@data-tn-element="jobTitle"]'):
                        job_link = job_each.get_attribute('href')
                        job_links.append({'location': location, 'job_link': job_link})
                    print('scraping {} page {}'.format(location, i + 1))
                    # Go next page
                    next_button = driver.find_element(By.XPATH, '//*[@data-testid="pagination-page-next"]')
                    next_button.click()
                except NoSuchElementException:
                    # If nothing find, we are at the end of all returned results
                    print("{} finished".format(location))
                    break
            else:
                try:
                    for job_each in driver.find_elements(By.XPATH,
                                                         '//*[@class="y735df0 y735dff  y735df0 y735dff _1iz8dgs5i _1iz8dgsj _1iz8dgsk _1iz8dgsl _1iz8dgsm _1iz8dgs7"]'):
                        job_link = job_each.get_attribute('href')
                        job_links.append({'location': location, 'job_link': job_link})
                    print('scraping {} page {}'.format(location, i + 1))
                    next_button = driver.find_element(By.XPATH, '//a[@aria-label="Next"]')
                    next_button.click()
                except NoSuchElementException as e:
                    print(f"{location} finished - error: {e}")
                    break
            time.sleep(3)
    # Write all jobs links to a json file so it can be reused later
    home_dir = os.path.expanduser("~")
    filepath = os.path.join(home_dir, "indeed_jobs_links.json")
    with open(filepath, 'w') as fp:
        json.dump(job_links, fp)

    # ***Go through each job url and gather detailed job info ***
    # Info of all jobs
    jobs_info = []
    for job_lk in job_links:
        # Make some random wait time between each page so we don't get banned
        m = random.randint(1, 5)
        time.sleep(m)
        # Retrieve single job url
        link = job_lk['job_link']
        driver.get(link)
        # Job city and province
        location = job_lk['location']

        if job_info_dict["site"] == "seek":
            # Job title
            title = driver.find_element(By.XPATH, '//h1[@data-automation="job-detail-title"]').text
            # Job company
            company = driver.find_element(By.XPATH, '//*[@data-automation="advertiser-name"]').text
            # Job description
            desc = driver.find_element(By.XPATH, '//*[@data-automation="jobAdDetails"]').text
            jobs_info.append({'link': link, 'location': location, 'title': title, 'company': company, 'desc': desc})
        else:
            title = driver.find_element(By.CLASS_NAME,
                                        '//*[@class="icl-u-xs-mb--xs icl-u-xs-mt--none jobsearch-JobInfoHeader-title"]').text
            company = driver.find_element(By.XPATH, '//*[@class="icl-u-lg-mr--sm icl-u-xs-mr--xs"]').text
            desc = driver.find_element(By.ID, '//*[@class="jobsearch-JobComponent-description icl-u-xs-mt--md"]').text
            jobs_info.append({'link': link, 'location': location, 'title': title, 'company': company, 'desc': desc})

    # Write all jobs info to a json file so it can be re-used later
    filepath = os.path.join(home_dir, "indeed_jobs_info.json")
    with open(filepath, 'w') as fp:
        json.dump(jobs_info, fp)

    # Close and quit webdriver
    driver.quit()
    end = time.time()  # end time
    # Calculate web scaping time
    scaping_time = (end - start) / 60.
    print('Took {0:.2f} minutes scraping {1:d} jobs'.format(scaping_time, len(jobs_info)))
    return jobs_info
