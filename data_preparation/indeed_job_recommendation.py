import sys
import config, web_scrapper, job_skill_graph


def main():
    # If city included, only search and recommend jobs in the city
    location = ''
    if (len(sys.argv) > 1):
        # Check if input city name matches our pre-defined list
        if (sys.argv[1] in config.JOB_LOCATIONS):
            location = sys.argv[1]
        else:
            sys.exit('*** Please try again. *** \nEither leave it blank or input a city from this list:\n{}'.format(
                '\n'.join(config.JOB_LOCATIONS)))
    # ---------------------------------------------------
    # ---- Scrape from web or read from local saved -----
    # ---------------------------------------------------
    jobs_info = web_scrapper.get_jobs_info(location)
    # ---------------------------------------------------
    # -------- job and skills graph construction ----------
    # ---------------------------------------------------
    j_s_graph = job_skill_graph.job_skill_graph_def(jobs_info)


if __name__ == "__main__":
    main()

