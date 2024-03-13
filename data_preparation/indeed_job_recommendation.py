import sys
import config, web_scrapper, job_skill_graph, extract_triples, kg_predication


class POTENT_KB():

    def __init__(self):
        self.entities = {}
        self.relations = []

    def relation_exists(self, head, relation_type, tail):
        for rel in self.relations:
            if rel['head'] == head and rel['type'] == relation_type and rel['tail'] == tail:
                return True
        return False

    def add_relation(self, head, relation_type, tail, spans):
        if not self.relation_exists(head, relation_type, tail):
            self.relations.append({
                'head': head,
                'type': relation_type,
                'tail': tail,
                'meta': {'spans': spans}
            })


def main():
    # If city included, only search and recommend jobs in the city
    location = ''
    if len(sys.argv) > 1:
        # Check if input city name matches our pre-defined list
        if sys.argv[1] in config.JOB_LOCATIONS:
            location = sys.argv[1]
        else:
            sys.exit('*** Please try again. *** \nEither leave it blank or input a city from this list:\n{}'.format(
                '\n'.join(config.JOB_LOCATIONS)))
    # ---------------------------------------------------
    # ---- Scrape from web or read from local saved -----
    # ---------------------------------------------------
    # jobs_info = web_scrapper.get_jobs_info(location)
    # ---------------------------------------------------
    # -------- job and skills graph construction ----------
    # ---------------------------------------------------
    # j_s_graph = job_skill_graph.job_skill_graph_def(jobs_info)
    # ---- flexible extract triples from text and kg augmentation-----
    # ---------------------------------------------------
    '''
        original_kb: kb from texts
        extracted_kb: kb from wikidata
        kb: combined kb with original_kb and extracted_kb
        potent_kb: prediction task executed on kb, the prediction result
    '''
    original_kb, extracted_kb, kb, if_neigh = extract_triples.from_text_to_kb(config.TEXT_FILE, config.EXTRACTOR_TYPE,
                                                                              config.IF_NEIGHBOUR,
                                                                              config.EXPEND_NUM, config.endpoint_url,
                                                                              config.MAX_NEIGH)
    # ---------------------------------------------------
    potential_tuples = kg_predication.link_predication(config.EXTRACTOR_TYPE, config.ENTITY_EMBEDDING_DIM,
                                                       config.KGE_METHOD, config.entity, config.relation,
                                                       config.POTENTIAL_ENT_NO)
    potent_kb = POTENT_KB()
    entity_list = set([tuple[2] for tuple in potential_tuples])
    entity_list.update(set([tuple[0] for tuple in potential_tuples]))
    for entity in entity_list:
        for key, value in kb.entities.items():
            if key == entity:
                potent_kb.entities[key] = value
    for line in potential_tuples:
        potent_kb.add_relation(line[0], line[1], line[2], [{'start': 0, 'end': 0}])
    extract_triples.save_network_html(extracted_kb, original_kb, potent_kb, if_neigh,
                                      f"network_{config.EXTRACTOR_TYPE}.html", config.IF_PRE)
    print("")


if __name__ == "__main__":
    main()
