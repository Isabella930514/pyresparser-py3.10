import sys
import config, extract_triples, kg_predication


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

    # ---- flexible extract triples from text and kg augmentation-----
    # ---------------------------------------------------
    '''
        original_kb: kb from texts
        extracted_kb: kb from wikidata
        kb: combined kb from original_kb and extracted_kb
        potent_kb: prediction task executed on kb, the prediction result
    '''
    original_kb, extracted_kb, kb, if_neigh = extract_triples.from_text_to_kb(config.TEXT_FILE, config.EXTRACTOR_TYPE,
                                                                              config.IF_NEIGHBOUR,
                                                                              config.EXPEND_NUM, config.endpoint_url,
                                                                              config.MAX_NEIGH)
    # ---------- knowledge graph prediction by specific node and rel --------
    kg_predication.link_predication(config.EXTRACTOR_TYPE,
                                    config.KGE_METHOD_LIST, config.entity,
                                    config.relation,
                                    config.POTENTIAL_ENT_NO, original_kb, extracted_kb, kb, if_neigh)
    print("~~~~~ Prediction Complete ~~~~~")


if __name__ == "__main__":
    main()
