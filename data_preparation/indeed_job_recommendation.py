from data_preparation import config
from data_preparation import extract_triples, kg_predication


def convert_kg(file_path, augment):
    # ---- flexible extract triples from text and kg augmentation-----
    # ---------------------------------------------------
    """
        original_kb: kb from texts
        extracted_kb: kb from wikidata
        kb: combined kb from original_kb and extracted_kb
        potent_kb: prediction task executed on kb, the prediction result
    """
    extract_triples.from_text_to_kb(file_path, config.EXTRACTOR_TYPE, augment, config.EXPEND_NUM, config.endpoint_url, config.MAX_NEIGH)
    # ---------- knowledge graph prediction by specific node and rel --------
    # kg_predication.link_predication(config.EXTRACTOR_TYPE,
    #                                 config.KGE_METHOD_LIST, config.entity,
    #                                 config.relation,
    #                                 config.POTENTIAL_ENT_NO, original_kb, extracted_kb, kb, if_neigh)
    print("~~~~~ Prediction Complete ~~~~~")

