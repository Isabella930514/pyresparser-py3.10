import config
from data_preparation import extract_triples, kg_predication
from data_preparation import save_network
from IPython.display import HTML
import IPython


def convert_kg(model, file_path, augment, prediction):
    filename = "./templates/template.html"
    # ---- flexible extract triples from text and kg augmentation-----
    """
        original_kb: kb from texts
        extracted_kb: kb from wikidata
        kb: combined kb from original_kb and extracted_kb
        potent_kb: prediction task executed on kb, the prediction result
    """
    original_kb, extracted_kb, potent_kb, kb, if_neigh = extract_triples.from_text_to_kb(model, file_path, augment,
                                                                                         config.EXPEND_NUM,
                                                                                         config.endpoint_url,
                                                                                         config.MAX_NEIGH)

    # ---------- knowledge graph prediction by specific node and rel --------
    if prediction:
        extracted_kb, original_kb, potent_kb = kg_predication.link_predication(model, config.KGE_METHOD_LIST, config.entity,
                                                                               config.relation,
                                                                               config.POTENTIAL_ENT_NO, original_kb,
                                                                               extracted_kb, kb)

    # ---------- save knowledge graph into a html file --------
    net = save_network.save_network_html(extracted_kb, original_kb, potent_kb, if_neigh, filename, prediction)
    net.show(filename)
    IPython.display.HTML(filename=filename)
    print("~~~~~ Prediction Complete ~~~~~")