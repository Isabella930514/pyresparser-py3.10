import IPython
from data_preparation import config
from data_preparation import extract_triples, kg_predication
from IPython.display import HTML
from pyvis.network import Network


def save_network_html(extracted_kb, origin_kb, potent_kb, if_neigh, filename, if_predict):
    net = Network(directed=True, width="2000px", height="1000px", bgcolor="#eeeeee")

    if not if_neigh and if_predict:
        for ee in potent_kb.entities:
            net.add_node(ee, shape="circle", color="#ed0e19", size=150)
        for r in potent_kb.relations:
            net.add_edge(r["head"], r["tail"], title=r["type"], width=5, color="#ed0e19")
        for e in origin_kb.entities:
            net.add_node(e, shape="circle", color="#00FF00")
        for r in origin_kb.relations:
            net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"])

    if if_neigh and not if_predict:
        for ee in extracted_kb.entities:
            try:
                net.add_node(ee, shape="circle", color="#e8a71c", size=150)
            except:
                print(ee)
        for r in extracted_kb.relations:
            net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"], color="#e8a71c")
        for e in origin_kb.entities:
            net.add_node(e, shape="circle", color="#00FF00")
        for r in origin_kb.relations:
            net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"])

    if if_neigh and if_predict:
        for ee in extracted_kb.entities:
            net.add_node(ee, shape="circle", color="#e8a71c", size=150)
        for r in extracted_kb.relations:
            net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"], color="#e8a71c")

        for ee in potent_kb.entities:
            net.add_node(ee, shape="circle", color="#ed0e19", size=150)
        for r in potent_kb.relations:
            net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"], width=5, color="#ed0e19")

        for e in origin_kb.entities:
            net.add_node(e, shape="circle", color="#00FF00")
        for r in origin_kb.relations:
            net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"])

    if not if_neigh and not if_predict:
        for e in origin_kb.entities:
            net.add_node(e, shape="circle", color="#00FF00")
        for r in origin_kb.relations:
            net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"])

    net.repulsion(
        node_distance=200,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )

    net.set_edge_smooth('dynamic')
    net.show(filename)


def convert_kg(model, file_path, augment, prediction):
    filename = f"./templates/network.html"
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

    save_network_html(extracted_kb, original_kb, potent_kb, if_neigh, filename, prediction)
    IPython.display.HTML(filename=filename)
    print("~~~~~ Prediction Complete ~~~~~")
