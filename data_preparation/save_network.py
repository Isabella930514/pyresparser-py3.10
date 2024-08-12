from pyvis.network import Network

RAD = "#ed0e19"
GREEN = "#00FF00"
ORENGE = "#e8a71c"


def save_network_html(extracted_kb, origin_kb, potent_kb, if_neigh, filename, if_predict):
    net = Network(directed=True, width="2000px", height="1000px", bgcolor="#eeeeee")
    existing_edges = set()

    if not if_neigh and if_predict:
        for ee in potent_kb.entities:
            net.add_node(ee, shape="circle", color=RAD, size=150)
        for r in potent_kb.relations:
            net.add_edge(r["head"], r["tail"], title=r["type"], width=5, color=RAD)
        for e in origin_kb.entities:
            net.add_node(e, shape="circle", color=GREEN)
        for r in origin_kb.relations:
            net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"])

    if if_neigh and not if_predict:
        if extracted_kb.relations:
            for ee in extracted_kb.entities:
                try:
                    net.add_node(ee, shape="circle", color=ORENGE, size=150)
                except:
                    print(ee)
            for r in extracted_kb.relations:
                net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"], color=ORENGE)
        for e in origin_kb.entities:
            net.add_node(e, shape="circle", color=GREEN)
        for r in origin_kb.relations:
            net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"])

    if if_neigh and if_predict:
        if extracted_kb.relations:
            for ee in extracted_kb.entities:
                net.add_node(ee, shape="circle", color=ORENGE, size=150)
            for r in extracted_kb.relations:
                edge = (r["head"], r["tail"])
                if edge not in existing_edges:
                    net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"], color=ORENGE)
                    existing_edges.add(edge)

        for ee in potent_kb.entities:
            net.add_node(ee, shape="circle", color=RAD, size=150)
        for r in potent_kb.relations:
            edge = (r["head"], r["tail"])
            if edge not in existing_edges:
                net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"], width=5, color=RAD)
                existing_edges.add(edge)

        for e in origin_kb.entities:
            net.add_node(e, shape="circle", color=GREEN)
        for r in origin_kb.relations:
            edge = (r["head"], r["tail"])
            if edge not in existing_edges:
                net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"])
                existing_edges.add(edge)

    if not if_neigh and not if_predict:
        for e in origin_kb.entities:
            net.add_node(e, shape="circle", color=GREEN)
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
    return net


