# pip install sparqlwrapper
# https://rdflib.github.io/sparqlwrapper/
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
from bs4 import BeautifulSoup
from itertools import islice
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKER = 64


class EX_KB:

    def __init__(self):
        self.entities = {}
        self.relations = []

    def add_entity(self, url, label):
        if label not in self.entities:
            self.entities[label] = {}
        self.entities[label] = {'url': url, 'summary': ''}

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


def get_results(endpoint_url, query):
    try:
        user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
        sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        return sparql.query().convert()
    except Exception as e:
        print(f"Error querying SPARQL endpoint: {e}")
        return None



def convert_format(ex_kb, results, key_label, endpoint_url):
    label_cache = {}

    def get_property_label(property_id, endpoint_url):

        if property_id in label_cache:
            return label_cache[property_id]

        query = f'''
            SELECT DISTINCT ?propertyLabel WHERE {{
              wd:{property_id} rdfs:label ?propertyLabel .
              FILTER(LANG(?propertyLabel) = "en")
            }}
            '''

        results = get_results(endpoint_url, query)
        results = list(results["results"]["bindings"])
        for result in results:
            property = result['propertyLabel']
            property_value = property['value']
            if property_value:
                label_cache[property_id] = property_value
                return property_value
            else:
                print("No match key for property")
                return property_id

    for item in results:
        property_uri = item['property']['value']
        object_uri = item['object']['value']
        object_label = item['objectLabel']['value']
        object_id = object_uri.split('/')[-1]
        if object_id.startswith('Q') and object_id[1:].isdigit():
            continue
        else:
            ex_kb.add_entity(object_uri, object_label)
            ex_kb.add_relation(key_label, get_property_label(property_uri.split('/')[-1], endpoint_url), object_label,
                               [{'start': 0, 'end': 0}])



def get_wikidata_id(wikipedia_url):
    response = requests.get(wikipedia_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    wikidata_link = soup.find("li", {"id": "t-wikibase"})
    if wikidata_link and wikidata_link.a:
        wikidata_url = wikidata_link.a.get('href')
        return wikidata_url.split('/')[-1]
    else:
        return "Wikidata ID not found."


def load_data(kb, expand_num, endpoint_url, max_neigh):
    ex_kb = EX_KB()
    node_size = len(kb.entities)
    print(f"-----the original kg has {node_size} entities-----")
    print(f"-----expect to expand {expand_num} entities and start to expand-----")

    kb.entities = dict(islice(kb.entities.items(), expand_num))
    node_list = list(kb.entities.keys())
    relations_to_remove = []
    for rel in kb.relations:
        if rel['head'] not in node_list or rel['tail'] not in node_list:
            relations_to_remove.append(rel)
    for rel in relations_to_remove:
        kb.relations.remove(rel)

    with ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
        futures = []
        for key_label, value in kb.entities.items():
            entity_link = value['url']
            futures.append(executor.submit(process_entity, key_label, entity_link, endpoint_url, ex_kb, max_neigh))

        for future in tqdm(as_completed(futures), total=len(futures)):
            pass

    # ex_kb.combine(kb)
    return ex_kb, kb


def process_entity(key_label, entity_link, endpoint_url, ex_kb, max_neigh):
    entity_key = get_wikidata_id(entity_link)
    query = f"""
    SELECT DISTINCT ?subjectLabel ?subject ?property ?object WHERE {{
      VALUES ?subject {{wd:{entity_key}}}
      ?subject ?property ?object .

      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en" .
        ?subject rdfs:label ?subjectLabel .
        ?object rdfs:label ?objectLabel .
      }}
    }}
    """
    results = get_results(endpoint_url, query)
    results = list(results["results"]["bindings"])

    # each node has maximum size of neigh
    if len(results) > max_neigh:
        results = results[:max_neigh]

    convert_format(ex_kb, results, key_label, endpoint_url)