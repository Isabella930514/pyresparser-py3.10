# pip install sparqlwrapper
# https://rdflib.github.io/sparqlwrapper/
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests


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

    def combine(self, kb):
        for key, value in kb.entities.items():
            if key not in self.entities:
                self.entities[key] = value
            else:
                pass
        self.relations += kb.relations


def get_results(endpoint_url, query):

    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def convert_format(ex_kb, results, key_label):
    endpoint_url = "https://query.wikidata.org/sparql"

    def get_property_label(property_id):

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
            ex_kb.add_entity(object_uri, object_label)
            ex_kb.add_relation(key_label, get_property_label(property_uri.split('/')[-1]), object_label, [{'start': 0, 'end': 0}])
        else:
            continue


def get_wikidata_id(wikipedia_url):
    response = requests.get(wikipedia_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    wikidata_link = soup.find("li", {"id": "t-wikibase"})
    if wikidata_link and wikidata_link.a:
        wikidata_url = wikidata_link.a.get('href')
        return wikidata_url.split('/')[-1]
    else:
        return "Wikidata ID not found."


def load_data(kb):
    endpoint_url = "https://query.wikidata.org/sparql"
    ex_kb = EX_KB()
    for key_label, value in tqdm(kb.entities.items()):
        entity_link = value['url']
        entity_key = get_wikidata_id(entity_link)
        query = f"""
        SELECT DISTINCT ?subjectLabel ?subject ?property ?object ?objectLabel WHERE {{
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
        results = list(results["results"]["bindings"])[:5]

        convert_format(ex_kb, results, key_label) # get ex_kb from wikidata
        ex_kb.combine(kb) # combine kb with ex_kb
    return ex_kb