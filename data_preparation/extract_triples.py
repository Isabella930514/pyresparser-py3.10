import math
import torch
import pandas as pd
import spacy
import wikipedia
import IPython
from tqdm import tqdm
from IPython.display import HTML
from pyvis.network import Network
from spacy.matcher import Matcher
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

pd.set_option('display.max_colwidth', 200)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
nlp = spacy.load('en_core_web_sm')

'''
remove all entities that doesn't have a page on Wikipedia
merge entities if they have the same wikipedia page
'''
class KB():
    def __init__(self):
        self.entities = {}
        self.relations = []


    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])


    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)


    def merge_relations(self, r1):
        r2 = [r for r in self.relations
              if self.are_relations_equal(r1, r)][0]
        spans_to_add = [span for span in r1["meta"]["spans"]
                        if span not in r2["meta"]["spans"]]
        r2["meta"]["spans"] += spans_to_add


    def get_wikipedia_data(self, candidate_entity):
        try:
            page = wikipedia.page(candidate_entity, auto_suggest=False)
            entity_data = {
                "title": page.title,
                "url": page.url,
                "summary": page.summary
            }
            return entity_data
        except:
            return None


    def add_entity(self, e):
        self.entities[e["title"]] = {k:v for k,v in e.items() if k != "title"}

    def add_relation(self, r):
        # check on wikipedia
        candidate_entities = [r["head"], r["tail"]]
        entities = [self.get_wikipedia_data(ent) for ent in candidate_entities]

        # if one entity does not exist, stop
        if any(ent is None for ent in entities):
            return

        # manage new entities
        for e in entities:
            self.add_entity(e)

        # rename relation entities with their wikipedia titles
        r["head"] = entities[0]["title"]
        r["tail"] = entities[1]["title"]

        # manage new relation
        if not self.exists_relation(r):
            self.relations.append(r)
        else:
            self.merge_relations(r)

    def print(self):
        print("Entities:")
        for e in self.entities.items():
            print(f"  {e}")
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")


def get_entities(sent):
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""
    prv_tok_text = ""
    prefix = ""
    modifier = ""

    for tok in nlp(sent):
        if tok.dep_ != "punct":
            if tok.dep_ == "compound":
                prefix = tok.text
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""

            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text

    return [ent1.strip(), ent2.strip()]


def get_relation(sent):

  doc = nlp(sent)

  matcher = Matcher(nlp.vocab)

  pattern = [{'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},
            {'POS':'ADJ','OP':"?"}]

  matcher.add("matching_1", [pattern])

  matches = matcher(doc)
  k = len(matches) - 1

  span = doc[matches[k][1]:matches[k][2]]

  return(span.text)


def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations


def save_network_html(kb, type, filename):
    net = Network(directed=True, width="700px", height="700px", bgcolor="#eeeeee")

    color_entity = "#00FF00"
    if type == 'REBEL':
        for e in kb.entities:
            net.add_node(e, shape="circle", color=color_entity)

        for r in kb.relations:
            net.add_edge(r["head"], r["tail"],
                         title=r["type"], label=r["type"])

    else:
        added_nodes = set()

        # Iterate over the rows of the DataFrame
        for index, row in kb.iterrows():
            source = row['source']
            target = row['target']
            edge_label = row['edge']

            # Add the source and target nodes if they haven't been added already
            if source not in added_nodes:
                node_title = f"Node: {source}"
                net.add_node(source, label=source, shape="circle", color="#00FF00", title=node_title)
                added_nodes.add(source)

            if target not in added_nodes:
                net.add_node(target, label=target, shape="circle", color="#00FF00", title=node_title)
                added_nodes.add(target)

            edge_title = f"Edge: {edge_label}"
            net.add_edge(source, target, title=edge_title, label=edge_label)

    net.repulsion(
        node_distance=200,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )

    net.set_edge_smooth('dynamic')
    net.show(filename)


def SPACY_extractor(candidate_sentences):
    entity_pairs = []

    for i in tqdm(candidate_sentences["sentence"]):
        entity_pairs.append(get_entities(i))
    relations = [get_relation(i) for i in tqdm(candidate_sentences["sentence"])]
    source = [i[0] for i in entity_pairs]
    target = [i[1] for i in entity_pairs]
    kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': relations})

    return kg_df


def REBEL_extractor(text, span_length, verbose):
    # tokenize whole text
    inputs = tokenizer([text], return_tensors="pt")

    # compute span boundaries
    num_tokens = len(inputs["input_ids"][0])
    if verbose:
        print(f"Input has {num_tokens} tokens")
    num_spans = math.ceil(num_tokens / span_length)
    if verbose:
        print(f"Input has {num_spans} spans")
    overlap = math.ceil((num_spans * span_length - num_tokens) /
                        max(num_spans - 1, 1))
    spans_boundaries = []
    start = 0
    for i in range(num_spans):
        spans_boundaries.append([start + span_length * i,
                                 start + span_length * (i + 1)])
        start -= overlap
    if verbose:
        print(f"Span boundaries are {spans_boundaries}")

    # transform input with spans
    tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                  for boundary in spans_boundaries]
    tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                    for boundary in spans_boundaries]
    inputs = {
        "input_ids": torch.stack(tensor_ids),
        "attention_mask": torch.stack(tensor_masks)
    }

    # generate relations
    num_return_sequences = 3
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences
    }
    generated_tokens = model.generate(
        **inputs,
        **gen_kwargs,
    )

    decoded_preds = tokenizer.batch_decode(generated_tokens,
                                           skip_special_tokens=False)

    kb = KB()
    i = 0
    for sentence_pred in decoded_preds:
        current_span_index = i // num_return_sequences
        relations = extract_relations_from_model_output(sentence_pred)
        for relation in relations:
            relation["meta"] = {
                "spans": [spans_boundaries[current_span_index]]
            }
            kb.add_relation(relation)
        i += 1

    return kb


def from_text_to_kb(file, type, span_length=25, verbose=True):

    if type == 'REBEL':
        with open(file, 'r', encoding='utf-8') as file_content:
            lines = file_content.readlines()
            lines = [line.strip() for line in lines[1:]]
            text = ' '.join(lines)
        kb = REBEL_extractor(text, span_length, verbose)
        filename = "network_REBEL.html"
        save_network_html(kb, type, filename)
        IPython.display.HTML(filename=filename)

    if type == 'spacy':
        candidate_sentences = pd.read_csv(file)

        kb = SPACY_extractor(candidate_sentences)
        filename = "network_spacy.html"
        save_network_html(kb, type, filename)
        IPython.display.HTML(filename=filename)
