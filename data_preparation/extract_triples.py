import itertools
import math
import os
import os.path
import time
import torch
import pandas as pd
import pickle
import spacy
import wikipedia
import IPython
import requests
from tqdm import tqdm
from datetime import datetime as dt
from IPython.display import HTML
from pyvis.network import Network
from spacy.matcher import Matcher
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_preparation import entity_neighbour_extractor as ene
from sklearn.model_selection import train_test_split

pd.set_option('display.max_colwidth', 200)
batch_size = 100
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
nlp = spacy.load('en_core_web_sm')

'''
remove all entities that doesn't have a page on Wikipedia
merge entities if they have the same wikipedia page
'''


def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


class KB:
    def __init__(self):
        self.entities = {}
        self.relations = []
        self.wiki_cache = {}

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
        if candidate_entity in self.wiki_cache:
            return self.wiki_cache[candidate_entity]
        try:
            page = wikipedia.page(candidate_entity, auto_suggest=False)
            entity_data = {
                "title": page.title,
                "url": page.url,
                "summary": ''
            }
            self.wiki_cache[candidate_entity] = entity_data
            return entity_data
        except:
            return None

    def add_entity(self, e):
        self.entities[e["title"]] = {k: v for k, v in e.items() if k != "title"}

    # get unique entity from wikipedia
    def process_entity(self, triple_list):
        entity_list = []
        for triple_l in triple_list:
            for triple_dic in triple_l:
                head = triple_dic["head"]
                tail = triple_dic["tail"]
                entity_list.append(head)
                entity_list.append(tail)

        entities_to_match = set([entity for entity in entity_list if entity not in self.wiki_cache])

        if not entities_to_match:
            # All entities were cached, no need to make a request
            return [self.wiki_cache[entity] for entity in entity_list]

        # batch_process
        chunks = chunked_iterable(entities_to_match, 10)
        for chunk in chunks:
            titles = '|'.join(chunk)
            query_to_title_map = {query: None for query in chunk}

            url = 'https://en.wikipedia.org/w/api.php'
            params = {
                'action': 'query',
                'format': 'json',
                'titles': titles,
                'prop': 'info|extracts',
                'inprop': 'url',
                'exintro': True,
                'explaintext': True,
            }

            response = requests.get(url, params=params)
            if not response.ok:
                return None

            pages = response.json().get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                # cannot find the page
                try:
                    if page_data["title"] == "" or page_data["title"].isdigit():
                        continue
                except:
                    print(page_data)
                title = page_data.get('title', '')
                page_url = page_data.get('fullurl', '')
                summary = page_data.get('extract', '')

                for query, mapped_title in query_to_title_map.items():
                    if mapped_title is None and title.lower() == query.lower():
                        query_to_title_map[query] = title

                entity_data = {
                    "title": title,
                    "url": page_url,
                    "summary": summary
                }

                original_query = next((k for k, v in query_to_title_map.items() if v == title), title)
                self.wiki_cache[original_query] = entity_data

    def add_relation(self, r):
        if r["head"] == r["tail"]:
            return

        candidate_entities = [r["head"], r["tail"]]

        for ent in candidate_entities:
            if ent not in self.wiki_cache.keys():
                return

        for e in candidate_entities:
            e_v = self.wiki_cache[e]
            self.add_entity(e_v)

        entity_detail = self.wiki_cache[candidate_entities[0]]
        r["head"] = entity_detail["title"]
        entity_detail = self.wiki_cache[candidate_entities[1]]
        r["tail"] = entity_detail["title"]

        if not self.exists_relation(r):
            self.relations.append(r)
        else:
            self.merge_relations(r)

    def combine(self, kb):
        for key, value in kb.entities.items():
            if key not in self.entities:
                self.entities[key] = value
            else:
                pass
        self.relations += kb.relations


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

            if tok.dep_.endswith("mod"):
                modifier = tok.text
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            if tok.dep_.find("subj"):
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""

            if tok.dep_.find("obj"):
                ent2 = modifier + " " + prefix + " " + tok.text

            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text

    return [ent1.strip(), ent2.strip()]


def get_relation(sent):
    doc = nlp(sent)

    matcher = Matcher(nlp.vocab)

    pattern = [{'DEP': 'ROOT'},
               {'DEP': 'prep', 'OP': "?"},
               {'DEP': 'agent', 'OP': "?"},
               {'POS': 'ADJ', 'OP': "?"}]

    matcher.add("matching_1", [pattern])

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]]

    return (span.text)


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


def save_network_html(extracted_kb, origin_kb, potent_kb, if_neigh, filename, if_predict):
    net = Network(directed=True, width="2000px", height="1000px", bgcolor="#eeeeee")

    if not if_neigh and if_predict:
        for e in origin_kb.entities:
            net.add_node(e, shape="circle", color="#00FF00")
        for r in origin_kb.relations:
            net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"])
        for ee in potent_kb.entities:
            try:
                net.add_node(ee, shape="circle", color="#e24353", size=5)
            except:
                continue

        for r in potent_kb.relations:
            try:
                net.add_edge(r["head"], r["tail"], title=r["type"], width=5, color="red")
            except:
                continue

    if if_neigh and not if_predict:
        for entity in list(origin_kb.entities.keys()):
            if entity in extracted_kb.entities:
                del extracted_kb.entities[entity]
        for ee in extracted_kb.entities:
            try:
                net.add_node(ee, shape="circle", color="#e03112")
            except:
                continue

        for r in extracted_kb.relations:
            try:
                net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"])
            except:
                continue
    if if_neigh and if_predict:
        for entity in list(origin_kb.entities.keys()):
            if entity in extracted_kb.entities:
                del extracted_kb.entities[entity]
        for ee in extracted_kb.entities:
            try:
                net.add_node(ee, shape="circle", color="#e03112")
            except:
                continue

        for r in extracted_kb.relations:
            try:
                net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"])
            except:
                continue

        for ee in potent_kb.entities:
            try:
                net.add_node(ee, shape="circle", color="#e28743")
            except:
                continue

        for r in potent_kb.relations:
            try:
                net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"], width=2)
            except:
                continue

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


def SPACY_extractor(candidate_sentences):
    entity_pairs = []

    for i in tqdm(candidate_sentences["sentence"]):
        entity_pairs.append(get_entities(i))
    relations = [get_relation(i) for i in tqdm(candidate_sentences["sentence"])]
    source = [i[0] for i in entity_pairs]
    target = [i[1] for i in entity_pairs]
    kg_df = pd.DataFrame({'head': source, 'type': relations, 'tail': target})
    kb = KB()
    for index, row in kg_df.iterrows():
        relation = row.to_dict()
        relation["meta"] = {
            "spans": ""
        }
        kb.add_relation(relation)
    return kb


def REBEL_extractor(model_name, text, span_length):
    # tokenize whole text
    inputs = tokenizer([text], return_tensors="pt")

    # compute span boundaries
    num_tokens = len(inputs["input_ids"][0])
    num_spans = math.ceil(num_tokens / span_length)
    overlap = math.ceil((num_spans * span_length - num_tokens) /
                        max(num_spans - 1, 1))
    spans_boundaries = []

    start = 0
    for i in range(num_spans):
        spans_boundaries.append([start + span_length * i,
                                 start + span_length * (i + 1)])
        start -= overlap

    def batch_inputs(input_ids, attention_mask, batch_size):
        total_samples = input_ids.size(0)

        batched_input_ids = []
        batched_attention_masks = []

        for start_idx in range(0, total_samples, batch_size):
            end_idx = start_idx + batch_size
            batched_input_ids.append(input_ids[start_idx:end_idx])
            batched_attention_masks.append(attention_mask[start_idx:end_idx])

        return batched_input_ids, batched_attention_masks

    # transform input with spans
    tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                  for boundary in spans_boundaries]
    tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                    for boundary in spans_boundaries]
    inputs = {
        "input_ids": torch.stack(tensor_ids),
        "attention_mask": torch.stack(tensor_masks)
    }

    batched_input_ids, batched_attention_masks = batch_inputs(inputs['input_ids'], inputs['attention_mask'], batch_size)

    # generate relations
    num_return_sequences = 3
    gen_kwargs = {
        "max_length": 25,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences
    }

    all_decoded_preds = []
    batch_num = len(batched_input_ids)
    print(f"----total has {batch_num} batches-----")
    print(f"-----start to run {model_name} extractor model-----")

    for index, (input_ids_batch, attention_mask_batch) in enumerate(zip(batched_input_ids, batched_attention_masks)):
        batch_inputs = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch
        }

        start = dt.now()
        generated_tokens = model.generate(**batch_inputs, **gen_kwargs)
        running_secs = (dt.now() - start).seconds
        print(f"-----running time: {running_secs}s for batch {index}-----")

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        all_decoded_preds.extend(decoded_preds)

    kb = KB()

    def process_prediction(sentence_pred):
        current_span_index = sentence_pred[1] // num_return_sequences
        relations = extract_relations_from_model_output(sentence_pred[0])
        results = []
        for relation in relations:
            relation["meta"] = {
                "spans": [spans_boundaries[current_span_index]]
            }
            results.append(relation)
        return results

    relation_list = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_prediction, (pred, i)): i for i, pred in enumerate(all_decoded_preds)}
        # find url and summary for target nodes on wikidata
        for future in tqdm(as_completed(futures), total=len(futures)):
            if len(future.result()) != 0:
                relation_list.append(future.result())
    kb.process_entity(relation_list)
    for relation in relation_list:
        for item in relation:
            kb.add_relation(item)
    return kb


def save_kg_to_csv(kb, file):
    lines = []
    for relation in kb.relations:
        if any(item is None for item in relation.values()):
            continue
        line = relation['head'] + ';' + relation['type'] + ';' + relation['tail'] + '\n'
        lines.append(line)
    with open(file, "w", encoding="utf-8") as kg_file:
        kg_file.writelines(lines)


def split_train_test(file, train_file, test_file, valid_file):
    with open(file, "r", encoding="utf-8") as f:
        data_lines = f.readlines()
    data_tuples = [line.strip().split(';') for line in data_lines]
    df = pd.DataFrame(data_tuples)
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    train_df.to_csv(train_file, index=False, header=False)
    val_df.to_csv(valid_file, index=False, header=False)
    test_df.to_csv(test_file, index=False, header=False)
    print("-----already splited into train, valid, and test datasets-----")


def generate_files(model, if_neigh):
    filename = f"./templates/network.html"
    path = f"./datasets/{model}"
    if not os.path.exists(path):
        os.makedirs(path)
    file = f"./datasets/{model}/kg_{if_neigh}.csv"
    train_file = f"{path}/train.csv"
    test_file = f"{path}/test.csv"
    valid_file = f"{path}/valid.csv"
    return file, filename, train_file, test_file, valid_file


def from_text_to_kb(file, model, if_neigh, expand_num, endpoint_url, max_neigh, span_length=25):
    start_time = time.time()
    directory = "./datasets"

    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f"kb_{model}.pkl")

    if model == 'REBEL':
        if os.path.exists(file_path):
            with open(file_path, "rb") as kb_file:
                kb = pickle.load(kb_file)
        else:
            with open(file, 'r', encoding='utf-8') as file_content:
                lines = file_content.readlines()
                lines = [line.strip() for line in lines[1:]]
                text = ' '.join(lines)
            # generate .pkl file with original kg from text
            kb = REBEL_extractor(model, text, span_length)
            with open(file_path, "wb") as kb_file:
                pickle.dump(kb, kb_file)
        print("-----triples extraction complete-----")

        file, filename, train_file, test_file, valid_file = generate_files(model, if_neigh)

        if if_neigh:
            extracted_kb, kb = ene.load_data(kb, expand_num, endpoint_url, max_neigh, if_neigh)
            save_network_html(extracted_kb, kb, '', if_neigh, filename, False)
            # original_kb is the kb extracted from tests
            original_kb = kb
            # kb means combined_kb now
            kb.combine(extracted_kb)
        else:
            extracted_kb, kb = ene.load_data(kb, expand_num, endpoint_url, max_neigh, if_neigh)
            save_network_html(extracted_kb, kb, '', if_neigh, filename, False)
            original_kb = kb
        save_kg_to_csv(kb, file)
        split_train_test(file, train_file, test_file, valid_file)
        IPython.display.HTML(filename=filename)
        return original_kb, extracted_kb, kb, if_neigh

    if model == 'SPACY':
        if os.path.exists(file_path):
            with open(file_path, "rb") as kb_file:
                kb = pickle.load(kb_file)
        else:
            candidate_sentences = pd.read_csv(file)
            kb = SPACY_extractor(candidate_sentences)
            with open(file_path, "wb") as kb_file:
                pickle.dump(kb, kb_file)
        print("-----triples extraction complete-----")

        file, filename, train_file, test_file, valid_file = generate_files(model, if_neigh)

        if if_neigh:
            extracted_kb, kb = ene.load_data(kb, expand_num, endpoint_url, max_neigh, if_neigh)
            save_network_html(extracted_kb, kb, '', if_neigh, filename, False)
            # orginal_kb is the kb extracted from tests
            original_kb = kb
            # kb means combined_kb now
            kb.combine(extracted_kb)
        else:
            extracted_kb, kb = ene.load_data(kb, expand_num, endpoint_url, max_neigh, if_neigh)
            save_network_html(extracted_kb, kb, '', if_neigh, filename, False)
            original_kb = kb
        save_kg_to_csv(kb, file)
        split_train_test(file, train_file, test_file, valid_file)
        IPython.display.HTML(filename=filename)
        return original_kb, extracted_kb, kb, if_neigh

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"-----kg augmentation complete, {model} running time:{elapsed_time}s-----")
    print(f"-----start to kg predication-----")
