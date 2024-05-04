import torch
import numpy as np
from data_preparation import config


class Predictor:
    def __init__(self, dataset, model_path):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        self.dataset = dataset
        self.all_facts_as_set_of_tuples = set(self.allFactsAsTuples())

    def get_id(self, nlp_model, entity_pre, relation_pre):
        with open(f"./datasets/{nlp_model}/entity_id.csv", "r") as entity_file:
            entities = entity_file.readlines()
        entities_list = [entity.strip().split(',') for entity in entities]
        entity_id = [entity[1] for entity in entities_list if entity[0] == entity_pre][0]

        with open(f"./datasets/{nlp_model}/rel_id.csv", "r") as rel_file:
            rels = rel_file.readlines()
        rels_list = [rel.strip().split(',') for rel in rels]
        rel_id = [rel[1] for rel in rels_list if rel[0] == relation_pre][0]

        return int(entity_id), int(rel_id)

    def get_value(self, nlp_model, find_entity):
        with open(f"./datasets/{nlp_model}/entity_id.csv", "r") as entity_file:
            entities = entity_file.readlines()
        entities_list = [entity.strip().split(',') for entity in entities]
        entity_value = [entity[0] for entity in entities_list if int(entity[1]) == find_entity][0]
        return entity_value

    def allFactsAsTuples(self):
        tuples = []
        for spl in self.dataset.data:
            for fact in self.dataset.data[spl]:
                tuples.append(tuple(fact))

        return tuples

    def create_queries(self, fact, head_or_tail_or_link):
        head, rel, tail = fact
        if head_or_tail_or_link == "head":
            return [(i, rel, tail) for i in range(self.dataset.num_ent())]
        elif head_or_tail_or_link == "tail":
            return [(head, rel, i) for i in range(self.dataset.num_ent())]
        elif head_or_tail_or_link == "link":
            return [(head, i, tail) for i in range(self.dataset.num_rel())]

    def add_fact_and_shred(self, fact, queries, raw_or_fil):
        if raw_or_fil == "raw":
            result = queries
        elif raw_or_fil == "fil":
            result = list(set(queries) - self.all_facts_as_set_of_tuples)

        return self.shred_facts(result)

    def shred_facts(self, triples):
        heads = [triples[i][0] for i in range(len(triples))]
        rels = [triples[i][1] for i in range(len(triples))]
        tails = [triples[i][2] for i in range(len(triples))]
        return torch.LongTensor(heads).to(self.device), torch.LongTensor(rels).to(self.device), torch.LongTensor(
            tails).to(self.device)

    def find_potential_tuples(self, sim_scores, queries, number, scenario):

        largest_indices = []
        for _ in range(min(number, len(sim_scores))):
            max_index = 0
            max_value = sim_scores[0]

            for i in range(1, len(sim_scores)):
                if sim_scores[i] > max_value and i not in largest_indices:
                    max_index = i
                    max_value = sim_scores[i]
            # potential tuples index list
            largest_indices.append(max_index)

        potential_tuples = []
        for index in largest_indices:
            if scenario == "link":
                potential_tuples.append({
                    "tuple": queries[index],
                    "similarity_score": sim_scores[index]
                })
            else:
                target_query = queries[index]
                if len(set(target_query)) == len(target_query):
                    potential_tuples.append(queries[index])
        return potential_tuples

    def predict(self, nlp_model, entity_pre, relation_pre, number):
        entity_id, rel_id = self.get_id(nlp_model, entity_pre, relation_pre)
        scenarios = ["head", "tail", "link"]
        final_tuples = []
        missing_component = 0.0

        for scenario in scenarios:
            if scenario in ["head", "tail"]:
                fact = np.array([missing_component, rel_id, entity_id]) if scenario == "head" else np.array([entity_id, rel_id, missing_component])
                queries = self.create_queries(fact, scenario)
                h_entities, relations, t_entities = self.add_fact_and_shred(fact, queries, "fil")
                sim_scores = self.model(h_entities, relations, t_entities).cpu().data.numpy()
                potential_tuples_id = self.find_potential_tuples(sim_scores, queries, number, scenario)
                for tuple_id in potential_tuples_id:
                    if scenario == "head":
                        entity_value = self.get_value(nlp_model, tuple_id[0])
                    else:
                        entity_value = self.get_value(nlp_model, tuple_id[2])
                    if entity_value == entity_pre or relation_pre == entity_pre or relation_pre == entity_value:
                        continue
                    tuple_item = [entity_value, relation_pre, entity_pre] if scenario == "head" else [entity_pre, relation_pre, entity_value]
                    final_tuples.append(tuple_item)
            elif scenario == "link":
                potential_tuples_list = []
                for tail in range(self.dataset.num_ent()):
                    if tail == entity_id:
                        continue
                    fact = np.array([entity_id, missing_component, tail])
                    queries = self.create_queries(fact, scenario)
                    h_entities, relations, t_entities = self.add_fact_and_shred(fact, queries, "fil")
                    sim_scores = self.model(h_entities, relations, t_entities).cpu().data.numpy()
                    potential_tuples_list.append(self.find_potential_tuples(sim_scores, queries, number, scenario))

                sorted_tuples = sorted(potential_tuples_list, key=lambda x: x[0]["similarity_score"], reverse=True)
                top_5_tuples = sorted_tuples[:config.MAX_LINK_PRE_NO]

                for item_w_sim in top_5_tuples:
                    data = item_w_sim[0]
                    data_tuple = data['tuple']
                    rel_value = self.get_value(nlp_model, data_tuple[1])
                    tail_value = self.get_value(nlp_model, data_tuple[2])
                    if rel_value == relation_pre:
                        continue
                    tuple_item = [entity_pre, rel_value, tail_value]
                    final_tuples.append(tuple_item)

        seen = {}
        unique_tuples = []
        for item in final_tuples:
            tuple_item = tuple(item)
            if tuple_item not in seen:
                seen[tuple_item] = True
                unique_tuples.append(item)

        return unique_tuples


