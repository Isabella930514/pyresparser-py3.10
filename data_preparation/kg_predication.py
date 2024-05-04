import argparse
import optuna
import time
import os
import pandas as pd
from data_preparation.trainer import Trainer
from data_preparation.tester import Tester
from data_preparation.predictor import Predictor
from data_preparation.dataset import Dataset

best_epoch_dict = {}


class POTENT_KB:

    def __init__(self):
        self.entities = {}
        self.relations = []

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


def objective(trial, dataset, kge_model):
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    ne = trial.suggest_categorical('ne', [100, 200, 300, 400, 500])
    emb_dim = trial.suggest_categorical('emb_dim', [100, 200, 300])
    reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    neg_ratio = trial.suggest_categorical('neg_ratio', [1, 2, 3])

    args = argparse.Namespace(
        ne=ne,
        lr=lr,
        model=kge_model,
        reg_lambda=reg_lambda,
        emb_dim=emb_dim,
        neg_ratio=neg_ratio,
        batch_size=batch_size,
        save_each=100
    )

    trainer = Trainer(dataset, args, kge_model, True)
    valid_loss = trainer.train(kge_model)
    return valid_loss


def run_optuna(dataset, kge_model, n_trials=50):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, dataset, kge_model), n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return trial.params


def construct_potential_kb(potential_tuples, kb):
    potent_kb = POTENT_KB()
    entity_list = set([tuple[2] for tuple in potential_tuples])
    entity_list.update(set([tuple[0] for tuple in potential_tuples]))
    for entity in entity_list:
        for key, value in kb.entities.items():
            if key == entity:
                potent_kb.entities[key] = value
    for line in potential_tuples:
        potent_kb.add_relation(line[0], line[1], line[2], [{'start': 0, 'end': 0}])

    return potent_kb


def link_predication(nlp_model, kge_model_list, entity, relation, number, original_kb, extracted_kb, kb):
    dataset = Dataset(nlp_model)

    # best_model_path = "models/REBEL/TransH/100.chkpnt"
    # if os.path.exists(best_model_path):
    #     print("~~~ Prediction A~~~")
    #     predictor = Predictor(dataset, best_model_path)
    #     final_tuples = predictor.predict(nlp_model, entity, relation, number)
    #     with open(f"./datasets/{nlp_model}/predicated_result_kg.csv", "w") as file:
    #         for item in final_tuples:
    #             file.write(str(item)+'\n')
    #     potent_kb = construct_potential_kb(final_tuples, kb)
    #     return extracted_kb, original_kb, potent_kb
    # else:
    for kge_model in kge_model_list:
        # ----- Bayesian Optimization Super Parameter-----#
        best_params = run_optuna(dataset, kge_model, 50)

        args = argparse.Namespace(**best_params, model=kge_model, dataset=nlp_model, save_each=100)

        pd.DataFrame.from_dict(data=dataset.ent2id, orient='index').to_csv(f"./datasets/{nlp_model}/entity_id.csv",
                                                                           header=False)
        pd.DataFrame.from_dict(data=dataset.rel2id, orient='index').to_csv(f"./datasets/{nlp_model}/rel_id.csv",
                                                                           header=False)

        print(f"-----train data size: {len(dataset.data['train'])}")
        print(f"-----valid data size: {len(dataset.data['valid'])}")
        print(f"-----test data size: {len(dataset.data['test'])}")
        print(f"-----entity size: {len(dataset.ent2id)}")
        print(f"-----relation size: {len(dataset.rel2id)}")
        print("~~~~ Training ~~~~")
        trainer = Trainer(dataset, args, kge_model, False)
        trainer.train(kge_model)

        print("~~~~ Select best epoch on validation set ~~~~")
        epochs2test = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
        dataset = Dataset(args.dataset)

        best_mrr = -1.0
        best_epoch = "0"
        for epoch in epochs2test:
            start = time.time()
            print(epoch)
            model_path = "models/" + args.dataset + "/" + kge_model + "/" + epoch + ".chkpnt"
            tester = Tester(dataset, model_path, "valid")
            mrr = tester.test(nlp_model, kge_model)
            if mrr > best_mrr:
                best_mrr = mrr
                best_epoch = epoch
            print(time.time() - start)

        print("Best epoch: " + best_epoch)
        best_epoch_dict[kge_model] = best_epoch

        print("~~~~ Testing on the best epoch ~~~~")
        best_model_path = "models/" + args.dataset + "/" + kge_model + "/" + best_epoch + ".chkpnt"
        tester = Tester(dataset, best_model_path, "test")
        tester.test(nlp_model, kge_model)

    with open(f"./datasets/{nlp_model}/model_comparision.csv", "r") as file:
        models_info = file.readlines()
    model_dict = {}
    for line in models_info:
        model_list = line.strip().split(',')
        model_dict[model_list[0]] = model_list[6]
    outperform_model = [key for key, value in model_dict.items() if value == max(model_dict.values())][0]
    print(f"~~~ The outperform model is {outperform_model} w best epoch {best_epoch_dict[outperform_model]} ~~~")

    print("~~~ Prediction ~~~")
    best_model_path = "models/" + nlp_model + "/" + outperform_model + "/" + best_epoch_dict[
        outperform_model] + ".chkpnt"
    print(f"the best model path is : {best_model_path}")
    predictor = Predictor(dataset, best_model_path)
    final_tuples = predictor.predict(nlp_model, entity, relation, number)
    with open(f"./datasets/{nlp_model}/predicated_result_kg.csv", "w") as file:
        for item in final_tuples:
            file.write(str(item) + '\n')
    potent_kb = construct_potential_kb(final_tuples, kb)
    return extracted_kb, original_kb, potent_kb
