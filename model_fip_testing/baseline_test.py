import datetime
import sys
import os
import csv
import gc
import gzip
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools


sys.path.insert(0, os.path.abspath('..'))

# Competitor models
from forgotten_items.competitors.top import Top
from forgotten_items.competitors.last import Last
from forgotten_items.competitors.clf import CLF
from forgotten_items.competitors.ibp import IntervalBasedPredictor
from forgotten_items.competitors.tbp import TBP
from forgotten_items.competitors.markovchain import MarkovChain
from forgotten_items.competitors.nmf import NMF
from forgotten_items.competitors.hrm import HRM
from forgotten_items.competitors.fpmc import FPMC
# ------------------------------------------------------------------------

# Contender models
from forgotten_items.contenders.xmt import XMT
from forgotten_items.contenders.txmt import TXMT


# ------------------------------------------------------------------------

from forgotten_items.imports.utilities.models import *
from forgotten_items.imports.utilities.data_management import *
from forgotten_items.imports.evaluation.evaluation_measures import *
from forgotten_items.imports.evaluation.calculate_aggregate_statistics import *
from forgotten_items.imports.utilities.cat_map import cod_mkt_cat2name
from forgotten_items.imports.utilities.cat_remap import remap_categories

def cleanup():
    global all_customers_baskets, items, customers_train_set, customers_test_set, performance

    # Clear large data structures
    if 'all_customers_baskets' in globals():
        all_customers_baskets.clear()
    if 'items' in globals():
        items.clear()
    if 'customers_train_set' in globals():
        customers_train_set.clear()
    if 'customers_test_set' in globals():
        customers_test_set.clear()
    if 'performance' in globals():
        performance.clear()

    # Delete any remaining model instances
    for var in list(globals()):
        if isinstance(globals()[var], (Top, Last, CLF, MarkovChain, IntervalBasedPredictor, HRM, NMF, FPMC, TBP)):
            del globals()[var]

    # Force garbage collection
    gc.collect()


def predict_with_model(model, customer_id, customer_data, next_baskets, new2old,
                       dataset_name, coop_level, test_partition_type, real_id, model_name,
                       pred_length, training_time):
    """
    Predicts next basket items using various model types while preserving all diagnostic prints
    and core prediction logic.
    """
    print("pred_length for predictions: ", pred_length)
    for i in range(len(next_baskets) - 1):
        # Getting the current basket date
        current_basket_date, current_basket_data = next_baskets[i]
        print(current_basket_date)

        # Extracting the current basket
        current_basket = set(current_basket_data['basket'].keys())
        print(current_basket)

        # Calculate the required prediction length
        required_length = len(current_basket) + pred_length

        print(f"\nProcessing basket at date: {current_basket_date}")
        print(f"Current basket size: {len(current_basket)}")

        # Get future baskets for evaluation
        future_basket_date, future_basket_data = next_baskets[i + 1]
        print(f'future_basket_date {future_basket_date}')

        future_basket = set(future_basket_data['basket'].keys())
        print(f'future_basket {future_basket}')

        if isinstance(model, Top):
            print("pred_basket")
            pred_start_time = datetime.datetime.now()
            pred_basket = model.predict(pred_length=required_length)
            pred_end_time = datetime.datetime.now()
            pred_basket = set(cod_mkt_cat2name.get(new2old[item], new2old[item]) for item in pred_basket)
            pred_basket = pred_basket - set(item for item in current_basket if item in pred_basket)
            pred_basket = set(list(pred_basket)[:pred_length])


        elif isinstance(model, Last):
            pred_start_time = datetime.datetime.now()
            pred_basket = model.predict()
            pred_end_time = datetime.datetime.now()
            pred_basket = set(cod_mkt_cat2name.get(new2old[item], new2old[item]) for item in pred_basket)
            pred_basket = pred_basket - set(item for item in current_basket if item in pred_basket)
            pred_basket = set(list(pred_basket)[:pred_length])

        elif isinstance(model, CLF):
            pred_start_time = datetime.datetime.now()
            pred_basket = model.predict(pred_length=required_length)
            pred_end_time = datetime.datetime.now()
            pred_basket = set(cod_mkt_cat2name.get(new2old[item], new2old[item]) for item in pred_basket)
            pred_basket = pred_basket - set(item for item in current_basket if item in pred_basket)
            pred_basket = set(list(pred_basket)[:pred_length])

        elif isinstance(model, MarkovChain):
            pred_start_time = datetime.datetime.now()
            pred_basket = model.predict(pred_length=required_length)
            pred_end_time = datetime.datetime.now()
            pred_basket = set(cod_mkt_cat2name.get(new2old[item], new2old[item]) for item in pred_basket)
            pred_basket = pred_basket - set(item for item in current_basket if item in pred_basket)
            pred_basket = set(list(pred_basket)[:pred_length])

        elif isinstance(model, IntervalBasedPredictor):
            pred_start_time = datetime.datetime.now()
            pred_basket_list, pred_set, poss_forgotten = model.predict_basket(customer_id,
                                                                              visit_date=current_basket_date,
                                                                              k=required_length)
            pred_end_time = datetime.datetime.now()
            pred_time = pred_end_time - pred_start_time
            pred_set = set(cod_mkt_cat2name.get(new2old[item], new2old[item]) for item in pred_set)
            pred_set = pred_set - set(item for item in current_basket if item in pred_set)
            pred_set = set(list(pred_set)[:pred_length])

            performance = []
            nb = 0  # next basket counter
            b = 0  # basket counter for individual scores

            evaluation = evaluate_prediction(future_basket, pred_set)
            performance.append(evaluation)

            nb += 1
            b += 1
            return pred_set, performance, nb, b

        elif isinstance(model, HRM):
            if customer_data:
                sorted_keys = sorted(customer_data.keys())
                last_key = sorted_keys[-1]
                last_basket = customer_data[last_key]['basket']
            else:
                last_basket = {}

            pred_start_time = datetime.datetime.now()
            pred_basket = model.predict(user_id=customer_id, last_basket=list(last_basket.keys()),
                                        pred_length=required_length)
            pred_end_time = datetime.datetime.now()
            customer_id = real_id
            print(customer_id)
            pred_basket = set(cod_mkt_cat2name.get(new2old[item], new2old[item]) for item in pred_basket)
            pred_basket = pred_basket - set(item for item in current_basket if item in pred_basket)
            pred_basket = set(list(pred_basket)[:pred_length])

        elif isinstance(model, FPMC):
            if customer_data:
                sorted_keys = sorted(customer_data.keys())
                last_key = sorted_keys[-1]
                last_basket = customer_data[last_key]['basket']
            else:
                last_basket = {}

            pred_start_time = datetime.datetime.now()
            pred_basket = model.predict(user_id=customer_id, last_basket=list(last_basket.keys()),
                                        pred_length=required_length)
            pred_end_time = datetime.datetime.now()
            customer_id = real_id
            print(customer_id)
            pred_basket = set(cod_mkt_cat2name.get(new2old[item], new2old[item]) for item in pred_basket)
            pred_basket = pred_basket - set(item for item in current_basket if item in pred_basket)
            pred_basket = set(list(pred_basket)[:pred_length])

        elif isinstance(model, NMF):
            pred_start_time = datetime.datetime.now()
            pred_basket = model.predict(user_id=customer_id, pred_length=required_length)
            pred_end_time = datetime.datetime.now()
            customer_id = real_id
            print(customer_id)
            pred_basket = set(cod_mkt_cat2name.get(new2old[item], new2old[item]) for item in pred_basket)
            pred_basket = pred_basket - set(item for item in current_basket if item in pred_basket)
            pred_basket = set(list(pred_basket)[:pred_length])

        elif isinstance(model, TBP):
            day_of_next_purchase = datetime.datetime.strptime(current_basket_date[0:10], '%Y_%m_%d')
            pred_start_time = datetime.datetime.now()
            pred_basket = model.predict(customer_data, day_of_next_purchase, nbr_patterns=None,
                                        pred_length=required_length)
            pred_end_time = datetime.datetime.now()
            pred_basket = set(cod_mkt_cat2name.get(new2old[item], new2old[item]) for item in pred_basket)
            pred_basket = pred_basket - set(item for item in current_basket if item in pred_basket)
            pred_basket = set(list(pred_basket)[:pred_length])

        elif isinstance(model, XMT):
            pred_start_time = datetime.datetime.now()
            pred_basket, explanations = model.predict_f(current_basket, current_basket_date, pred_length)
            pred_end_time = datetime.datetime.now()

            for item, explanation in zip(pred_basket, explanations):
                print()

        elif isinstance(model, TXMT):
            pred_start_time = datetime.datetime.now()
            pred_basket, explanations = model.predict_f(current_basket, current_basket_date, pred_length)
            pred_end_time = datetime.datetime.now()

            for item, explanation in zip(pred_basket, explanations):
                print()

        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        pred_time = pred_end_time - pred_start_time
        performance = []
        nb = 0  # next basket counter
        b = 0  # basket counter for individual scores

        evaluation = evaluate_prediction(future_basket, pred_basket)
        performance.append(evaluation)

        nb += 1
        b += 1
        return pred_basket, performance, nb, b
def preprocess_data(dataset_name, coop_level, test_partition_type, model_name=None, pred_length=None):
    print(f"Reading dataset: {dataset_name}")

    def get_dataset_path():
        """Get the path to the datasets directory relative to this script."""
        # This script is in baseline_testing/, so we need to go up one level to reach datasets/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(os.path.dirname(current_dir), 'datasets')
        return dataset_dir

    dataset_dir = get_dataset_path()
    path1 = os.path.join(dataset_dir, 'coop100.json')
    path2 = os.path.join(dataset_dir, 'tafeng.json')


    if dataset_name == 'tafeng':
        dataset = read_data(path2)
        item2category = None
    elif dataset_name == 'coop':
        # Comment out if NOT using the full dataset
        dataset = read_data(path1)
        if coop_level == 'category':
            item2category = get_item2category('market.csv', category_index['categoria'])
        else:
            item2category = None
    else:
        print(datetime.datetime.now(), 'Unknown dataset')
        return None

    print("Dataset read")
    print(datetime.datetime.now(), 'Customers', len(dataset))
    # print(dataset)

    if isinstance(test_partition_type, int):
        split_mode = str(test_partition_type)
    elif test_partition_type == 'fixed':
        split_mode = 'loo'
    elif test_partition_type == 'random':
        split_mode = 'rnd'
    elif test_partition_type == 'standard':
        split_mode = '70'
    elif test_partition_type == 'last4':
        split_mode = '-4'
    else:
        print(datetime.datetime.now(), 'Unknown test partition type')
        return None

    print(datetime.datetime.now(), f'Partition dataset into train / test ({split_mode})')
    customers_train_set, customers_test_set = split_train_test(dataset,
                                                               split_mode=split_mode,
                                                               min_number_of_basket=10,
                                                               min_basket_size=1,
                                                               max_basket_size=float('inf'),
                                                               min_item_occurrences=2,
                                                               item2category=item2category,
                                                               large_basket=large_basket,
                                                               max_days=max_days,
                                                               min_forgotten_items=min_forgotten_items)

    customers_test_set = remap_categories(customers_test_set, cod_mkt_cat2name)
    customers_train_set = remap_categories(customers_train_set, cod_mkt_cat2name)

    customers_train_set, new2old, old2new = remap_items_with_data(customers_train_set)

    print(datetime.datetime.now(), 'Customers for test', len(customers_train_set),
          '%.2f%%' % (100.0 * len(customers_train_set) / len(dataset)))

    # del for numbered datasets
    # del dataset, old2new, item2category, test_partition_type, coop_number

    # del for full coop / tafeng
    del dataset, old2new, item2category, test_partition_type
    gc.collect()
    return customers_train_set, customers_test_set, new2old


def run_experiment(customers_train_set, customers_test_set, new2old,
                       dataset_name, coop_level, test_partition_type, model_name,
                       pred_length=5, n_factor=100):

    print(datetime.datetime.now(), 'Creating and building models')
    start_time = datetime.datetime.now()
    # List of baskets for HRM() and FPMC()
    all_customers_baskets = []
    # Pred_length and performances for TBP
    performance = defaultdict(list)

    # pred_lengths for experiments
    min_pred_length = pred_length  # 2
    max_pred_length = 11 # 21
    pred_lengths = list(range(min_pred_length, max_pred_length))


    if model_name == "hrm":
        for customer_id in customers_train_set:
            customer_baskets = data2baskets(customers_train_set[customer_id])
            all_customers_baskets.append(customer_baskets)
        items = get_items(all_customers_baskets)
        # Number of epochs can be adjusted as needed
        hrm = HRM(n_user=len(customers_train_set), n_item=len(items), u_dim=50, v_dim=50, verbose=True)
        hrm.build_model(all_customers_baskets)

        end_time = datetime.datetime.now()
        training_time = end_time - start_time
        print(datetime.datetime.now(), f'Model {model_name}, pred_length {pred_length} built in', end_time - start_time)
        print(datetime.datetime.now(), 'Perform predictions')
        for i, customer_id in enumerate(customers_train_set):
            real_id = customer_id
            customer_data = customers_train_set[customer_id]['data']
            next_baskets = customers_test_set[customer_id]['data']
            # Sort test baskets by date
            next_baskets = sorted(
                next_baskets.items(),
                key=lambda x: datetime.datetime.strptime('_'.join(x[0].split('_')[:3]), '%Y_%m_%d')
            )
            # print(f"Customer data length: {len(customer_data)}")
            print(f"Customer ID: {customer_id}, progress: {i}/{len(customers_train_set)}")
            for length in pred_lengths:
                pred_basket, evaluations, nb, b = predict_with_model(hrm, i, customer_data, next_baskets,
                                                                     new2old, dataset_name, coop_level, test_partition_type,
                                                                     real_id, model_name, length, training_time)
                performance[customer_id] = evaluations
                # print(f"Number of next baskets (nb): {nb}")
                # print(f"Number of evaluated baskets (b): {b}")
                # print(f"Evaluations: {evaluations}")
                # print()
            # del customer_data, next_baskets, pred_basket, evaluations
            # gc.collect()
        # del hrm, i
        # gc.collect()

    elif model_name in ["fpmc", "nmf"]:
        # customers_recsys = dict()
        for customer_id in customers_train_set:
            customer_baskets = data2baskets(customers_train_set[customer_id])
            all_customers_baskets.append(customer_baskets)

        items = get_items(all_customers_baskets)

        if model_name == "fpmc":
            model = FPMC(n_user=len(customers_train_set), n_item=len(items), n_factor=n_factor)
            model.build_model(all_customers_baskets, set(items))
        else:  # nmf
            model = NMF(n_user=len(customers_train_set), n_item=len(items), n_factor=n_factor)
            model.build_model(all_customers_baskets)

        end_time = datetime.datetime.now()
        training_time = end_time - start_time
        print(datetime.datetime.now(), f'Model {model_name}, pred_length {pred_length} built in', end_time - start_time)
        print(datetime.datetime.now(), 'Perform predictions')

        i = 0
        for customer_id in customers_train_set:
            built_model = model
            real_id = customer_id
            customer_data = customers_train_set[customer_id]['data']
            next_baskets = customers_test_set[customer_id]['data']
            # Sort test baskets by date
            next_baskets = sorted(
                next_baskets.items(),
                key=lambda x: datetime.datetime.strptime('_'.join(x[0].split('_')[:3]), '%Y_%m_%d')
            )
            # print(f"Customer data length: {len(customer_data)}")
            print(f"Customer ID: {customer_id}, progress: {i}/{len(customers_train_set) - 1}")
            for length in pred_lengths:
                pred_basket, evaluations, nb, _ = predict_with_model(built_model, i, customer_data, next_baskets,
                                                                     new2old, dataset_name, coop_level,
                                                                     test_partition_type, real_id,
                                                                     model_name, length, training_time)
                # Store evaluations directly
                performance[customer_id] = evaluations
                # print(f"Number of evaluated baskets: {nb}")
                # print(f"Evaluations: {evaluations}")
                # print()
            i += 1
            # del customer_data, next_baskets, pred_basket, evaluations
            # gc.collect()
        # customers_recsys = {customer_id: model for customer_id in customers_train_set}
        # del training_time, end_time, model, built_model
        # gc.collect()

    elif model_name == "tbp":
        customers_recsys = dict()
        c = 0
        for customer_id in customers_train_set:
            print(datetime.datetime.now(), customer_id, c, '/', len(customers_train_set))
            start_time = datetime.datetime.now()
            tbp = TBP()
            tbp.build_model(customers_train_set[customer_id])
            customers_recsys[customer_id] = tbp

            end_time = datetime.datetime.now()
            training_time = end_time - start_time
            print(datetime.datetime.now(), f'Model {model_name}, pred_length {pred_length} built in', end_time - start_time)
            print(datetime.datetime.now(), 'Perform predictions')
        # for customer_id in customers_train_set:
            real_id = customer_id
            if customer_id not in customers_recsys:
                continue
            tbp_model = customers_recsys[customer_id]
            customer_data = customers_train_set[customer_id]['data']
            next_baskets = customers_test_set[customer_id]['data']
            # Sort test baskets by date
            next_baskets = sorted(
                next_baskets.items(),
                key=lambda x: datetime.datetime.strptime('_'.join(x[0].split('_')[:3]), '%Y_%m_%d')
            )
            # print(f"Customer data length: {len(customer_data)}")
            print(f"Customer ID: {customer_id}, progress: {c}/{len(customers_train_set)}")
            for length in pred_lengths:
                pred_basket, evaluations, nb, _ = predict_with_model(tbp_model, customer_id, customer_data,
                                                                     next_baskets,
                                                                     new2old, dataset_name, coop_level,
                                                                     test_partition_type, real_id,
                                                                     model_name, length, training_time)
                performance[customer_id] = evaluations
            # del training_time, end_time, customer_data, next_baskets, pred_basket, evaluations
            # del tbp_model, customers_recsys[customer_id], tbp
            # gc.collect()
            c += 1
                # print(f"Number of evaluated baskets: {nb}")
                # print(f"Evaluations: {evaluations}")
                # print()
        # del customers_recsys, c
        # gc.collect()

    elif model_name == 'last':
        customers_recsys = dict()
        for customer_id in customers_train_set:
            customer_train_set = customers_train_set[customer_id]
            baskets = data2baskets(customer_train_set)
            model = Last()
            model.build_model(baskets)
            customers_recsys[customer_id] = model
        end_time = datetime.datetime.now()
        training_time = end_time - start_time
        print(datetime.datetime.now(), f'Model {model_name}, pred_length {pred_length} built in', end_time - start_time)
        print(datetime.datetime.now(), 'Perform predictions')
        i = 1
        for customer_id in customers_train_set:
            real_id = customer_id
            if customer_id not in customers_recsys:
                continue
            model = customers_recsys[customer_id]
            customer_data = customers_train_set[customer_id]['data']
            next_baskets = customers_test_set[customer_id]['data']
            # Sort test baskets by date
            next_baskets = sorted(
                next_baskets.items(),
                key=lambda x: datetime.datetime.strptime('_'.join(x[0].split('_')[:3]), '%Y_%m_%d')
            )
            # print(f"Customer data length: {len(customer_data)}")
            print(f"Customer ID: {customer_id}, progress: {i}/{len(customers_train_set)}")
            pred_basket, evaluations, nb, _ = predict_with_model(model, customer_id, customer_data, next_baskets,
                                                                 new2old, dataset_name, coop_level,
                                                                 test_partition_type, real_id,
                                                                 model_name, pred_length, training_time)
            performance[customer_id] = evaluations
            # print(f"Number of evaluated baskets: {nb}")
            # print(f"Evaluations: {evaluations}")
            # print()
            i += 1
        # del customer_data, next_baskets, pred_basket, evaluations, baskets, customer_train_set, model, i
        # del training_time, end_time, customers_recsys
        # gc.collect()

    elif model_name != "tbp":
        customers_recsys = dict()
        for customer_id in customers_train_set:
            customer_train_set = customers_train_set[customer_id]
            baskets = data2baskets(customer_train_set)
            print("Model for customer: ", customer_id)
            if model_name == 'top':
                model = Top()
                model.build_model(baskets)
            elif model_name == 'clf':
                model = CLF()
                model.build_model(customer_train_set)
            elif model_name == 'markov':
                model = MarkovChain()
                model.build_model(baskets)
            elif model_name == 'ibp':
                model = IntervalBasedPredictor()
                model.build_model(customer_train_set)
            elif model_name == "xmt":
                model = XMT(new2old)
                model.build_model(customer_train_set)
            elif model_name == "txmt":
                model = TXMT(new2old)
                model.build_model(customer_train_set)
            else:
                raise ValueError(f"Unsupported model type: {model_name}")
            customers_recsys[customer_id] = model
            # print("customers_recsys ", len(customers_recsys))

        end_time = datetime.datetime.now()
        training_time = end_time - start_time
        print(datetime.datetime.now(), f'Model {model_name}, pred_length {pred_length} built in', end_time - start_time)
        print(datetime.datetime.now(), 'Perform predictions')
        i = 1
        for customer_id in customers_train_set:
            real_id = customer_id
            if customer_id not in customers_recsys:
                continue
            model = customers_recsys[customer_id]
            customer_data = customers_train_set[customer_id]['data']
            next_baskets = customers_test_set[customer_id]['data']
            # Sort test baskets by date
            next_baskets = sorted(
                next_baskets.items(),
                key=lambda x: datetime.datetime.strptime('_'.join(x[0].split('_')[:3]), '%Y_%m_%d')
            )
            # print(f"Customer data length: {len(customer_data)}")
            print(f"Customer ID: {customer_id}, progress: {i}/{len(customers_train_set)}")
            print('pred_lengths:',pred_lengths)
            for length in pred_lengths:
                pred_basket, evaluations, nb, _ = predict_with_model(model, customer_id, customer_data, next_baskets,
                                                                     new2old, dataset_name, coop_level,
                                                                     test_partition_type, real_id,
                                                                     model_name, length, training_time)
                performance[customer_id] = evaluations
                print("performance")
                # print(f"Number of evaluated baskets: {nb}")
                # print(f"Evaluations: {evaluations}")
                # print()
                i += 1
        # del customer_data, next_baskets, pred_basket, evaluations, baskets, customer_train_set, model, i
        # del training_time, end_time, customers_recsys
        # gc.collect()
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    # Calculate aggregate statistics for each next basket
    stats_per_basket = []
    recall_per_basket = []
    num_baskets = max(len(evals) for evals in performance.values())
    for basket_index in range(num_baskets):
        f1_values = [customer_evals[basket_index]['f1_score'] for customer_evals in performance.values() if
                     basket_index < len(customer_evals)]
        recalls = [customer_evals[basket_index]['recall'] for customer_evals in performance.values() if
                   basket_index < len(customer_evals)]
        stats = calculate_aggregate(f1_values)
        recall = calculate_aggregate(recalls)
        stats_per_basket.append(stats)
        recall_per_basket.append(recall)
    # Calculate overall statistics
    all_f1_values = [eval['f1_score'] for customer_evals in performance.values() for eval in customer_evals]
    all_recalls = [eval['recall'] for customer_evals in performance.values() for eval in customer_evals]
    overall_stats = calculate_aggregate(all_f1_values)
    overall_recall = calculate_aggregate(all_recalls)

    print(f"Model: {model_name}")
    print("Overall Statistics:")
    print(f"  Average F1 Score: {overall_stats['avg']:.4f}")

    for i, (stats, recall) in enumerate(zip(stats_per_basket, recall_per_basket)):
        print(f"Next Basket {i + 1}:")
        print(f"  Average F1 Score: {stats['avg']:.4f}")

    print("Train length", len(customers_train_set))
    print("Test length", len(customers_train_set))
    end_time = datetime.datetime.now()
    print(datetime.datetime.now(), 'Prediction performed in', end_time - start_time)
    cleanup()
    return performance, stats_per_basket, recall_per_basket, overall_stats, overall_recall


def run_single_experiment(experiment, shared_data):
    print(f"Running experiment: {experiment}")
    customers_train_set, customers_test_set, new2old = shared_data
    result = run_experiment(customers_train_set, customers_test_set, new2old, *experiment)

    if result is not None:
        performance, stats_per_basket, recall_per_basket, overall_stats, overall_recall = result
        print("Experiment completed.")
        return (experiment, performance, stats_per_basket, recall_per_basket, overall_stats, overall_recall)
    else:
        print("Experiment did not return expected results.")
        return None


def load_data_once(dataset_name, coop_level, test_partition_type):
    # Load and preprocess the dataset once here
    customers_train_set, customers_test_set, new2old = preprocess_data(dataset_name, coop_level, test_partition_type)
    return customers_train_set, customers_test_set, new2old

def process_result(future):
    try:
        result = future.result()
        if result is not None:
            experiment, performance, stats_per_basket, recall_per_basket, overall_stats, overall_recall = result
            print(f"Finished experiment: {experiment}")
            # Process and print results
            print("Stats for this experiment:")
            for j, (stats, recall) in enumerate(zip(stats_per_basket, recall_per_basket)):
                print(f"Next Basket {j + 1}:")
            print("Overall Statistics:")
            print(f" Overall Agg. F1 score: {overall_stats}")
            print(f"  Overall recall: {overall_recall}")
            print("\n")
        gc.collect()
    except Exception as e:
        print(f"Experiment failed: {e}")



def run_all_experiments(dataset_name, splits, models, min_pred_length):
    def experiment_generator():
        for split, model in itertools.product(splits, models):
            if dataset_name == 'coop':
                yield (dataset_name, 'category', split, model, min_pred_length)
            else:
                yield (dataset_name, None, split, model, min_pred_length)

    max_workers = multiprocessing.cpu_count() - 4  # Leave four CPUs free
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for experiment in experiment_generator():
            shared_data = load_data_once(dataset_name, experiment[1], experiment[2])
            future = executor.submit(run_single_experiment, experiment, shared_data)
            future.add_done_callback(process_result)
            futures.append(future)

        for future in as_completed(futures):
            pass

    print("All experiments completed.")


if __name__ == "__main__":
    min_pred_length = 10  # 2

    splits = [90]  #All splits: [10, 20, 30, 40, 50, 60, 70, 80, 90, 'fixed']   20 da fare

    # models =["top", 'tbp', 'txmt', 'xmt', 'markov', 'last', 'ibp', 'clf', 'fpmc', 'nmf', 'hrm', 'tbp']

    models =["hrm"]

    run_all_experiments('coop', splits, models, min_pred_length)

    # For Tafeng dataset
    # run_all_experiments('tafeng', splits, models, min_pred_length)

    print("All experiments completed.")
