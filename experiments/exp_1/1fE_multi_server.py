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

# from concurrent.futures import ProcessPoolExecutor
from competitors.top import *
from competitors.last import *
from competitors.clf import *
from competitors.ibp import *
from competitors.tbp import *
from competitors.markovchain import *
from competitors.nmf import *
# from competitors.hrm import *
from competitors.hrm_multi_threads import *
from competitors.fpmc import *

from contenders.TARS_fbp_last import *

from imports.utilities.models import *
from imports.utilities.data_management import *
from imports.evaluation.evaluation_measures import *
from imports.evaluation.calculate_aggregate_statistics import *

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
                       pred_length, training_time, csv_file='predictions_1fE.csv'):
    print("pred_length for predictions: ", pred_length)
    if isinstance(model, Top):
        pred_start_time = datetime.datetime.now()
        pred_basket = model.predict(pred_length=pred_length)
        pred_end_time = datetime.datetime.now()
        # Remap predicted items
        pred_basket = set([new2old[item] for item in pred_basket])

        # Attempt to return different evaluations for the 4 baskets separately
        # return pred_basket, [(evaluation, b) for evaluation in performance], nb, b

    elif isinstance(model, Last):
        pred_start_time = datetime.datetime.now()
        pred_basket = model.predict()
        pred_end_time = datetime.datetime.now()
        # Remap predicted items
        pred_basket = set([new2old[item] for item in pred_basket])

        # Attempt to return different evaluations for the 4 baskets separately
        # return pred_basket, [(evaluation, b) for evaluation in performance], nb, b

    elif isinstance(model, CLF):
        o = 1
        # pred_start_time = datetime.datetime.now()
        # pred_basket = model.predict(pred_length=pred_length)
        # pred_end_time = datetime.datetime.now()
        # Remap predicted items
        # pred_basket = set([new2old[item] for item in pred_basket])

        # Attempt to return different evaluations for the 4 baskets separately
        # return pred_basket, [(evaluation, b) for evaluation in performance], nb, b

    elif isinstance(model, MarkovChain):
        o = 1
        # pred_basket = model.predict()
        # Remap predicted items
        # pred_start_time = datetime.datetime.now()
        # pred_basket = set([new2old[item] for item in pred_basket])
        # pred_end_time = datetime.datetime.now()

        # Attempt to return different evaluations for the 4 baskets separately
        # return pred_basket, [(evaluation, b) for evaluation in performance], nb, b

    elif isinstance(model, IntervalBasedPredictor):
        # Get the date of the first next basket
        performance = []
        nb = 0  # next basket counter
        b = 0  # basket counter for individual scores

        # Create or append to CSV file
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write header if file doesn't exist
            if not file_exists:
                writer.writerow(
                    ['dataset_name', 'category_mode', 'split_mode', 'model_name', 'training_time', 'customer_id',
                     'pred_time', 'next_basket_number', 'pred_length', 'next_basket_id', 'actual_basket',
                     'predicted_basket', 'precision', 'recall', 'f1_score', 'f05_score', 'f2_score',
                     'hit_score'])

            for next_basket_id in next_baskets:
                # print(f"Basket nº: {nb + 1}")
                # Predict basket using IBP
                pred_start_time = datetime.datetime.now()
                pred_basket_list, pred_set, poss_forgotten = model.predict_basket(customer_id,
                                                                                  visit_date=next_basket_id,
                                                                                  k=pred_length)
                pred_end_time = datetime.datetime.now()
                pred_time = pred_end_time - pred_start_time
                pred_set = set([new2old[item] for item in pred_set])
                next_basket = next_baskets[next_basket_id]['basket']
                next_basket = set(next_basket.keys())
                # print("Prediction:")
                # print(pred_set)
                # print("Possibly forgotten items:")
                # print(poss_forgotten)
                # print("Actual next basket:")
                # print(next_basket)
                evaluation = evaluate_prediction(next_basket, pred_set)
                performance.append(evaluation)
                # print("Performance:")
                # print(evaluation)

                # Calculate IBP-specific metrics
                # rho = model.calculate_performance(pred_set, next_basket)
                # rev = model.calculate_revenue_increase(pred_set, next_basket)
                # print(f"Performance (rho): {rho}")
                # print(f"Increase in revenue (rev): {rev}")

                # Write to CSV
                writer.writerow([
                    dataset_name,
                    coop_level if coop_level is not None else 'None',
                    test_partition_type,
                    model_name,
                    training_time,
                    customer_id,
                    pred_time,
                    nb + 1,  # next_basket_number
                    pred_length,
                    next_basket_id,
                    ','.join(map(str, next_basket)),
                    ','.join(map(str, pred_set)),
                    evaluation['precision'],
                    evaluation['recall'],
                    evaluation['f1_score'],
                    evaluation['f05_score'],
                    evaluation['f2_score'],
                    evaluation['hit_score']
                ])

                nb += 1
                b += 1
                # Delete next_basket and call garbage collector
                # del next_basket, evaluation
                # gc.collect()
        # Delete large variables and call garbage collector before returning
        # del customer_data, next_baskets, new2old, writer
        # gc.collect()
        return pred_set, performance, nb, b
        # Attempt to return different evaluations for the 4 baskets separately
        # return pred_basket, [(evaluation, b) for evaluation in performance], nb, b

    elif isinstance(model, HRM):
        # Get the last basket from the training data for the current user
        if customer_data:
            sorted_keys = sorted(customer_data.keys())
            last_key = sorted_keys[-1]
            last_basket = customer_data[last_key]['basket']
        else:
            last_basket = {}

    elif isinstance(model, FPMC):
        # Get the last basket from the training data for the current user
        if customer_data:
            sorted_keys = sorted(customer_data.keys())
            last_key = sorted_keys[-1]
            last_basket = customer_data[last_key]['basket']
        else:
            last_basket = {}

    elif isinstance(model, NMF):
        o = 1

    elif isinstance(model, TBP):
        o = 1

    elif isinstance(model, TARSForgottenBasketPredictor):
        o = 1

    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    performance = []
    nb = 0  # next basket counter
    b = 0  # basket counter for individual scores

    # Create or append to CSV file
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(['dataset_name', 'category_mode', 'split_mode', 'model_name', 'training_time', 'customer_id',
                             'pred_time', 'next_basket_number', 'pred_length', 'next_basket_id', 'actual_basket',
                             'predicted_basket', 'precision', 'recall', 'f1_score', 'f05_score', 'f2_score',
                             'hit_score'])

        for next_basket_id in next_baskets:
            if isinstance(model, Top):
                o = 1
            elif isinstance(model, Last):
                o = 1
            elif isinstance(model, CLF):
                pred_start_time = datetime.datetime.now()
                pred_basket = model.predict(pred_length=pred_length)
                pred_end_time = datetime.datetime.now()
                # Remap predicted items
                pred_basket = set([new2old[item] for item in pred_basket])

            elif isinstance(model, MarkovChain):
                pred_basket = model.predict(pred_length=pred_length)
                # Remap predicted items
                pred_start_time = datetime.datetime.now()
                pred_basket = set([new2old[item] for item in pred_basket])
                pred_end_time = datetime.datetime.now()

            elif isinstance(model, HRM):
                # Predict using HRM
                pred_start_time = datetime.datetime.now()
                pred_basket = model.predict(user_id=customer_id, last_basket=list(last_basket.keys()),
                                            pred_length=pred_length)
                pred_end_time = datetime.datetime.now()
                print(real_id)
                print(customer_id)
                # Remap predicted items
                pred_basket = set([new2old[item] for item in pred_basket])

            elif isinstance(model, FPMC):
                # Predict using FPMC
                pred_start_time = datetime.datetime.now()
                pred_basket = model.predict(user_id=customer_id, last_basket=list(last_basket.keys()),
                                            pred_length=pred_length)
                pred_end_time = datetime.datetime.now()
                print(real_id)
                print(customer_id)
                # Remap predicted items
                pred_basket = set([new2old[item] for item in pred_basket])

            elif isinstance(model, NMF):
                # Predict using NMF
                pred_start_time = datetime.datetime.now()
                pred_basket = model.predict(user_id=customer_id, pred_length=pred_length)
                pred_end_time = datetime.datetime.now()
                print(real_id)
                print(customer_id)
                # Remap predicted items
                pred_basket = set([new2old[item] for item in pred_basket])

            elif isinstance(model, TBP):
                day_of_next_purchase = datetime.datetime.strptime(next_basket_id[0:10], '%Y_%m_%d')
                # Predict basket using TBP
                pred_start_time = datetime.datetime.now()
                pred_basket = model.predict(customer_data, day_of_next_purchase, nbr_patterns=None,
                                            pred_length=pred_length)
                pred_end_time = datetime.datetime.now()
                # Remap predicted items
                pred_basket = set([new2old[item] for item in pred_basket])

            elif isinstance(model, TARSForgottenBasketPredictor):
                next_basket = next_baskets[next_basket_id]['basket']
                next_basket = set(next_basket.keys())

                pred_start_time = datetime.datetime.now()
                pred_basket, explanations = model.predict_f(next_basket, next_basket_id, pred_length=pred_length)
                pred_end_time = datetime.datetime.now()
                # Remap the predicted items
                print("Not-yet-remapped prediction")
                print(pred_basket)
                pred_basket = set([new2old[item] for item in pred_basket])
                print("Prediction: ")
                print(pred_basket)

                for item, explanation in zip(pred_basket, explanations):
                    print(explanation)
                    print()

            else:
                raise ValueError(f"Unsupported model type: {type(model)}")

            pred_time = pred_end_time - pred_start_time
            # print(f"Basket nº: {nb + 1}")
            # print("Predicted basket:")
            # print(pred_basket)

            next_basket = next_baskets[next_basket_id]['basket']
            next_basket = set(next_basket.keys())
            # print("Actual next basket:")
            # print(next_basket)

            evaluation = evaluate_prediction(next_basket, pred_basket)
            performance.append(evaluation)
            # print("Performance:")
            # print(evaluation)

            # Write to CSV
            writer.writerow([
                dataset_name,
                coop_level if coop_level is not None else 'None',
                test_partition_type,
                model_name,
                training_time,
                real_id,
                pred_time,
                nb + 1,  # next_basket_number
                pred_length,
                next_basket_id,
                ','.join(map(str, next_basket)),
                ','.join(map(str, pred_basket)),
                evaluation['precision'],
                evaluation['recall'],
                evaluation['f1_score'],
                evaluation['f05_score'],
                evaluation['f2_score'],
                evaluation['hit_score']
            ])

            nb += 1
            b += 1
            # Delete next_basket and call garbage collector
            # del next_basket, evaluation
            # gc.collect()
    # Delete large variables and call garbage collector before returning
    # del customer_data, next_baskets, new2old, writer
    # gc.collect()
    return pred_basket, performance, nb, b


# def read_data_gz(file_path):
#     """Read a gzipped JSON file where each line is a separate JSON object."""
#     customers_data = {}
#     with gzip.open(file_path, 'rt', encoding='utf-8') as f:
#         for line in f:
#             customer_data = json.loads(line)
#             customer_id = customer_data['customer_id']
#             customers_data[customer_id] = customer_data
#     return customers_data


def preprocess_data(dataset_name, coop_level, test_partition_type, model_name=None, pred_length=None):
    print(f"Reading dataset{dataset_name}")

    if dataset_name == 'tafeng':
        dataset = read_data(path + 'tafeng.json')
        item2category = None
    elif dataset_name.startswith('coop'):
        # Comment the next two lines if dataset is not numbered
        # coop_number = dataset_name[4:]  # Extract the number after 'coop'
        # dataset = read_data(path + f'coop_data_clean_split_{coop_number}.json')

        # Comment out if NOT using the full dataset
        # dataset = read_data(path + f'coop_data_clean.json')
        # dataset = read_data(path + f'coop_data_15k_clean.json')
        # dataset = read_data(path + f'coop_data_10k_clean.json')

        # Test dataset
        # dataset = read_data(path + f'coop_data_10.json')

        dataset = read_data(path + f'coop_data_clean_split_one.json')

        if coop_level == 'category':
            item2category = get_item2category(path + 'market.csv', category_index['categoria'])
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
                                                               item2category=item2category)

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
                       pred_length=5, n_factor=20):

    print(datetime.datetime.now(), 'Creating and building models')
    start_time = datetime.datetime.now()
    # List of baskets for HRM() and FPMC()
    all_customers_baskets = []
    # Pred_length and performances for TBP
    performance = defaultdict(list)

    # pred_lengths for experiments
    min_pred_length = pred_length  # 2
    max_pred_length = 21  # 21
    pred_lengths = list(range(min_pred_length, max_pred_length))

    if model_name == "hrm":
        for customer_id in customers_train_set:
            customer_baskets = data2baskets(customers_train_set[customer_id])
            all_customers_baskets.append(customer_baskets)
        items = get_items(all_customers_baskets)
        v_suggestion = int(np.sqrt(len(items)))
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
            # print(f"Customer data length: {len(customer_data)}")
            print(f"Customer ID: {customer_id}, progress: {i}/{len(customers_train_set)}")
            for length in pred_lengths:
                pred_basket, evaluations, nb, b = predict_with_model(hrm, i, customer_data, next_baskets,
                                                                     new2old, dataset_name, coop_level, test_partition_type,
                                                                     real_id, model_name, length, training_time,
                                                                     csv_file=f'1fE/{model_name}/split_{test_partition_type}/pred_1fE_{test_partition_type}_{model_name}_{length}.csv')
                # Store evaluations directly
                performance[customer_id] = evaluations
                # print(f"Number of next baskets (nb): {nb}")
                # print(f"Number of evaluated baskets (b): {b}")
                # print(f"Evaluations: {evaluations}")
                # print()
            del customer_data, next_baskets, pred_basket, evaluations
            gc.collect()
        del hrm, i
        gc.collect()

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
            # print(f"Customer data length: {len(customer_data)}")
            print(f"Customer ID: {customer_id}, progress: {i}/{len(customers_train_set) - 1}")
            for length in pred_lengths:
                pred_basket, evaluations, nb, _ = predict_with_model(built_model, i, customer_data, next_baskets,
                                                                     new2old, dataset_name, coop_level,
                                                                     test_partition_type, real_id,
                                                                     model_name, length, training_time,
                                                                     csv_file=f'1fE/{model_name}/split_{test_partition_type}/pred_1fE_{test_partition_type}_{model_name}_{length}.csv')
                # Store evaluations directly
                performance[customer_id] = evaluations
                # print(f"Number of evaluated baskets: {nb}")
                # print(f"Evaluations: {evaluations}")
                # print()
            i += 1
            del customer_data, next_baskets, pred_basket, evaluations
            gc.collect()
        # customers_recsys = {customer_id: model for customer_id in customers_train_set}
        del training_time, end_time, model, built_model
        gc.collect()

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
            # print(f"Customer data length: {len(customer_data)}")
            print(f"Customer ID: {customer_id}, progress: {c}/{len(customers_train_set)}")
            for length in pred_lengths:
                pred_basket, evaluations, nb, _ = predict_with_model(tbp_model, customer_id, customer_data,
                                                                     next_baskets,
                                                                     new2old, dataset_name, coop_level,
                                                                     test_partition_type, real_id,
                                                                     model_name, length, training_time,
                                                                     csv_file=f'1fE/{model_name}/split_{test_partition_type}/pred_1fE_{test_partition_type}_{model_name}_{length}.csv')
                # Store evaluations directly
                performance[customer_id] = evaluations
            del training_time, end_time, customer_data, next_baskets, pred_basket, evaluations
            del tbp_model, customers_recsys[customer_id], tbp
            gc.collect()
            # print(f"Number of evaluated baskets: {nb}")
            # print(f"Evaluations: {evaluations}")
            # print()
            c += 1
        del customers_recsys, c
        gc.collect()

    elif model_name == "last":
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
            # print(f"Customer data length: {len(customer_data)}")
            print(f"Customer ID: {customer_id}, progress: {i}/{len(customers_train_set)}")
            pred_basket, evaluations, nb, _ = predict_with_model(model, customer_id, customer_data, next_baskets,
                                                                 new2old, dataset_name, coop_level,
                                                                 test_partition_type, real_id,
                                                                 model_name, pred_length, training_time,
                                                                 csv_file=f'1fE/{model_name}/split_{test_partition_type}/pred_1fE_{test_partition_type}_{model_name}_{length}.csv')
            # Store evaluations directly
            performance[customer_id] = evaluations
            # print(f"Number of evaluated baskets: {nb}")
            # print(f"Evaluations: {evaluations}")
            # print()
            i += 1
        del customer_data, next_baskets, pred_basket, evaluations, baskets, customer_train_set, model, i
        del training_time, end_time, customers_recsys
        gc.collect()

    elif model_name != "tbp":
        customers_recsys = dict()
        for customer_id in customers_train_set:
            customer_train_set = customers_train_set[customer_id]
            baskets = data2baskets(customer_train_set)
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
            elif model_name == "TARS_fbp_last":
                model = TARSForgottenBasketPredictor(new2old)
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
            # print(f"Customer data length: {len(customer_data)}")
            print(f"Customer ID: {customer_id}, progress: {i}/{len(customers_train_set)}")
            for length in pred_lengths:
                pred_basket, evaluations, nb, _ = predict_with_model(model, customer_id, customer_data, next_baskets,
                                                                     new2old, dataset_name, coop_level,
                                                                     test_partition_type, real_id,
                                                                     model_name, length, training_time,
                                                                     csv_file=f'1fE/{model_name}/split_{test_partition_type}/pred_1fE_{test_partition_type}_{model_name}_{length}.csv')
                # Store evaluations directly
                performance[customer_id] = evaluations
                # print(f"Number of evaluated baskets: {nb}")
                # print(f"Evaluations: {evaluations}")
                # print()
            i += 1
        del customer_data, next_baskets, pred_basket, evaluations, baskets, customer_train_set, model, i
        del training_time, end_time, customers_recsys
        gc.collect()
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
            if model in ['tbp', 'TARS_fbp_last'] and dataset_name == 'coop':
                yield (dataset_name, 'category', split, model, min_pred_length)
            else:
                yield (dataset_name, None, split, model, min_pred_length)

    max_workers = multiprocessing.cpu_count() - 4  # Leave one CPU free
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



# ONE FOR EACH

# test
# path = "../dataset/coop test/"

# Full
# path = "../dataset/Full dataset/clean/"

# tafeng
# path = "../dataset/tafeng/"

# 1k
path = '../dataset/1k/'

# Split case

# if __name__ == "__main__":
#     min_pred_length = 2  # 2
#     # max_pred_length = 21  # 21
#     # pred_lengths = list(range(min_pred_length, max_pred_length))
#     splits = [10, 20, 30, 40, 50, 60, 70, 80, 90, 'fixed']  #[10, 20, 30, 40, 50, 60, 70, 80, 90, 'fixed']
#     coop_numbers = range(1, 2)
#     # range(1, 5) i.e. 4k split  OR range(1, 3) i.e. 8k split
#
#     models = ['tbp', 'top', 'markov', 'last', 'ibp', 'clf', 'fpmc', 'nmf', 'hrm']
#
#     #to do:  ['tbp', 'top', 'markov', 'last', 'ibp', 'clf', 'fpmc', 'nmf', 'hrm']
#
#     #All models: ['tbp', 'top', 'markov', 'last', 'ibp', 'clf', 'fpmc', 'nmf', 'hrm']
#
#     for model in models:
#         for coop_number in coop_numbers:
#             for split in splits:
#                 # Load the dataset only once per combination of coop_number and split
#                 shared_data = load_data_once(f'coop{coop_number}', None, split)
#
#                 # Limit of concurrent processes
#                 max_workers = 1
#
#                 with ProcessPoolExecutor(max_workers=max_workers) as executor:
#                     futures = []
#                     # for pred_length in pred_lengths:
#                     if model == 'tbp':
#                         experiment = (f'coop{coop_number}', 'category', split, model, min_pred_length)
#                     else:
#                         experiment = (f'coop{coop_number}', None, split, model, min_pred_length)
#
#                     future = executor.submit(run_single_experiment, experiment, shared_data)
#                     future.add_done_callback(process_result)
#                     futures.append(future)
#
#                     # Wait for all futures to complete
#                     for future in as_completed(futures):
#                         pass
#
#         print(f"All experiments for model {model} completed.")
#
#     print("All experiments completed.")


#  Whole 15k dataset case:
if __name__ == "__main__":
    min_pred_length = 2  # 2
    # max_pred_length = 21  # 21
    # pred_lengths = list(range(min_pred_length, max_pred_length))
    splits = splits  #All splits: [10, 20, 30, 40, 50, 60, 70, 80, 90, 'fixed']

    # models = ["top"]

    models = models

    #to do:  ['tbp', 'markov', 'ibp', 'clf', 'fpmc', 'nmf', 'hrm', "TARS_fbp_last"]

    #All models: ['tbp', 'markov', 'ibp', 'clf', 'fpmc', 'nmf', 'hrm']

    # For Coop dataset
    # numbered splits
    # for coop_number in range(1, 18):
    #     run_all_experiments(f'coop{coop_number}', splits, models, min_pred_length)


    # run_all_experiments('coop', splits, models, min_pred_length)

    # For Tafeng dataset
    run_all_experiments('tafeng', splits, models, min_pred_length)

    print("All experiments completed.")

    print("All experiments completed.")




#  Tafeng
# if __name__ == "__main__":
#     min_pred_length = 2  # 2
#     # max_pred_length = 21  # 21
#     # pred_lengths = list(range(min_pred_length, max_pred_length))
#     splits = [10, 20, 30, 40, 50, 60, 70, 80, 90, 'fixed']  #All splits: [10, 20, 30, 40, 50, 60, 70, 80, 90, 'fixed']
#
#     models = ['tbp', 'top', 'markov', 'last', 'ibp', 'clf', 'fpmc', 'nmf', 'hrm']
#
#     #to do:  ['tbp', 'top', 'markov', 'last', 'ibp', 'clf', 'fpmc', 'nmf', 'hrm']
#
#     #All models: ['tbp', 'top', 'markov', 'last', 'ibp', 'clf', 'fpmc', 'nmf', 'hrm']
#
#     for model in models:
#         for split in splits:
#             # Load the dataset only once per split
#             shared_data = load_data_once('tafeng', None, split)
#             # Limit of concurrent processes
#             max_workers = 1
#             with ProcessPoolExecutor(max_workers=max_workers) as executor:
#                 futures = []
#                 experiment = ('tafeng', None, split, model, min_pred_length)
#                 future = executor.submit(run_single_experiment, experiment, shared_data)
#                 future.add_done_callback(process_result)
#                 futures.append(future)
#                 # Wait for all futures to complete
#                 for future in as_completed(futures):
#                     pass
#         print(f"All experiments for model {model} completed.")
#
#     print("All experiments completed.")