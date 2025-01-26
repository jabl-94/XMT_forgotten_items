import datetime
from collections import defaultdict
import os
import numpy as np

from forgotten_items.competitors.hrm import HRM
from forgotten_items.imports.utilities.data_management import (
    read_data,
    data2baskets,
    get_items,
    split_train_test_og,
    remap_items_with_data
)
from forgotten_items.imports.evaluation.evaluation_measures import evaluate_prediction
from forgotten_items.imports.evaluation.calculate_aggregate_statistics import calculate_aggregate

def get_dataset_path():
    """Get the path to the datasets directory relative to this script."""
    # This script is in baseline_testing/, so we need to go up one level to reach datasets/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(os.path.dirname(current_dir), 'datasets')
    return dataset_dir


def main():
    dataset_dir = get_dataset_path()
    path1 = os.path.join(dataset_dir, 'coop100.json')
    path2 = os.path.join(dataset_dir, 'tafeng.json')

    print("Reading dataset")
    dataset = read_data(path1)
    print("Dataset read")

    # print(dataset)

    # print("Getting baskets")
    # b = data2baskets(dataset)
    #
    # print("Getting the number of items")
    # n_items = get_items(dataset)
    # print(len(n_items))


    # print(list(dataset.keys()))

    test_partition_type = 'fixed'

    if test_partition_type == 'fixed':
        split_mode = 'loo'
    elif test_partition_type == 'random':
        split_mode = 'rnd'
    else:
        print(datetime.datetime.now(), 'Unknown test partition type')

    print(datetime.datetime.now(), 'Partition dataset into train / test')
    customers_train_set, customers_test_set = split_train_test_og(dataset,
                                                               split_mode=split_mode,
                                                               min_number_of_basket=10,
                                                               min_basket_size=1,
                                                               max_basket_size=float('inf'),
                                                               min_item_occurrences=2)

    customers_train_set, new2old, old2new = remap_items_with_data(customers_train_set)

    print(datetime.datetime.now(), 'Customers for test', len(customers_train_set), '%.2f%%' % (100.0 * len(customers_train_set) / len(dataset)))

    print(datetime.datetime.now(), 'Create and build models')

    start_time = datetime.datetime.now()

    # for customer_id in list(customers_train_set.keys()):
    # print(datetime.datetime.now(), 'Customer ID: ' , customer_id, "\n")
    # customer_train_set = customers_train_set[customer_id]
    # baskets = data2baskets(customer_train_set)
    all_customers_baskets = []
    for customer_id in customers_train_set:
        customer_baskets = data2baskets(customers_train_set[customer_id])
        all_customers_baskets.append(customer_baskets)

    items = get_items(all_customers_baskets)

    # print(all_customers_baskets[0])

    v_suggestion = (int(np.sqrt(len(items))))
    print(v_suggestion)

    hrm = HRM(n_user=len(customers_train_set), n_item=len(items), u_dim=50, v_dim=50, verbose=True)
    hrm.build_model(all_customers_baskets)

    end_time = datetime.datetime.now()
    print(datetime.datetime.now(), 'Models built in', end_time - start_time)

    print(datetime.datetime.now(), 'Perform predictions')
    performance = defaultdict(list)

    i = 0
    for customer_id in customers_train_set:
        model = hrm
        customer_data = customers_train_set[customer_id]['data']
        next_baskets = customers_test_set[customer_id]['data']

        # Get the last basket from the training data for the current user
        if customer_data:
            sorted_keys = sorted(customer_data.keys())
            last_key = sorted_keys[-1]
            last_basket = customer_data[last_key]['basket']
        else:
            last_basket = {}

        print(customer_id - 1, "\n")
        pred_basket = model.predict(user_id=i, last_basket=list(last_basket.keys()))
        pred_basket = set([new2old[item] for item in pred_basket])
        print(pred_basket)

        for next_basket_id in next_baskets:
            next_basket = next_baskets[next_basket_id]['basket']
            next_basket = set(next_basket.keys())
            print(next_basket)

            evaluation = evaluate_prediction(next_basket, pred_basket)
            performance[customer_id].append(evaluation)
            print(performance[customer_id])
        i += 1

    print("Train length" ,len(customers_train_set))
    print("Test length", len(customers_train_set))
    end_time = datetime.datetime.now()
    print(datetime.datetime.now(), 'Prediction performed in', end_time - start_time)

    f1_values = list()

    for customer_id in performance:
        for evaluation in performance[customer_id]:
            f1_values.append(evaluation['f1_score'])

    stats = calculate_aggregate(f1_values)
    print(datetime.datetime.now(), 'NMF', 'avg', stats['avg'])
    print(stats)
    print(f1_values)


if __name__ == "__main__":
    main()