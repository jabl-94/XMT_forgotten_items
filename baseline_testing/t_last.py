import datetime
from collections import defaultdict
import os

from forgotten_items.competitors.last import Last
from forgotten_items.imports.utilities.data_management import *
from forgotten_items.imports.evaluation.evaluation_measures import *
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

    print(list(dataset.keys()))

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


    customers_set = dict()
    customers_recsys = dict()
    start_time = datetime.datetime.now()

    for customer_id in list(customers_train_set.keys()):
        print(datetime.datetime.now(), customer_id)
        print(type(customers_train_set[customer_id]))
        customer_train_set = customers_train_set[customer_id]
        baskets = data2baskets(customer_train_set)
        last = Last()
        last.build_model(baskets)
        customers_recsys[customer_id] = last

    end_time = datetime.datetime.now()
    print(datetime.datetime.now(), 'Models built in', end_time - start_time)

    print(datetime.datetime.now(), 'Perform predictions')
    performance = defaultdict(list)

    for customer_id in customers_train_set:
            if customer_id not in customers_recsys:
                continue

            model = customers_recsys[customer_id]
            customer_data = customers_train_set[customer_id]['data']
            next_baskets = customers_test_set[customer_id]['data']

            for next_basket_id in next_baskets:
                pred_basket = model.predict()
                print("Not-yet remapped prediction")
                print(pred_basket)
                pred_basket = set([new2old[item] for item in pred_basket])
                print("Prediction")
                print(pred_basket)

                next_basket = next_baskets[next_basket_id]['basket']
                next_basket = set(next_basket.keys())
                print("Actual next basket")
                print(next_basket)

                evaluation = evaluate_prediction(next_basket, pred_basket)
                performance[customer_id].append(evaluation)
                print("Performance:")
                print(performance[customer_id])
                print("\n")

    end_time = datetime.datetime.now()
    print(datetime.datetime.now(), 'Prediction performed in', end_time - start_time)

    f1_values = list()

    for customer_id in performance:
        for evaluation in performance[customer_id]:
            f1_values.append(evaluation['f1_score'])

    stats = calculate_aggregate(f1_values)
    print(datetime.datetime.now(), 'LAST', 'avg', stats['avg'])
    print(stats)
    print(f1_values)


if __name__ == "__main__":
    main()

