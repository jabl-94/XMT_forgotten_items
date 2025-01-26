import datetime
from collections import defaultdict
import os

from forgotten_items.competitors.fpmc import FPMC
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

    # path1 = 'datasets/coop100.json'
    # path2 = 'datasets/tafeng.json'

    print("Reading dataset")
    dataset = read_data(path1)
    print("Dataset read")
    # print(dataset)

    # Set up test partition configuration
    test_partition_type = 'fixed'
    split_mode = 'loo' if test_partition_type == 'fixed' else 'rnd'

    print(datetime.datetime.now(), 'Partition dataset into train / test')
    customers_train_set, customers_test_set = split_train_test_og(
        dataset,
        split_mode=split_mode,
        min_number_of_basket=10,
        min_basket_size=1,
        max_basket_size=float('inf'),
        min_item_occurrences=2
    )

    # Remap items for consistency
    customers_train_set, new2old, old2new = remap_items_with_data(customers_train_set)
    print(datetime.datetime.now(), 'Customers for test', len(customers_train_set),
          '%.2f%%' % (100.0 * len(customers_train_set) / len(dataset)))

    # Model building phase
    print(datetime.datetime.now(), 'Create and build models')
    start_time = datetime.datetime.now()

    # Prepare all customer baskets
    all_customers_baskets = [
        data2baskets(customers_train_set[customer_id])
        for customer_id in customers_train_set
    ]

    # Get unique items and build FPMC model
    items = get_items(all_customers_baskets)
    n_factor = 100
    fpmc_model = FPMC(n_user=len(customers_train_set), n_item=len(items), n_factor=n_factor)
    fpmc_model.build_model(all_customers_baskets, set(items))

    end_time = datetime.datetime.now()
    print(datetime.datetime.now(), 'Models built in', end_time - start_time)

    # Prediction and evaluation phase
    print(datetime.datetime.now(), 'Perform predictions')
    performance = defaultdict(list)

    for i, customer_id in enumerate(customers_train_set):
        customer_data = customers_train_set[customer_id]['data']
        next_baskets = customers_test_set[customer_id]['data']

        # Get last basket from training data
        last_basket = {}
        if customer_data:
            sorted_keys = sorted(customer_data.keys())
            last_basket = customer_data[sorted_keys[-1]]['basket']

        # Evaluate predictions for each next basket
        for next_basket_id in next_baskets:
            print(customer_id - 1)
            pred_basket = fpmc_model.predict(user_id=i, last_basket=list(last_basket.keys()))
            pred_basket = set(new2old[item] for item in pred_basket)

            next_basket = set(next_baskets[next_basket_id]['basket'].keys())
            evaluation = evaluate_prediction(next_basket, pred_basket)
            performance[customer_id].append(evaluation)
            print(performance[customer_id])

    # Calculate and display final statistics
    print("Train length", len(customers_train_set))
    print("Test length", len(customers_train_set))

    end_time = datetime.datetime.now()
    print(datetime.datetime.now(), 'Prediction performed in', end_time - start_time)

    f1_values = [
        evaluation['f1_score']
        for customer_id in performance
        for evaluation in performance[customer_id]
    ]

    stats = calculate_aggregate(f1_values)
    print(datetime.datetime.now(), 'FPMC', 'avg', stats['avg'])
    print(stats)
    print(f1_values)


if __name__ == "__main__":
    main()