import datetime
from collections import defaultdict
import os

from forgotten_items.competitors.top import Top
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

    # print(list(dataset.keys()))

    test_partition_type = 'last4'

    # Partition types 'fixed', 'random', 'standard', 'last4'
    if test_partition_type == 'fixed':
        split_mode = 'loo'
    elif test_partition_type == 'random':
        split_mode = 'rnd'
    elif test_partition_type == 'standard':
        split_mode = '70'
    elif test_partition_type == 'last4':
        split_mode = '-4'
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
    avg_f1_per_k = {}

    for customer_id in list(customers_train_set.keys()):
        print(datetime.datetime.now(), customer_id)
        print(type(customers_train_set[customer_id]))
        customer_train_set = customers_train_set[customer_id]
        baskets = data2baskets(customer_train_set)
        top = Top()
        top.build_model(baskets)
        customers_recsys[customer_id] = top

    end_time = datetime.datetime.now()
    print(datetime.datetime.now(), 'Models built in', end_time - start_time)

    print(datetime.datetime.now(), 'Perform predictions')
    performance = defaultdict(list)

    for customer_id in customers_train_set:
            if customer_id not in customers_recsys:
                continue

            top = customers_recsys[customer_id]
            customer_data = customers_train_set[customer_id]['data']
            next_baskets = customers_test_set[customer_id]['data']

            # Getting the average value of K for each customer
            # print(len(customer_data))
            # total_baskets = len(customer_data)
            # total_items = 0
            # for transactions in customer_data:
            #     for baskets in customer_data[transactions]:
            #         num_items = len(customer_data[transactions][baskets])
            #         total_items = total_items + num_items
            #
            # k = total_items/total_baskets
            # k_rounded = round(k)
            # print("Value of k: ", k_rounded)

            for next_basket_id in next_baskets:
                pred_basket = top.predict(pred_length=10)
                print("Not-yet-remapped prediction")
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
                print(performance[customer_id][0]['hit_score'])
                # if performance[customer_id][0]['hit_score'] > 0.0:
                #     avg_f1_per_k = update_f1_scores(avg_f1_per_k, k_rounded, performance[customer_id][0]['f1_score'])
                # print(avg_f1_per_k)
                print("\n")

    # median_f1_per_k = calculate_median_f1_scores(avg_f1_per_k)
    # median_f1_per_k = sorted(median_f1_per_k)
    # print("Median F1 scores per k:", median_f1_per_k)

    end_time = datetime.datetime.now()
    print(datetime.datetime.now(), 'Prediction performed in', end_time - start_time)

    # Saving the median values for k
    # csv_filename = 'TOP_median_f1_scores_per_k.csv'
    # with open(csv_filename, 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(['k', 'median_f1_score'])
    #     csvwriter.writerows(median_f1_per_k)
    #
    # print(f"Median F1 scores per k saved to {csv_filename}")

    f1_values = list()

    for customer_id in performance:
        for evaluation in performance[customer_id]:
            f1_values.append(evaluation['f1_score'])

    stats = calculate_aggregate(f1_values)
    print(datetime.datetime.now(), 'TOP', 'avg', stats['avg'])
    print(stats)
    print(f1_values)
    # print(dataset[1])

    # top_pred[customer_id] = customers_recsys[customer_id].predict(pred_length=5)
    #
    #
    #     # Print the prediction
    #     print(top_pred[customer_id])



if __name__ == "__main__":
    main()
