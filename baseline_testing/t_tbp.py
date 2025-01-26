import datetime
from collections import defaultdict
import os

from forgotten_items.competitors.tbp import TBP
from forgotten_items.imports.utilities.data_management import (
    read_data,
    data2baskets,
    get_item2category,
    split_train_test_og,
    category_index,
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
    print('Test TBP')

    pred_length = 10

    dataset_dir = get_dataset_path()
    path1 = os.path.join(dataset_dir, 'coop100.json')
    path2 = os.path.join(dataset_dir, 'tafeng.json')

    dataset = 'coop'  # tafeng
    coop_level = 'category'  # category, segment
    test_partition_type = 'fixed'  # fixed, random

    min_pred_length = 2
    max_pred_length = 21
    pred_lengths = list(range(min_pred_length, max_pred_length))

    print(datetime.datetime.now(), 'Read dataset', dataset)

    if dataset == 'tafeng':
        customers_data = read_data(path2)
        item2category = None
    elif dataset.startswith('coop'):
        customers_data = read_data(path1)
        if coop_level == 'category':
            item2category = get_item2category('market.csv', category_index['categoria'])
        else:
            item2category = None
    else:
        print(datetime.datetime.now(), 'Unknown dataset')
        return

    print(datetime.datetime.now(), 'Customers', len(customers_data))

    if test_partition_type == 'fixed':
        split_mode = 'loo'
    elif test_partition_type == 'random':
        split_mode = 'rnd'
    else:
        print(datetime.datetime.now(), 'Unknown test partition type')
        return

    print(datetime.datetime.now(), 'Partition dataset into train / test')
    customers_train_set, customers_test_set = split_train_test_og(customers_data,
                                                               split_mode=split_mode,
                                                               min_number_of_basket=10,
                                                               min_basket_size=1,
                                                               max_basket_size=float('inf'),
                                                               min_item_occurrences=2,
                                                               item2category=item2category)

    customers_train_set, new2old, old2new = remap_items_with_data(customers_train_set)

    print(datetime.datetime.now(), 'Customers for test', len(customers_train_set),
          '%.2f%%' % (100.0 * len(customers_train_set) / len(customers_data)))

    print(datetime.datetime.now(), 'Create and build models')
    customers_recsys = dict()
    start_time = datetime.datetime.now()
    for customer_id in list(customers_train_set.keys()):
        print(datetime.datetime.now(), customer_id)
        customer_train_set = customers_train_set[customer_id]
        tbp = TBP()
        tbp.build_model(customer_train_set)
        customers_recsys[customer_id] = tbp

    end_time = datetime.datetime.now()
    print(datetime.datetime.now(), 'Models built in', end_time - start_time)

    print(datetime.datetime.now(), 'Perform predictions')
    performances = defaultdict(list)
    start_time = datetime.datetime.now()
    for customer_id in customers_train_set:
        if customer_id not in customers_recsys:
            continue

        tbp = customers_recsys[customer_id]
        customer_data = customers_train_set[customer_id]['data']
        next_baskets = customers_test_set[customer_id]['data']
        print("customer: ", customer_id)

        for next_basket_id in next_baskets:
            day_of_next_purchase = datetime.datetime.strptime(next_basket_id[0:10], '%Y_%m_%d')
            pred_basket = tbp.predict(customer_data, day_of_next_purchase, nbr_patterns=None, pred_length=pred_length)
            pred_basket = set([new2old[item] for item in pred_basket])
            print("basket: ", next_basket_id)
            print("Prediction")
            print(pred_basket)

            next_basket = next_baskets[next_basket_id]['basket']
            next_basket = set(next_basket.keys())
            print("Actual next basket")
            print(next_basket)

            evaluation = evaluate_prediction(next_basket, pred_basket)
            performances[customer_id].append(evaluation)
            print("Performance:")
            print(performances[customer_id])
            print("\n")

    end_time = datetime.datetime.now()
    print(datetime.datetime.now(), 'Prediction performed in', end_time - start_time)

    f1_values = list()
    for customer_id in performances:
        for evaluation in performances[customer_id]:
            f1_values.append(evaluation['f1_score'])

    stats = calculate_aggregate(f1_values)
    print(datetime.datetime.now(), 'TBP', 'avg', stats['avg'])
    print(stats)
    print(f1_values)




if __name__ == "__main__":
    main()

    # print(customers_data)

    # # Estimate basket lengths for each month
    # month_ebl = estimate_month_basket_length(customers_data[0])

    # # Plot the results
    # months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    # plt.figure(figsize=(10, 6))
    # plt.bar(months, month_ebl, color='skyblue')
    # plt.xlabel('Month')
    # plt.ylabel('Estimated Basket Length (EBL)')
    # plt.title('Estimated Basket Length for Each Month')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()

    # # Extract relevant values for the boxplot
    # median = stats['med']
    # q1 = stats['25p']
    # q3 = stats['75p']
    # whisker_low = stats['10p']
    # whisker_high = stats['90p']

    # # Create a dictionary with the boxplot data
    # boxplot_data = {
    #     'whisker_low': whisker_low,
    #     'q1': q1,
    #     'median': median,
    #     'q3': q3,
    #     'whisker_high': whisker_high
    # }

    # # Generate boxplot
    # fig, ax = plt.subplots()
    # ax.boxplot([[boxplot_data['whisker_low'], boxplot_data['q1'], boxplot_data['median'], boxplot_data['q3'], boxplot_data['whisker_high']]],
    #            vert=False, patch_artist=True,
    #            boxprops=dict(facecolor='lightblue', color='blue'),
    #            medianprops=dict(color='red'),
    #            whiskerprops=dict(color='blue'),
    #            capprops=dict(color='blue'))

    # # Set plot title and labels
    # ax.set_title('Boxplot of Distribution')
    # ax.set_yticklabels(['Distribution'])

    # # Display the plot
    # plt.show()