import os
import gc

from forgotten_items.contenders.txmt import *

from forgotten_items.imports.utilities.data_management import *
from forgotten_items.imports.evaluation.calculate_aggregate_statistics import *
from forgotten_items.imports.utilities.cat_map import cod_mkt_cat2name
from forgotten_items.imports.utilities.cat_remap import remap_categories


def main():

    def evaluate_forgotten_items_prediction(prediction, future_basket, window_size=1):
        """
        Evaluate forgotten items prediction by checking if predicted items appear in future purchases
        """
        if not prediction:
            print("Warning: Empty prediction set")
            return None

        if not future_basket:
            print("Warning: No future baskets available")
            return None

        # Convert future_baskets to sorted list of (date, basket) tuples
        future_purchases = sorted(
            [(datetime.datetime.strptime('_'.join(date.split('_')[:3]), '%Y_%m_%d'), basket['basket'])
             for date, basket in future_basket.items()],
            key=lambda x: x[0]
        )[:window_size]

        if not future_purchases:
            print("Warning: No valid future purchases found")
            return None

        # Combine all future purchases into one set
        all_future_items = set()
        for _, basket in future_purchases:
            all_future_items.update(basket.keys())

        if not all_future_items:
            print("Warning: No items found in future baskets")
            return None

        # Calculate metrics
        true_positives = len(prediction & all_future_items)
        false_positives = len(prediction - all_future_items)
        false_negatives = len(all_future_items - prediction)

        # Calculate evaluation metrics
        precision = true_positives / len(prediction) if len(prediction) > 0 else 0
        recall = true_positives / len(all_future_items) if len(all_future_items) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': float(precision),  # Convert to native Python float
            'recall': float(recall),
            'f1_score': float(f1_score),
            'true_positives': int(true_positives),  # Convert to native Python int
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'prediction_size': int(len(prediction)),
            'future_items_size': int(len(all_future_items))
        }


    def safe_calculate_aggregate(values):
        """Calculate aggregate statistics safely with JSON-serializable outputs"""
        if not values:
            return {
                'avg': 0.0,
                'std': 0.0,
                'var': 0.0,
                'min': 0.0,
                'max': 0.0,
                '10p': 0.0,
                '25p': 0.0,
                '50p': 0.0,
                '75p': 0.0,
                '90p': 0.0
            }

        values = [float(v) for v in values if v is not None]  # Convert to native Python float
        if not values:
            return {
                'avg': 0.0,
                'std': 0.0,
                'var': 0.0,
                'min': 0.0,
                'max': 0.0,
                '10p': 0.0,
                '25p': 0.0,
                '50p': 0.0,
                '75p': 0.0,
                '90p': 0.0
            }

        return {
            'avg': float(np.mean(values)),
            'std': float(np.std(values)),
            'var': float(np.var(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            '10p': float(np.percentile(values, 10)),
            '25p': float(np.percentile(values, 25)),
            '50p': float(np.percentile(values, 50)),
            '75p': float(np.percentile(values, 75)),
            '90p': float(np.percentile(values, 90))
        }

    def get_dataset_path():
        """Get the path to the datasets directory relative to this script."""
        # This script is in baseline_testing/, so we need to go up one level to reach datasets/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(os.path.dirname(current_dir), 'datasets')
        return dataset_dir

    dataset_dir = get_dataset_path()
    path1 = os.path.join(dataset_dir, 'coop100.json')
    path2 = os.path.join(dataset_dir, 'tafeng.json')


    # Configuration
    dataset_name = 'coop'
    coop_level = 'category'
    test_partition_type = 'standard'
    model_name = "txmt"
    min_pred_length, max_pred_length = 2, 21
    pred_lengths = list(range(min_pred_length, max_pred_length))
    pred_length = 5  # for TBP

    print(datetime.datetime.now(), 'Read dataset', dataset_name)

    # Read dataset
    if dataset_name == 'tafeng':
        dataset = read_data(path2)
        item2category = None
    elif dataset_name.startswith('coop'):
        dataset = read_data(path1)
        item2category = get_item2category('market.csv',
                                          category_index['categoria']) if coop_level == 'category' else None
    else:
        raise ValueError('Unknown dataset')

    print(datetime.datetime.now(), 'Customers', len(dataset))

    # Set split mode
    split_mode = {'fixed': 'loo', 'random': 'rnd', 'standard': '70', 'last4': '-4', 'last2': '-2'}.get(test_partition_type)
    if not split_mode:
        raise ValueError('Unknown test partition type')

    print(datetime.datetime.now(), 'Partition dataset into train / test')
    customers_train_set, customers_test_set = split_train_test(dataset,
                                                               split_mode=split_mode,
                                                               min_number_of_basket=10,
                                                               min_basket_size=1,
                                                               max_basket_size=float('inf'),
                                                               min_item_occurrences=2,
                                                               item2category=item2category,
                                                               large_basket=10,
                                                               max_days=1,
                                                               min_forgotten_items=10)

    customers_test_set = remap_categories(customers_test_set, cod_mkt_cat2name)
    customers_train_set = remap_categories(customers_train_set, cod_mkt_cat2name)

    # Remap items but only get new2old mapping
    customers_train_set, new2old, old2new = remap_items_with_data(customers_train_set)

    print(datetime.datetime.now(), 'Customers for test', len(customers_train_set),
          '%.2f%%' % (100.0 * len(customers_train_set) / len(dataset)))

    performance = defaultdict(list)
    f1_values = []

    start_time = datetime.datetime.now()

    # Example usage in main evaluation loop
    valid_predictions_count = 0
    total_predictions_attempted = 0

    for customer_id in list(customers_train_set.keys()):
        print(f"\nProcessing customer {customer_id}")
        customer_train_set = customers_train_set[customer_id]

        print(list(customers_train_set.keys()))

        # Build model
        model = TXMT(new2old)
        model.build_model(customer_train_set)

        # Get test baskets
        test_baskets = customers_test_set[customer_id]['data']

        # Sort test baskets by date
        sorted_test_baskets = sorted(
            test_baskets.items(),
            key=lambda x: datetime.datetime.strptime('_'.join(x[0].split('_')[:3]), '%Y_%m_%d')
        )

        # For each basket except the last one (need 1 basket for evaluation)
        # for i in range(len(sorted_test_baskets) - 1):
        total_predictions_attempted += 1
        current_basket_date, current_basket_data = sorted_test_baskets[0]
        current_basket = set(current_basket_data['basket'].keys())

        print(f"\nProcessing basket at date: {current_basket_date}")
        print(f"Current basket size: {len(current_basket)}")

        # Get future baskets for evaluation
        future_basket = dict(sorted_test_baskets[1:2])
        print(f"Number of future baskets available: {len(future_basket)}")

        # Get prediction
        pred_basket, explanations = model.predict_f(current_basket,
                                                current_basket_date,
                                                pred_length)
        # pred_basket = set(cod_mkt_cat2name.get(new2old[item], new2old[item]) for item in pred_set)

        print(f"Prediction size: {len(pred_basket)}")

        # Evaluate prediction against future purchases
        evaluation = evaluate_forgotten_items_prediction(pred_basket, future_basket)
        if evaluation is not None:
            valid_predictions_count += 1
            performance[customer_id].append(evaluation)
            f1_values.append(evaluation['f1_score'])

            print(f"Current basket: {current_basket}")
            print(f"Future basket: {future_basket}")
            print(f"Predicted forgotten items: {pred_basket}")
            print(f"Performance: {evaluation}")
            print("Explanations:")
            for explanation in explanations:
                print(explanation)
                # print()
        else:
            print("Warning: Could not evaluate this prediction")

        # Clean up to free memory
        del model
        gc.collect()

    end_time = datetime.datetime.now()
    print("\nEvaluation Summary:")
    print(f"Total predictions attempted: {total_predictions_attempted}")
    print(f"Valid predictions: {valid_predictions_count}")
    print(f"Success rate: {(valid_predictions_count / total_predictions_attempted) * 100:.2f}%")
    print(f'Time taken: {end_time - start_time}')

    # Calculate and print statistics
    stats = safe_calculate_aggregate(f1_values)
    print("\nPerformance Statistics:")
    print(f"Average F1 Score: {stats['avg']:.4f}")
    print(f"Standard Deviation: {stats['std']:.4f}")
    print(f"Min/Max F1: {stats['min']:.4f}/{stats['max']:.4f}")
    print(f"Median F1 (50th percentile): {stats['50p']:.4f}")
    print("\nDetailed Statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()