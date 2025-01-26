import json
import random
import datetime
import pandas as pd
import numpy as np
from collections import defaultdict

# def read_data(filename):
#     customers_data = dict()
#     data = open(filename, 'r')
#     id = 0
#     for row in data:
#         customer_data = json.loads(row)
#         customer_id = customer_data['customer_id']
#         # if customer_id in customers_data:
#         id += 1
#         customers_data[id] = customer_data
#     data.close()
#
#     return customers_data

def read_data(filename):
    customers_data = dict()
    data = open(filename, 'r')
    for row in data:
        customer_data = json.loads(row)
        customer_id = customer_data['customer_id']
        customers_data[customer_id] = customer_data
    data.close()

    return customers_data


def write_data(filename, customers_data):
    newfile = open(filename, 'w')
    for customer_id in customers_data:
        customer_json_data = json.dumps(customers_data[customer_id])
        newfile.write(customer_json_data + '\n')
    newfile.flush()
    newfile.close()


def get_item2category(filename, category_level=7):
    df = pd.read_csv(filename, delimiter=';', skipinitialspace=True)
    item2category = dict()
    for row in df.values:
        cod_mkt_id = str(row[0])
        cod_mkt = row[1]
        item2category[cod_mkt_id] = cod_mkt[:category_level]
    return item2category


def get_date(basket_id, customer_data):
    basket_data = customer_data['data'][basket_id]
    date_str = '%s-%s-%s %s:%s:%s' % (
        basket_data['anno'], basket_data['mese_n'], basket_data['giorno_n'],
        basket_data['ora'], basket_data['minuto'], '0')
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    return date


def split_train_test_og(customers_data, split_mode='loo', min_number_of_basket=10, min_basket_size=1,
                     max_basket_size=float('inf'), min_item_occurrences=1, item2category=None):
    """
    split_mode = 'loo' leave one out<
    split_mode = 70 training percentage (70-30)
    min_number_of_basket = 10 minimum number of baskets per customer
    min_basket_size = 1 minimum basket size
    max_basket_size = float('inf') maximum basket size
    """

    customers_train_set = dict()
    customers_test_set = dict()
    for customer_id in customers_data:
        customer_data = customers_data[customer_id]

        if len(customer_data['data']) < min_number_of_basket:
            continue

        if item2category is not None:
            for basket_id in customer_data['data']:
                basket = customer_data['data'][basket_id]['basket']
                basket_category = dict()
                for item in basket:
                    if item != 'null' and item in item2category:
                        category = item2category[item]
                        length = len(
                            basket[item])
                        if category not in basket_category:
                            basket_category[category] = [0] * length
                        for i in range(length):
                            basket_category[category][i] += basket[item][i]
                customer_data['data'][basket_id]['basket'] = basket_category

        train_set = dict()
        test_set = dict()

        if split_mode == 'loo':
            split_index = len(customer_data['data']) - 1
        elif split_mode == 'rnd':
            max_test_nbr = int(np.round(len(customer_data['data']) * 0.1))
            if max_test_nbr < 2:
                test_nbr = 1
            else:
                test_nbr = random.randint(2, max_test_nbr)
            split_index = len(customer_data['data']) - test_nbr
        else:
            if int(split_mode) > 0:
                train_percentage = int(split_mode)
                split_index = int(len(customer_data['data']) * train_percentage / 100.0)
            else:
                test_nbr = -int(split_mode)
                split_index = len(customer_data['data']) - test_nbr

        sorted_basket_ids = sorted(customer_data['data'])
        train_basket_ids = sorted_basket_ids[:split_index]
        test_basket_ids = sorted_basket_ids[split_index:]

        if min_item_occurrences > 1:
            item_count = defaultdict(int)
            for basket_id in train_basket_ids:
                basket = customer_data['data'][basket_id]['basket'].keys()
                # print(basket)
                for item in basket:
                    item_count[item] += 1

            for basket_id in train_basket_ids:
                basket = customer_data['data'][basket_id]['basket'].keys()
                # print(basket)
                items_to_remove = []
                for item in basket:
                    if item_count[item] < min_item_occurrences:
                        # print(item)
                        # print(item_count[item])
                        items_to_remove.append(item)

                for item in items_to_remove:
                    del customer_data['data'][basket_id]['basket'][item]

        for basket_id in train_basket_ids:
            basket = customer_data['data'][basket_id]['basket'].keys()
            if min_basket_size - 1 < len(basket) < max_basket_size:
                train_set[basket_id] = customer_data['data'][basket_id]

        for basket_id in test_basket_ids:
            basket = customer_data['data'][basket_id]['basket'].keys()
            if min_basket_size - 1 < len(basket) < max_basket_size:
                test_set[basket_id] = customer_data['data'][basket_id]

        if len(train_set) == 0 or len(test_set) == 0:
            continue

        customers_train_set[customer_id] = {'customer_id': customer_id, 'data': train_set}
        customers_test_set[customer_id] = {'customer_id': customer_id, 'data': test_set}

    return customers_train_set, customers_test_set



# Version that splits by split_type, and searches for forgotten-item baskets
def split_train_test(customers_data, split_mode='loo', min_number_of_basket=10, min_basket_size=1,
                     max_basket_size=float('inf'), min_item_occurrences=1, item2category=None,
                     large_basket=10, max_days=2, min_forgotten_items=10):
    """
    Splits customer data into training and test sets based on specified criteria, with forgotten-item pair logic.
    split_mode = 'loo' leave one out<
    split_mode = 70 training percentage (70-30)
    min_number_of_basket = 10 minimum number of baskets per customer
    min_basket_size = 1 minimum basket size
    max_basket_size = float('inf') maximum basket size
    large_basket = int, number of items a basket must have for it to be considered a large basket
    max_days = int, max time threshold in days for a subsequent basket to be considered as a repurchase
    min_forgotten_items = int, minimum items threshold a repurchase must have to be confirmed as forgotten_items basket
    """

    customers_train_set = dict()
    customers_test_set = dict()

    for customer_id in customers_data:
        customer_data = customers_data[customer_id]

        # Ensure minimum basket count per customer
        if len(customer_data['data']) < min_number_of_basket:
            continue

        # Optional: Convert items to categories if item2category is provided
        if item2category is not None:
            for basket_id in customer_data['data']:
                basket = customer_data['data'][basket_id]['basket']
                basket_category = dict()
                for item in basket:
                    if item != 'null' and item in item2category:
                        category = item2category[item]
                        length = len(basket[item])
                        if category not in basket_category:
                            basket_category[category] = [0] * length
                        for i in range(length):
                            basket_category[category][i] += basket[item][i]
                customer_data['data'][basket_id]['basket'] = basket_category

        # Initialize train and test sets
        train_set = dict()
        test_set = dict()

        # Determine the initial split index based on split_mode
        if split_mode == 'loo':
            split_index = len(customer_data['data']) - 1
        elif split_mode == 'rnd':
            max_test_nbr = int(np.round(len(customer_data['data']) * 0.1))
            test_nbr = random.randint(2, max_test_nbr if max_test_nbr >= 2 else 1)
            split_index = len(customer_data['data']) - test_nbr
        else:
            train_percentage = int(split_mode) if int(split_mode) > 0 else 100 + int(split_mode)
            split_index = int(len(customer_data['data']) * train_percentage / 100.0)

        # Sort basket ids and identify initial train and test basket ids
        sorted_basket_ids = sorted(customer_data['data'])
        initial_train_basket_ids = sorted_basket_ids[:split_index]
        initial_test_basket_ids = sorted_basket_ids[split_index:]

        # Additional Forgotten-Item Logic with Date Handling
        qualifying_pair_found = False

        for i in range(len(initial_test_basket_ids) - 1):
            # Get the first basket in the test set and the next basket to check forgotten-item logic
            basket_id = initial_test_basket_ids[i]
            next_basket_id = initial_test_basket_ids[i + 1]

            # Ensure first test basket is a 'large_basket' and the next one meets forgotten-item criteria
            basket = customer_data['data'][basket_id]['basket']
            next_basket = customer_data['data'][next_basket_id]['basket']

            # Check basket size for 'large_basket' and 'min_forgotten_items'
            if len(basket) >= large_basket and len(next_basket) >= min_forgotten_items:
                # Extract dates from basket ids to check the time gap
                date_str1 = '_'.join(basket_id.split('_')[:3])
                date_str2 = '_'.join(next_basket_id.split('_')[:3])
                date1 = datetime.datetime.strptime(date_str1, '%Y_%m_%d')
                date2 = datetime.datetime.strptime(date_str2, '%Y_%m_%d')
                date_difference = (date2 - date1).days

                # Check forgotten-item condition with max_days criteria
                forgotten_items = [item for item in next_basket if item not in basket]
                if date_difference <= max_days and len(forgotten_items) >= min_forgotten_items:
                    # Set qualifying test set and split index
                    split_index = initial_test_basket_ids.index(basket_id)
                    train_basket_ids = initial_train_basket_ids + initial_test_basket_ids[:split_index]
                    test_basket_ids = initial_test_basket_ids[split_index:]
                    qualifying_pair_found = True
                    break

        # Skip customer if no qualifying pair found
        if not qualifying_pair_found:
            continue

        # Filter baskets based on item occurrences and size limits
        if min_item_occurrences > 1:
            item_count = defaultdict(int)
            for basket_id in train_basket_ids:
                for item in customer_data['data'][basket_id]['basket']:
                    item_count[item] += 1

            for basket_id in train_basket_ids:
                items_to_remove = [item for item in customer_data['data'][basket_id]['basket']
                                   if item_count[item] < min_item_occurrences]
                for item in items_to_remove:
                    del customer_data['data'][basket_id]['basket'][item]

        # Assign baskets to train and test sets if they meet basket size criteria
        for basket_id in train_basket_ids:
            basket = customer_data['data'][basket_id]['basket']
            if min_basket_size - 1 < len(basket) < max_basket_size:
                train_set[basket_id] = customer_data['data'][basket_id]

        for basket_id in test_basket_ids:
            basket = customer_data['data'][basket_id]['basket']
            if min_basket_size - 1 < len(basket) < max_basket_size:
                test_set[basket_id] = customer_data['data'][basket_id]

        # Only add customer to results if both train and test sets have data
        if train_set and test_set:
            customers_train_set[customer_id] = {'customer_id': customer_id, 'data': train_set}
            customers_test_set[customer_id] = {'customer_id': customer_id, 'data': test_set}

    return customers_train_set, customers_test_set

def data2baskets(customer_data):
    baskets = list()
    for i, basket_id in enumerate(sorted(customer_data['data'])):
        basket_data = customer_data['data'][basket_id]['basket']
        basket = list()
        for item in basket_data:
            basket.append(item)
        baskets.append(basket)

    return baskets


def remap_items(baskets):
    new2old = dict()
    old2new = dict()
    new_baskets = list()
    for u, user_baskets in enumerate(baskets):
        new_user_baskets = list()
        for t, basket in enumerate(user_baskets):
            new_basket = list()
            for i in basket:
                if i not in old2new:
                    new_i = len(old2new)
                    old2new[i] = new_i
                    new2old[new_i] = i
                new_basket.append(old2new[i])
            new_user_baskets.append(new_basket)
        new_baskets.append(new_user_baskets)
    return new_baskets, new2old, old2new


def remap_items_with_data(baskets):
    new2old = dict()
    old2new = dict()
    new_baskets = dict()
    for customer_id in baskets:
        new_user_baskets = {'customer_id': customer_id, 'data': dict()}
        user_baskets = baskets[customer_id]
        for basket_id in user_baskets['data']:
            basket = user_baskets['data'][basket_id]['basket']
            new_basket = dict()
            for i in basket:
                if i not in old2new:
                    new_i = len(old2new)
                    old2new[i] = new_i
                    new2old[new_i] = i
                new_basket[old2new[i]] = basket[i]

            new_user_baskets['data'][basket_id] = dict()
            new_user_baskets['data'][basket_id]['basket'] = new_basket
        new_baskets[customer_id] = new_user_baskets

    return new_baskets, new2old, old2new


def get_items(baskets):
    items = dict()
    # print(baskets)
    # print(len(baskets))
    for user_baskets in baskets:
        # print(user_baskets)
        # print(len(user_baskets))
        for basket in user_baskets:
            # print(basket)
            # print(len(basket))
            for item in basket:
                # print(item)
                items[item] = 0
    return items


def count_users_items(baskets):
    # print(baskets)
    user_count = defaultdict(int)
    users_item_count = defaultdict(lambda: defaultdict(int))
    item_count = defaultdict(int)
    for u, user_basket in enumerate(baskets):

        # print(u, user_basket)

        user_item_count = defaultdict(int)
        num_purchases = 0
        for basket in user_basket:

            # print(basket)

            for item in basket:

                # print(item)

                num_purchases += 1.0
                item_count[item] += 1.0
                user_item_count[item] += 1.0

        user_count[u] = num_purchases
        users_item_count[u] = user_item_count

    return user_count, item_count, users_item_count


category_index = {
    'settore': 2,
    'reparto': 4,
    'categoria': 7,
    'sottocategoria': 9,
    'segmento': 11,
}
