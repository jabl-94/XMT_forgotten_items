import random
from datetime import datetime
from typing import Dict, List, Tuple, Union


class IntervalBasedPredictor:
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        self.alpha = alpha
        self.beta = beta
        self.customers: Dict[int, Dict] = {}

    def _parse_date(self, date_input: Union[str, List[str]]) -> datetime:
        if isinstance(date_input, list):
            date_string = '_'.join(date_input[:3])
        else:
            date_string = '_'.join(date_input.split('_')[:3])
        return datetime.strptime(date_string, '%Y_%m_%d')

    def predict_basket(self, customer_id: int, visit_date: Union[str, List[str]], k: int = None) -> Tuple[
        List[Tuple[int, int, float]], set[int], List[int]]:
        # Check if the customer exists in our records
        if customer_id not in self.customers:
            return [], set(), []  # Return empty results if customer not found

        customer = self.customers[customer_id]
        Ti = customer['Ti']  # Average time between visits for this customer
        visit_time = self._parse_date(visit_date).timestamp()  # Convert visit date to timestamp

        predicted_basket = []
        possibly_forgotten = []

        # Iterate through each item the customer has previously purchased
        for item_id, item_data in customer['items'].items():
            Gj, Lj, Uj, Qj = item_data
            # Gj: Average time between purchases of this item
            # Lj: Last purchase time of this item
            # Uj: Number of times this item was purchased
            # Qj: Average quantity purchased of this item

            if Lj + Gj <= visit_time:
                # If the expected purchase time has passed, add to predicted basket
                predicted_basket.append((item_id, Uj, Qj))
                possibly_forgotten.append(item_id)
            elif Lj + Gj > visit_time and Lj + Gj < visit_time + Ti:
                # If the expected purchase time is within the next visit interval
                r = random.random()
                if r >= (Lj + Gj - visit_time) / Ti:
                    # Probabilistically add to predicted basket
                    predicted_basket.append((item_id, Uj, Qj))
                    possibly_forgotten.append(item_id)

        # print("All possibly forgotten items:", possibly_forgotten)
        # print("Sorted predicted basket before limiting:", predicted_basket)

        # Sort the predicted basket by number of purchases (Uj), descending
        predicted_basket.sort(key=lambda x: x[1], reverse=True)

        # Limit the basket size if k is specified
        if k is not None:
            predicted_basket = predicted_basket[:k]
            # print(f"Final predicted basket (limited to {k} items):", predicted_basket)
        else:
            # print("Final predicted basket (unlimited):", predicted_basket)
            print()
        # Create a set of predicted item IDs
        pred_set = set(item[0] for item in predicted_basket)

        return predicted_basket, pred_set, possibly_forgotten

    def build_model(self, customer_data: Dict):
        customer_id = customer_data['customer_id']
        if customer_id not in self.customers:
            self.customers[customer_id] = {
                'Ti': 0,
                'last_visit': None,
                'items': {}
            }

        customer = self.customers[customer_id]
        visits = sorted(customer_data['data'].items(), key=lambda x: self._parse_date(x[0]))

        for visit_date, visit_data in visits:
            # Convert the visit date string to a Unix timestamp
            visit_time = self._parse_date(visit_date).timestamp()

            # Extract the basket information for this visit
            basket = visit_data['basket']

            # Update time between visits
            if customer['last_visit'] is None:
                # If this is the first recorded visit, simply set the last visit time
                customer['last_visit'] = visit_time
            else:
                if customer['Ti'] == 0:
                    # If Ti (average time between visits) hasn't been set yet,
                    # calculate it as the difference between current and last visit
                    customer['Ti'] = visit_time - customer['last_visit']
                else:
                    # Update Ti using an exponential moving average
                    # Formula: new_Ti = α * old_Ti + (1 - α) * (current_interval)
                    # where α (alpha) is a smoothing factor between 0 and 1
                    customer['Ti'] = (
                            self.alpha * customer['Ti'] +
                            (1 - self.alpha) * (visit_time - customer['last_visit'])
                    )

                # Update the last visit time for the next iteration
                customer['last_visit'] = visit_time

            # Update item data
            for item_id, item_info in basket.items():
                item_id = int(item_id)
                quantity = item_info[0]

                if item_id not in customer['items']:
                    customer['items'][item_id] = [0, visit_time, 1, quantity]  # [Gj, Lj, Uj, Qj]
                else:
                    Gj, Lj, Uj, Qj = customer['items'][item_id]

                    # Update Gj (average time between purchases)
                    new_Gj = self.alpha * Gj + (1 - self.alpha) * (visit_time - Lj)

                    # Update Qj (average quantity purchased)
                    new_Qj = self.beta * Qj + (1 - self.beta) * quantity

                    # Update Uj (number of purchases) and Lj (last purchase time)
                    new_Uj = Uj + 1
                    new_Lj = visit_time

                    customer['items'][item_id] = [new_Gj, new_Lj, new_Uj, new_Qj]

    def calculate_performance(self, predicted_basket: set, actual_basket: set) -> float:

        # This method calculates ρ (rho), which represents the expected fractional increase in purchased items
        # due to reminding customers about potentially forgotten items.

        predicted_items = set(int(item) for item in predicted_basket)
        actual_items = set(int(item) for item in actual_basket)

        correct_predictions = len(predicted_items.intersection(actual_items))
        s = correct_predictions / len(actual_items) if actual_items else 0

        F = len(actual_items)
        B = len(predicted_items)

        rho = (s * F) / (B ** 2) if B > 0 else 0

        return round(rho, 3)

    def calculate_revenue_increase(self, predicted_basket: set, actual_basket: set, gamma: float = 0.05) -> float:

        # Calculates the expected increase in revenue based on the paper's formula.
        #
        # Args:
        # predicted_basket: Set of item IDs predicted to be in the basket
        # actual_basket: Set of item IDs actually in the basket
        # gamma: factor representing customer forgetfulness (default 0.05 or 5%)
        #
        # Returns:
        # float: The expected increase in revenue

        s = self.calculate_performance(predicted_basket, actual_basket)
        B = len(actual_basket)
        F = gamma * B  # Estimated number of forgotten items
        rev = (s * F) / B
        return rev


# Example usage remains the same


# # Example usage
# ibp = IntervalBasedPredictor()
#
# # Sample data
# customer_data = {
#     'customer_id': 0,
#     'data': {
#         '2012_07_12_1227': {'basket': {'559': [1.0, 1.0], '2065': [1.0, 1.0], '4600': [1.0, 1.0]}},
#         '2011_03_19_887': {'basket': {'150': [1.0, 1.0], '1647': [1.0, 1.0], '2089': [1.0, 1.0], '4052': [1.0, 1.0],
#                                       '3690': [1.0, 1.0], '3682': [1.0, 1.0], '445': [1.0, 1.0], '2496': [1.0, 1.0],
#                                       '4655': [1.0, 1.0], '3666': [1.0, 1.0], '2402': [1.0, 1.0], '3739': [1.0, 1.0],
#                                       '2198': [1.0, 1.0], '2065': [1.0, 1.0]}},
#         '2012_05_31_1194': {'basket': {'2731': [1.0, 1.0], '2138': [1.0, 1.0], '391': [1.0, 1.0], '3086': [1.0, 1.0],
#                                        '2065': [1.0, 1.0], '162': [1.0, 1.0]}},
#         '2012_08_10_1249': {'basket': {'2471': [1.0, 1.0], '1263': [1.0, 1.0], '4132': [1.0, 1.0], '699': [1.0, 1.0],
#                                        '2176': [1.0, 1.0], '4849': [1.0, 1.0], '4600': [1.0, 1.0]}}
#     },
#     'customer_id': 1,
#     'data': {
#         '2012_07_12_1227': {'basket': {'559': [1.0, 1.0], '2065': [1.0, 1.0], '4600': [1.0, 1.0]}},
#         '2011_03_19_887': {'basket': {'150': [1.0, 1.0], '1647': [1.0, 1.0], '2089': [1.0, 1.0], '4052': [1.0, 1.0],
#                                       '3690': [1.0, 1.0], '3682': [1.0, 1.0], '445': [1.0, 1.0], '2496': [1.0, 1.0],
#                                       '4655': [1.0, 1.0], '3666': [1.0, 1.0], '2402': [1.0, 1.0], '3739': [1.0, 1.0],
#                                       '2198': [1.0, 1.0], '2065': [1.0, 1.0]}},
#         '2012_05_31_1194': {'basket': {'2731': [1.0, 1.0], '2138': [1.0, 1.0], '391': [1.0, 1.0], '3086': [1.0, 1.0],
#                                        '2065': [1.0, 1.0], '162': [1.0, 1.0]}},
#         '2012_08_10_1249': {'basket': {'2471': [1.0, 1.0], '1263': [1.0, 1.0], '4132': [1.0, 1.0], '699': [1.0, 1.0],
#                                        '2176': [1.0, 1.0], '4849': [1.0, 1.0], '4600': [1.0, 1.0]}}
#     }
# }
#
# # Training
# ibp.learn_time_intervals(customer_data)


# # Prediction
# for i in customer_data:
#     predicted_basket = ibp.predict_basket(1, '2012_09_15_1285', k=8)
#     print("Predicted basket:", predicted_basket)
#
#     # Performance calculation
#     actual_basket = {'2471': [1.0, 1.0], '1263': [1.0, 1.0], '4132': [1.0, 1.0], '699': [1.0, 1.0], '2176': [1.0, 1.0]}
#     performance = ibp.calculate_performance(predicted_basket, actual_basket)
#     print("Performance (rho):", performance)

    # , visit_date = keys_string