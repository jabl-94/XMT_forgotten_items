from collections import defaultdict, Counter
from typing import Dict, Set, List, Tuple
import datetime
from statistics import mean, median
import numpy as np
from forgotten_items.competitors.tars import TARSTree, calculate_intervals_support, calcualte_active_rp, calcualte_item_score


class TXMT:
    def __init__(self, new2old: Dict[int, int]):
        self.new2old = new2old
        self.old2new = {v: k for k, v in new2old.items()}
        self.customer_baskets = None
        self.state = 'initialized'

        # TARS components
        self.tars_tree = None
        self.tars_patterns = None
        self.rs_intervals_support = None

        # XMT tracking structures
        self.item_frequencies = Counter()
        self.purchase_intervals = defaultdict(list)
        self.avg_intervals = {}
        self.last_purchase_dates = {}
        self.co_purchase_counts = defaultdict(Counter)
        self.repurchase_patterns = defaultdict(Counter)
        self.total_repurchase_opportunities = defaultdict(int)
        self.seasonal_patterns = defaultdict(lambda: defaultdict(int))

        # Parameters
        self.min_purchases = 5
        self.large_basket_threshold = 10
        self.repurchase_window = 2
        self.seasonal_boost = 1.5
        self.max_interval_multiplier = 3.0
        self.max_absolute_days = 90

        # Seasons definition
        self.seasons = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'fall': [9, 10, 11]
        }

    def _parse_date(self, date_str: str) -> datetime.datetime:
        return datetime.datetime.strptime('_'.join(date_str.split('_')[:3]), '%Y_%m_%d')

    def _get_season(self, date: datetime.datetime) -> str:
        month = date.month
        for season, months in self.seasons.items():
            if month in months:
                return season
        return 'unknown'

    def _calculate_item_frequencies(self):
        total_baskets = len(self.customer_baskets['data'])
        for date_key, basket_data in self.customer_baskets['data'].items():
            for item in basket_data['basket']:
                self.item_frequencies[item] += 1

        for item in self.item_frequencies:
            self.item_frequencies[item] /= total_baskets

    def _analyze_purchase_intervals(self):
        sorted_baskets = sorted(
            self.customer_baskets['data'].items(),
            key=lambda x: self._parse_date(x[0])
        )
        last_purchase = {}

        for date_str, basket_data in sorted_baskets:
            current_date = self._parse_date(date_str)
            for item in basket_data['basket']:
                if item in last_purchase:
                    interval = (current_date - last_purchase[item]).days
                    if interval > 0:
                        self.purchase_intervals[item].append(interval)
                last_purchase[item] = current_date
                self.last_purchase_dates[item] = date_str

        for item, intervals in self.purchase_intervals.items():
            if len(intervals) >= self.min_purchases:
                self.avg_intervals[item] = median(intervals)
    def _analyze_repurchase_patterns(self):
        """Analyze which items are frequently repurchased within a short window"""
        sorted_baskets = sorted(
            self.customer_baskets['data'].items(),
            key=lambda x: self._parse_date(x[0])
        )

        for i in range(len(sorted_baskets) - 1):
            current_date = self._parse_date(sorted_baskets[i][0])
            current_basket = set(sorted_baskets[i][1]['basket'].keys())

            # Only analyze repurchases for large baskets
            if len(current_basket) >= self.large_basket_threshold:
                for item in current_basket:
                    self.total_repurchase_opportunities[item] += 1

                # Look at subsequent baskets within window
                for j in range(i + 1, len(sorted_baskets)):
                    compare_date = self._parse_date(sorted_baskets[j][0])
                    days_between = (compare_date - current_date).days

                    if days_between > self.repurchase_window:
                        break

                    compare_basket = set(sorted_baskets[j][1]['basket'].keys())
                    repurchased_items = current_basket & compare_basket

                    for item in repurchased_items:
                        self.repurchase_patterns[item][days_between] += 1

    def _analyze_co_purchases(self):
        for basket_data in self.customer_baskets['data'].values():
            basket_items = set(basket_data['basket'].keys())
            for item1 in basket_items:
                for item2 in basket_items:
                    if item1 != item2:
                        self.co_purchase_counts[item1][item2] += 1

    def _analyze_seasonal_patterns(self):
        for date_str, basket_data in self.customer_baskets['data'].items():
            date = self._parse_date(date_str)
            season = self._get_season(date)
            for item in basket_data['basket']:
                self.seasonal_patterns[item][season] += 1

        for item in self.seasonal_patterns:
            total = sum(self.seasonal_patterns[item].values())
            if total > 0:
                for season in self.seasonal_patterns[item]:
                    self.seasonal_patterns[item][season] /= total

    def build_model(self, customer_train_set):
        """Build both XMT and TARS models"""
        self.customer_baskets = customer_train_set

        # Build XMT components
        self._calculate_item_frequencies()
        self._analyze_purchase_intervals()
        self._analyze_repurchase_patterns()
        self._analyze_co_purchases()
        self._analyze_seasonal_patterns()

        # Build TARS components
        self.tars_tree = TARSTree(customer_train_set, root_value=None, root_count=None, root_timeseries=None)
        self.tars_patterns = self.tars_tree.mine_patterns(max_rec_dept=0, patterns_subset=None, nbr_patterns=None,
                                                          get_items_in_order_of_occurrences=True)
        self.rs_intervals_support = calculate_intervals_support(self.tars_patterns, self.tars_tree)

        self.state = 'built'

    def predict_f(self, current_basket: Set[str], basket_date: str, max_final_predictions: int = 3) -> Tuple[
        Set[str], List[str]]:
        if self.state != 'built':
            raise Exception('Model not built, prediction not available')

        # Map categories to internal IDs
        current_basket_mapped = set()
        for item in current_basket:
            if item in self.old2new:
                current_basket_mapped.add(self.old2new[item])

        current_date = self._parse_date(basket_date)
        current_season = self._get_season(current_date)

        # Generate initial full basket prediction (X't)
        x_t_prime_scores = {}
        for item in self.item_frequencies:
            if item in self.last_purchase_dates and item in self.avg_intervals:
                last_purchase_date = self._parse_date(self.last_purchase_dates[item])
                days_since_last = (current_date - last_purchase_date).days
                expected_interval = self.avg_intervals[item]

                if (days_since_last > expected_interval * self.max_interval_multiplier or
                        days_since_last > self.max_absolute_days):
                    continue

                base_score = self.item_frequencies[item]

                if (expected_interval <= days_since_last <=
                        expected_interval * self.max_interval_multiplier):
                    base_score *= 1.5

                if item in self.seasonal_patterns:
                    season_rate = self.seasonal_patterns[item][current_season]
                    if season_rate > 0.4:
                        base_score *= self.seasonal_boost

                x_t_prime_scores[item] = base_score

        # Generate X't (expected full basket)
        min_basket_size = max(
            len(current_basket_mapped) + max_final_predictions,
            int(1.5 * len(current_basket_mapped))
        )
        x_t_prime = set(sorted(
            x_t_prime_scores.keys(),
            key=lambda x: x_t_prime_scores[x],
            reverse=True
        )[:min_basket_size])

        # Calculate potentially missing items (X't - Xt)
        potential_missing = x_t_prime - current_basket_mapped

        # Get TARS scores for potential missing items
        rs_purchases, _ = calcualte_active_rp(
            self.customer_baskets['data'],
            self.rs_intervals_support,
            current_date
        )

        # Refine with TARS patterns
        self.tars_patterns = self.tars_tree.mine_patterns(
            max_rec_dept=1,
            patterns_subset=rs_purchases,
            nbr_patterns=None,
            get_items_in_order_of_occurrences=False
        )
        self.rs_intervals_support = calculate_intervals_support(self.tars_patterns, self.tars_tree)

        # Get TARS scores for potential missing items
        tars_scores = calcualte_item_score(self.tars_tree, rs_purchases, self.rs_intervals_support)

        # Calculate final scores only for potential missing items
        final_scores = {}
        for item in potential_missing:
            # Start with XMT base score
            xmt_score = x_t_prime_scores.get(item, 0)

            # Add co-purchase evidence
            co_purchase_boost = 0
            for basket_item in current_basket_mapped:
                if basket_item in self.co_purchase_counts[item]:
                    co_occurrence = self.co_purchase_counts[item][basket_item]
                    if co_occurrence > 5:
                        co_purchase_boost += 0.2

            # Strong repurchase patterns
            repurchase_boost = 0
            if (item in self.repurchase_patterns and
                    self.total_repurchase_opportunities[item] >= 5):  # Minimum opportunities threshold
                repurchase_count = sum(self.repurchase_patterns[item].values())
                repurchase_rate = repurchase_count / self.total_repurchase_opportunities[item]
                if repurchase_rate > 0.3:  # Strong repurchase pattern
                    repurchase_boost += 0.5

            # Add TARS score
            tars_score = tars_scores.get(item, 0)
            if tars_score > 0:
                max_tars = max(tars_scores.values())
                tars_score = tars_score / max_tars

            # Final xmt score
            xmt_score += co_purchase_boost + repurchase_boost

            # Final sum
            final_scores[item] = xmt_score + tars_score

        # Select top predictions
        prediction_items = sorted(
            final_scores.keys(),
            key=final_scores.get,
            reverse=True
        )[:max_final_predictions]

        final_prediction = set(self.new2old[item] for item in prediction_items)

        # Generate explanations
        explanations = []
        for item in prediction_items:
            category_name = self.new2old[item]
            explanation = f"Item {category_name} might be forgotten because:\n"

            # XMT-based explanations
            if item in self.last_purchase_dates and item in self.avg_intervals:
                last_purchase = self._parse_date(self.last_purchase_dates[item])
                days_since = (current_date - last_purchase).days
                avg_interval = self.avg_intervals[item]

                if days_since <= avg_interval * self.max_interval_multiplier:
                    explanation += (f"1. Last purchased {days_since} days ago "
                                    f"(typically bought every {avg_interval:.1f} days)\n")

            # Co-purchase patterns
            relevant_items = []
            for basket_item in current_basket_mapped:
                if basket_item in self.co_purchase_counts[item]:
                    relevant_items.append((
                        self.new2old[basket_item],
                        self.co_purchase_counts[item][basket_item]
                    ))

            if relevant_items:
                explanation += "2. Often bought with current basket items: "
                top_pairs = sorted(relevant_items, key=lambda x: x[1], reverse=True)[:3]
                explanation += ", ".join(f"{pair[0]}" for pair in top_pairs) + "\n"

            # TARS-based explanation
            if item in tars_scores and tars_scores[item] > 0:
                explanation += f"3. TARS pattern analysis suggests this item is likely to be needed "
                explanation += f"(confidence: {tars_scores[item] / max(tars_scores.values()):.1%})\n"

            # Repurchase pattern
            if item in self.repurchase_patterns and self.total_repurchase_opportunities[item] > 0:
                repurchase_count = sum(self.repurchase_patterns[item].values())
                repurchase_rate = repurchase_count / self.total_repurchase_opportunities[item]
                if repurchase_rate > 0.3:
                    explanation += f"4. Often repurchased soon after large shopping trips "
                    explanation += f"({repurchase_rate:.1%} of opportunities)\n"

            explanations.append(explanation)

        return final_prediction, explanations