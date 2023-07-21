import numpy as np
<<<<<<< HEAD
import pandas as pd

from photonai.processing.metrics import Scorer

=======
from photonai.processing.metrics import Scorer
>>>>>>> ohbm2023

class BaseConfigSelector:

    def prepare_metrics(self, list_of_non_failed_configs, metric):
        classification_metrics = ['balanced_accuracy', 'precision', 'recall', 'f1_score', 'matthews_corrcoef']
        regression_metrics = ['mean_absolute_error', 'mean_squared_error', 'explained_variance']

        # right now we can only do this ugly hack, sorry!
        is_it_classification = metric in classification_metrics
        all_metric_mean = {}
        all_metric_std = {}
        metric_list = classification_metrics if is_it_classification is True else regression_metrics
        for m in metric_list:
            all_metric_mean[m] = [c.get_test_metric(m, "mean") for c in list_of_non_failed_configs]
            all_metric_std[m] = [c.get_test_metric(m, "std") for c in list_of_non_failed_configs]
        # -----------------
        return all_metric_mean, all_metric_std, metric_list


class DefaultConfigSelector(BaseConfigSelector):

    def __call__(self, list_of_non_failed_configs, metric, fold_operation, maximize_metric):

        all_metrics_mean, all_metrics_std, _= self.prepare_metrics(list_of_non_failed_configs, metric)
        best_config_metric_values = [c.get_test_metric(metric, fold_operation) for c in list_of_non_failed_configs]

        if maximize_metric:
            # max metric
            best_config_metric_nr = np.argmax(best_config_metric_values)
        else:
            # min metric
            best_config_metric_nr = np.argmin(best_config_metric_values)

        best_config_outer_fold = list_of_non_failed_configs[best_config_metric_nr]

        return best_config_outer_fold


class RandomConfigSelector(BaseConfigSelector):

    def __call__(self, list_of_non_failed_configs, metric, fold_operation, maximize_metric):
        
        best_config_metric_nr = np.random.randint(0, len(list_of_non_failed_configs))
        best_config_outer_fold = list_of_non_failed_configs[best_config_metric_nr]

        return best_config_outer_fold
    
class RankingConfigSelector(BaseConfigSelector):
     
    def rank_metrics(self, metric_values, maximize_metric):

        # Sort the metric values in ascending (maximization) or descending (minimization) order.
        # The bigger the rank, the better the config

        if maximize_metric:
            # Replace None with a large or small value (e.g., positive infinity or negative infinity)
            metric_values = [value if value is not None else float('-inf') for value in metric_values]
            sorted_indices = np.argsort(metric_values)

        else:
            # Replace None with a large or small value (e.g., positive infinity or negative infinity)
            metric_values = [value if value is not None else float('inf') for value in metric_values]
            sorted_indices = np.argsort(metric_values)[::-1]

        ranks = np.empty(len(metric_values), dtype=int)
        ranks[sorted_indices] = np.arange(1, len(metric_values) + 1)

        # Rescale the ranks between 0 and 1
        ranks = (ranks - np.min(ranks)) / (np.max(ranks) - np.min(ranks))

        return ranks

    def __call__(self, list_of_non_failed_configs, metric, fold_operation, maximize_metric):
        all_metrics_mean, all_metrics_std, metric_list = self.prepare_metrics(list_of_non_failed_configs, metric)

        metric_ranks = np.array([])

        for i, current_metric in enumerate(metric_list):
            best_config_metric_values = [c.get_test_metric(current_metric, fold_operation) for c in
                                         list_of_non_failed_configs]

            # Rank the metric values
            current_metric_ranks = self.rank_metrics(best_config_metric_values,
                                                     Scorer.greater_is_better_distinction(current_metric))

            metric_ranks = np.append(metric_ranks, current_metric_ranks)

        total_ranks = np.sum(metric_ranks)

        # Select the configuration with the best rank sum (the bigger the rank the better)
        best_config_metric_nr = np.argmax(total_ranks)

        best_config_outer_fold = list_of_non_failed_configs[best_config_metric_nr]

        return best_config_outer_fold

    
class BestStdConfigSelector:

    def __call__(self, list_of_non_failed_configs, metric, fold_operation, maximize_metric):

        num_configs = len(list_of_non_failed_configs)
        data_array = np.zeros((num_configs, 2))
        Scorer.greater_is_better_distinction('variance_explained')
        # all_metrics_mean, all_metrics_std = self.prepare_metrics(list_of_non_failed_configs, metric)
        data_array[:, 0] = [c.get_test_metric(metric, fold_operation) for c in list_of_non_failed_configs]
        data_array[:, 1] = [c.get_test_metric(metric, "std") for c in list_of_non_failed_configs]

        best_config_metric_df = pd.DataFrame(data=data_array,
                                             columns=["values", "std"], index=np.arange(0, num_configs))
        best_config_metric_df = best_config_metric_df.sort_values(['values', 'std'],
                                                                  ascending=[not maximize_metric, True])
        best_config_metric_nr = best_config_metric_df.index[0]
        best_config_outer_fold = list_of_non_failed_configs[best_config_metric_nr]
        return best_config_outer_fold


class PercentileConfigSelector(BaseConfigSelector):

    def __init__(self, percentile: int = 90):
        self.percentile = percentile

    def __call__(self, list_of_non_failed_configs, metric, fold_operation, maximize_metric):

        all_metrics_mean, all_metrics_std = self.prepare_metrics(list_of_non_failed_configs, metric)
        best_config_metric_values = [c.get_test_metric(metric, fold_operation) for c in list_of_non_failed_configs]

        position = (self.percentile / 100.) if maximize_metric else 1 - (self.percentile / 100.)
        index = position * (len(best_config_metric_values) - 1)
        index = int(index + 0.5)  # index to integer conversion
        # following line is a hack for speedup
        best_config_outer_fold = list_of_non_failed_configs[np.argpartition(best_config_metric_values, index)[index]]

        return best_config_outer_fold


class WeightedRankingConfigSelector(RankingConfigSelector):

    def __call__(self, list_of_non_failed_configs, metric, fold_operation, maximize_metric):
        
        all_metrics_mean, all_metrics_std, metric_list = self.prepare_metrics(list_of_non_failed_configs, metric)
        
        metric_ranks = np.array([]) 

        for i, current_metric in enumerate(metric_list) :
        
            best_config_metric_values = [c.get_test_metric(current_metric, fold_operation) for c in list_of_non_failed_configs]
            current_metric_ranks = self.rank_metrics(best_config_metric_values, Scorer.greater_is_better_distinction(current_metric))

            if current_metric == metric : 
                current_metric_ranks = current_metric_ranks * 0.5
            else :
                current_metric_ranks = current_metric_ranks * (0.5 / (len(metric_list) - 1))

            # Rank the metric values
            metric_ranks = np.append(metric_ranks, current_metric_ranks)

        total_ranks = np.sum(metric_ranks)
        
        # Select the configuration with the best rank sum (the bigger the rank the better)
        best_config_metric_nr = np.argmax(total_ranks)
        
        best_config_outer_fold = list_of_non_failed_configs[best_config_metric_nr]

        return best_config_outer_fold

