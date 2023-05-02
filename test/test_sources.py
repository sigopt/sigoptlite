# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import mock
import pytest

from libsigopt.views.rest.search_next_points import SearchNextPoints
from libsigopt.views.rest.spe_search_next_points import SPESearchNextPoints

from sigoptlite.builders import LocalExperimentBuilder
from sigoptlite.sources import EmptySuggestionError, GPSource, RandomSearchSource, SPESource

from test.base_test import UnitTestsBase


class TestRandomSearch(UnitTestsBase):
  @pytest.mark.parametrize("num_observations", [0, 7])
  def test_basic(self, any_meta, num_observations):
    experiment = LocalExperimentBuilder(any_meta)
    observations = self.make_random_observations(experiment, num_observations)
    source = RandomSearchSource(experiment)
    suggestion = source.get_suggestion(observations)
    self.assert_valid_suggestion(suggestion, experiment)


class TestGPNextPoints(UnitTestsBase):
  @staticmethod
  def assert_valid_hyperparameters(hyperparameters, experiment):
    assert hyperparameters["alpha"] > 0

    for lengthscales, parameter in zip(hyperparameters["length_scales"], experiment.parameters):
      if parameter.is_categorical:
        assert len(lengthscales) == len(parameter.categorical_values)
      else:
        assert len(lengthscales) == 1
      assert bool(ll > 0 for ll in lengthscales)

    if experiment.is_multitask:
      assert hyperparameters["task_length"] is not None
      assert hyperparameters["task_length"] > 0

  @pytest.mark.parametrize("feature", ["default", "multitask", "metric_constraint"])
  def test_hyperparameter_update(self, feature):
    experiment_meta = self.get_experiment_feature(feature)
    experiment = LocalExperimentBuilder(experiment_meta)
    observations = self.make_random_observations(experiment, 5)
    source = GPSource(experiment)
    defaut_hyperparameters = GPSource.get_default_hyperparameters(experiment)
    hyperparameters_list = source.update_hyperparameters(observations, defaut_hyperparameters)
    assert hyperparameters_list != defaut_hyperparameters
    assert len(hyperparameters_list) > 0
    for hyperparameters in hyperparameters_list:
      self.assert_valid_hyperparameters(hyperparameters, experiment)

  def test_valid_suggestion(self, any_meta):
    experiment = LocalExperimentBuilder(any_meta)
    observations = self.make_random_observations(experiment, 5)
    source = GPSource(experiment)
    suggestion = source.get_suggestion(observations)
    self.assert_valid_suggestion(suggestion, experiment)

  @mock.patch.object(SearchNextPoints, "view")
  def test_multisolution_calls_search(self, mock_view):
    fake_point = [1.265, 2.151, 3.1205]
    mock_view.return_value = {"points_to_sample": [fake_point]}

    experiment_meta = self.get_experiment_feature("multisolution")
    experiment = LocalExperimentBuilder(experiment_meta)
    observations = self.make_random_observations(experiment, 5)
    source = GPSource(experiment)
    point, task_cost = source.next_point(observations)
    assert point == [fake_point]
    assert task_cost is None

  @pytest.mark.parametrize("feature", ["categorical", "integer"])
  def test_space_exhausted_empty_next_points(self, feature):
    experiment_meta = self.get_experiment_feature(feature)
    experiment = LocalExperimentBuilder(experiment_meta)

    parameter = experiment_meta["parameters"][0]
    elements = []
    if feature == "categorical":
      elements = parameter["categorical_values"]
    elif feature == "integer":
      bound_min, bound_max = parameter["bounds"]["min"], parameter["bounds"]["max"]
      elements = [*range(bound_min, bound_max + 1)]

    observations = [
      self.make_observation(
        experiment=experiment,
        assignments={parameter["name"]: element},
        values=[dict(name=experiment.metrics[0].name, value=i, value_stddev=0)],
      )
      for i, element in enumerate(elements)
    ]

    next_points, _ = GPSource(experiment).next_point(observations)
    assert next_points == []
    with pytest.raises(EmptySuggestionError) as exception_info:
      GPSource(experiment).get_suggestion(observations)
    msg = "Unable to generate suggestions. Maybe all unique suggestions were sampled?"
    assert exception_info.value.args[0].startswith(msg)


class TestSPENextPoints(UnitTestsBase):
  def test_valid_suggestion(self, any_meta):
    experiment = LocalExperimentBuilder(any_meta)
    observations = self.make_random_observations(experiment, 5)
    source = SPESource(experiment)
    suggestion = source.get_suggestion(observations)
    self.assert_valid_suggestion(suggestion, experiment)

  @mock.patch.object(SPESearchNextPoints, "view")
  def test_multisolution_calls_search(self, mock_view):
    fake_point = [9.657, 8.321, 7.6518]
    mock_view.return_value = {"points_to_sample": [fake_point]}

    experiment_meta = self.get_experiment_feature("multisolution")
    experiment = LocalExperimentBuilder(experiment_meta)
    observations = self.make_random_observations(experiment, 5)
    source = SPESource(experiment)
    point, task_cost = source.next_point(observations)
    assert point == [fake_point]
    assert task_cost is None


class TestDiscreteOptions(UnitTestsBase):
  @pytest.mark.parametrize("source_class", [RandomSearchSource, SPESource])
  @pytest.mark.parametrize("feature,num_observations", [("categorical", 5), ("integer", 15)])
  def test_exhausted_options(self, source_class, feature, num_observations):
    experiment_meta = self.get_experiment_feature(feature)
    experiment = LocalExperimentBuilder(experiment_meta)
    source = source_class(experiment)
    observations = self.make_random_observations(experiment, num_observations)
    suggestion = source.get_suggestion(observations)
    self.assert_valid_suggestion(suggestion, experiment)
