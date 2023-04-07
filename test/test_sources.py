# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import mock
import numpy
import pytest

from libsigopt.aux.constant import CATEGORICAL_EXPERIMENT_PARAMETER_NAME
from libsigopt.aux.errors import SigoptComputeError
from libsigopt.views.rest.search_next_points import SearchNextPoints
from libsigopt.views.rest.spe_search_next_points import SPESearchNextPoints

from sigoptlite.builders import LocalExperimentBuilder, LocalObservationBuilder
from sigoptlite.sources import GPSource, RandomSearchSource, SPESource

from test.base_test import UnitTestsBase
from test.constants import DEFAULT_METRICS, PARAMETER_CATEGORICAL


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
      if parameter.type == CATEGORICAL_EXPERIMENT_PARAMETER_NAME:
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

  def test_space_exhausted_empty_next_points(self):
    experiment_meta = dict(
      parameters=[PARAMETER_CATEGORICAL],
      metrics=DEFAULT_METRICS,
    )
    experiment = LocalExperimentBuilder(experiment_meta)
    observations = []
    for category in PARAMETER_CATEGORICAL["categorical_values"]:
      observation_dict = dict(
        assignments={PARAMETER_CATEGORICAL["name"]: category},
        values=[dict(name=experiment.metrics[0].name, value=numpy.random.rand(), value_stddev=0)],
      )
      observation = LocalObservationBuilder(observation_dict, experiment=experiment)
      observations.append(observation)

    next_points, _ = GPSource(experiment).next_point(observations)
    assert next_points == []
    with pytest.raises(SigoptComputeError):
      GPSource(experiment).get_suggestion(observations)


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
