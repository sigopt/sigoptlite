# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import mock
import numpy
import pytest

from sigoptlite.broker import Broker
from sigoptlite.builders import LocalExperimentBuilder
from sigoptlite.sources import GPSource

from test.base_test import UnitTestsBase
from test.constants import CATEGORICAL_EXPERIMENT_PARAMETER_NAME, DEFAULT_METRICS, INT_EXPERIMENT_PARAMETER_NAME


class TestBroker(UnitTestsBase):
  @pytest.mark.parametrize("num_observations", [1])
  def test_basic(self, experiment_meta, num_observations):
    experiment = LocalExperimentBuilder(experiment_meta)
    broker = Broker(experiment)
    for _ in range(num_observations):
      suggestion = broker.create_suggestion()
      self.assert_valid_suggestion(suggestion, experiment)

      local_observation = self.make_random_observation(experiment, suggestion=suggestion)
      observation_dict = local_observation.get_client_observation(experiment)
      broker.create_observation(**observation_dict)

  @mock.patch.object(GPSource, "next_point")
  def test_fallback_random_successful_when_categories_exhausted(self, mock_next_point):
    parameter_name = "c"
    categorical_values = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    parameter_high_category_count = dict(
      name=parameter_name,
      type=CATEGORICAL_EXPERIMENT_PARAMETER_NAME,
      categorical_values=categorical_values,
    )
    experiment_meta = dict(
      parameters=[parameter_high_category_count],
      metrics=DEFAULT_METRICS,
    )
    experiment = LocalExperimentBuilder(experiment_meta)
    broker = Broker(experiment)
    for category in parameter_high_category_count["categorical_values"]:
      observation_dict = dict(
        assignments={parameter_name: category},
        values=[dict(name=experiment.metrics[0].name, value=numpy.random.rand(), value_stddev=0)],
      )
      broker.create_observation(**observation_dict)

    mock_next_point.return_value = [[]], None
    broker.create_suggestion()

  @mock.patch.object(GPSource, "next_point")
  def test_fallback_random_successful_when_integers_exhausted(self, mock_next_point):
    parameter_name = "i"
    start = 1
    stop = 100
    parameter_high_int_count = dict(
      name=parameter_name,
      type=INT_EXPERIMENT_PARAMETER_NAME,
      bounds=dict(min=start, max=stop),
    )
    experiment_meta = dict(
      parameters=[parameter_high_int_count],
      metrics=DEFAULT_METRICS,
    )
    experiment = LocalExperimentBuilder(experiment_meta)
    broker = Broker(experiment)
    for i in numpy.arange(start, stop + 1):
      observation_dict = dict(
        assignments={parameter_name: i},
        values=[dict(name=experiment.metrics[0].name, value=numpy.random.rand(), value_stddev=0)],
      )
      broker.create_observation(**observation_dict)

    mock_next_point.return_value = [[]], None
    broker.create_suggestion()
