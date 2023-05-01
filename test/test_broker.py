# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import mock
import pytest

from sigoptlite.broker import Broker
from sigoptlite.builders import LocalExperimentBuilder
from sigoptlite.sources import GPSource

from test.base_test import UnitTestsBase


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

  @pytest.mark.parametrize("feature", ["categorical", "integer"])
  @mock.patch.object(GPSource, "next_point")
  def test_fallback_random_successful_when_suggestions_exhausted(self, mock_next_point, feature):
    mock_next_point.return_value = [], None
    num_observations = 3
    experiment_meta = self.get_experiment_feature(feature)
    experiment = LocalExperimentBuilder(experiment_meta)
    with mock.patch("sigoptlite.broker.Broker.use_random", new_callable=mock.PropertyMock) as mock_use_random:
      mock_use_random.return_value = False
      broker = Broker(experiment)
      for local_observation in self.make_random_observations(experiment, num_observations):
        observation_dict = local_observation.get_client_observation(experiment)
        broker.create_observation(**observation_dict)
      suggestion = broker.create_suggestion()

    mock_use_random.assert_called_once()
    mock_next_point.assert_called_once()
    self.assert_valid_suggestion(suggestion, experiment)
