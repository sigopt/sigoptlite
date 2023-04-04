# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import pytest
from sigopt import Connection
from sigopt.exception import SigOptException

from sigoptlite.driver import LocalDriver

from test.base_test import UnitTestsBase


class TestMultitask(UnitTestsBase):
  @pytest.fixture
  def conn(self):
    return Connection(driver=LocalDriver)

  @pytest.fixture
  def base_meta(self):
    return dict(
      parameters=[
        dict(name="x0", type="double", bounds=dict(min=0, max=1)),
        dict(name="x1", type="double", bounds=dict(min=0, max=1)),
        dict(name="x2", type="int", bounds=dict(min=0, max=100)),
        dict(name="x3", type="int", bounds=dict(min=0, max=100)),
        dict(name="x4", type="categorical", categorical_values=["c1", "c2"]),
      ],
      metrics=[dict(name="metric")],
      observation_budget=10,
    )

  def test_improper_tasks_negative_cost(self, conn, base_meta):
    tasks = [dict(name="cheap", cost=-0.1), dict(name="expensive", cost=1)]
    experiment_meta = base_meta
    experiment_meta["tasks"] = tasks
    with pytest.raises(SigOptException) as exception_info:
      conn.experiments().create(**experiment_meta)
    msg = "Validation failed for LocalExperimentBuilder: .cost must be greater than 0"
    assert exception_info.value.args[0] == msg

  def test_improper_tasks_cost_greater_than_one(self, conn, base_meta):
    tasks = [dict(name="cheap", cost=1), dict(name="expensive", cost=2)]
    experiment_meta = base_meta
    experiment_meta["tasks"] = tasks
    with pytest.raises(SigOptException) as exception_info:
      conn.experiments().create(**experiment_meta)
    msg = "Validation failed for LocalExperimentBuilder: .cost must be less than or equal to 1"
    assert exception_info.value.args[0] == msg

  def test_improper_tasks_cost_none_equal_one(self, conn, base_meta):
    tasks = [dict(name="cheap", cost=0.1), dict(name="expensive", cost=0.9)]
    experiment_meta = base_meta
    experiment_meta["tasks"] = tasks
    with pytest.raises(SigOptException) as exception_info:
      conn.experiments().create(**experiment_meta)
    msg = "For multitask sigoptlite experiment, exactly one task must have cost == 1 (none present)."
    assert exception_info.value.args[0] == msg

  def test_improper_tasks_not_object(self, conn, base_meta):
    tasks = [0.1, 1]
    experiment_meta = base_meta
    experiment_meta["tasks"] = tasks
    with pytest.raises(SigOptException) as exception_info:
      conn.experiments().create(**experiment_meta)
    msg = "Validation failed for LocalExperimentBuilder: Invalid type for .tasks[0]: 0.1 - expected type object"
    assert exception_info.value.args[0] == msg

  def test_improper_tasks_no_cost(self, conn, base_meta):
    tasks = [dict(name="cheap"), dict(name="expensive")]
    experiment_meta = base_meta
    experiment_meta["tasks"] = tasks
    with pytest.raises(SigOptException) as exception_info:
      conn.experiments().create(**experiment_meta)
    msg = """Validation failed for LocalExperimentBuilder: Missing required json key "cost" in: {"name": "cheap"}"""
    assert exception_info.value.args[0] == msg

  def test_improper_tasks_same_names(self, conn, base_meta):
    tasks = [dict(name="cheap", cost=0.1), dict(name="cheap", cost=1)]
    experiment_meta = base_meta
    experiment_meta["tasks"] = tasks
    with pytest.raises(SigOptException) as exception_info:
      conn.experiments().create(**experiment_meta)
    msg = "For multitask sigoptlite experiment, all task names must be distinct"
    assert exception_info.value.args[0] == msg

  def test_improper_tasks_same_cost(self, conn, base_meta):
    tasks = [dict(name="cheap", cost=0.1), dict(name="expensive", cost=0.1)]
    experiment_meta = base_meta
    experiment_meta["tasks"] = tasks
    with pytest.raises(SigOptException) as exception_info:
      conn.experiments().create(**experiment_meta)
    msg = "For multitask sigoptlite experiment, all task costs must be distinct"
    assert exception_info.value.args[0] == msg

  def test_improper_task_costs_negative(self, conn, base_meta):
    experiment_meta = base_meta
    experiment_meta["tasks"] = [dict(name="cheap", cost=-0.1), dict(name="expensive", cost=1)]
    with pytest.raises(SigOptException) as exception_info:
      conn.experiments().create(**experiment_meta)
    msg = "Validation failed for LocalExperimentBuilder: .cost must be greater than 0"
    assert exception_info.value.args[0] == msg

  def test_single_task_forbidden(self, conn, base_meta):
    experiment_meta = base_meta
    experiment_meta["tasks"] = [dict(name="cheap", cost=1)]
    with pytest.raises(SigOptException) as exception_info:
      conn.experiments().create(**experiment_meta)
    msg = "For multitask sigoptlite experiment, at least 2 tasks must be present"
    assert exception_info.value.args[0] == msg

  def test_multitask_no_observation_budget_forbidden(self, conn):
    experiment_meta = self.get_experiment_feature("multitask")
    experiment_meta.pop("observation_budget")
    with pytest.raises(SigOptException) as exception_info:
      conn.experiments().create(**experiment_meta)
    msg = "observation_budget is required for a sigoptlite experiment with tasks (multitask)"
    assert exception_info.value.args[0] == msg
