# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0

from libsigopt.aux.errors import SigoptValidationError
from libsigopt.aux.validate_schema import validate as validate_against_schema

from sigoptlite.models import *
from sigoptlite.validators import (
  EXPERIMENT_CREATE_SCHEMA,
  OBSERVATION_CREATE_SCHEMA,
  validate_experiment,
  validate_observation,
  validate_parameter,
)


def create_experiment_from_template(experiment_template, **kwargs):
  experiment_meta = dataclass_to_dict(experiment_template)
  experiment_meta.update(kwargs)
  return LocalExperimentBuilder(experiment_meta)


class BuilderBase(object):
  def __new__(cls, input_dict, **kwargs):
    try:
      cls.validate_input_dict(input_dict)
    except AssertionError as e:
      raise ValueError(f"Invalid input for {cls.__name__} {e}") from e
    except SigoptValidationError as e:
      raise ValueError(f"Validation failed for {cls.__name__}: {e}") from e

    local_object = cls.create_object(**input_dict)
    cls.validate_object(local_object, **kwargs)
    return local_object

  @classmethod
  def validate_input_dict(cls, input_dict):
    raise NotImplementedError

  @classmethod
  def create_object(cls, **input_dict):
    raise NotImplementedError

  @classmethod
  def validate_object(cls, _):
    pass

  @classmethod
  def set_object(cls, input_dict, field, local_class):
    if not input_dict.get(field):
      return
    input_dict[field] = local_class(input_dict[field])

  @classmethod
  def set_list_of_objects(cls, input_dict, field, local_class):
    if not input_dict.get(field):
      return
    input_dict[field] = [local_class(i) for i in input_dict[field]]

  @staticmethod
  def get_num_distinct_elements(lst):
    return len(set(lst))


class LocalExperimentBuilder(BuilderBase):
  cls_name = "sigoptlite experiment"

  @classmethod
  def validate_input_dict(cls, input_dict):
    validate_against_schema(input_dict, EXPERIMENT_CREATE_SCHEMA)

  @classmethod
  def create_object(cls, **input_dict):
    cls.set_list_of_objects(input_dict, field="parameters", local_class=LocalParameterBuilder)
    cls.set_list_of_objects(input_dict, field="metrics", local_class=LocalMetricBuilder)
    cls.set_list_of_objects(input_dict, field="conditionals", local_class=LocalConditionalBuilder)
    cls.set_list_of_objects(input_dict, field="tasks", local_class=LocalTaskBuilder)
    cls.set_list_of_objects(input_dict, field="linear_constraints", local_class=LocalLinearConstraintBuilder)
    return LocalExperiment(**input_dict)

  @classmethod
  def validate_object(cls, experiment):
    validate_experiment(experiment, cls.cls_name)


class LocalParameterBuilder(BuilderBase):
  @classmethod
  def validate_input_dict(cls, input_dict):
    assert isinstance(input_dict["name"], str)
    assert isinstance(input_dict["type"], str)

  @classmethod
  def create_object(cls, **input_dict):
    cls.set_object(input_dict, field="bounds", local_class=LocalBoundsBuilder)
    if input_dict.get("categorical_values"):
      categorical_values = input_dict["categorical_values"]
      get_name_options = {str: lambda x: x, dict: lambda x: x["name"]}
      get_name = get_name_options[type(categorical_values[0])]
      sorted_categorical_values = sorted(categorical_values, key=get_name)
      input_dict["categorical_values"] = [
        LocalCategoricalValue(name=get_name(cv), enum_index=i + 1) for i, cv in enumerate(sorted_categorical_values)
      ]
    if input_dict.get("conditions"):
      input_dict["conditions"] = [LocalCondition(name=n, values=v) for n, v in input_dict["conditions"].items()]
    cls.set_object(input_dict, field="prior", local_class=LocalParameterPriorBuilder)
    return LocalParameter(**input_dict)

  @classmethod
  def validate_object(cls, parameter):
    validate_parameter(parameter)


class LocalMetricBuilder(BuilderBase):
  @classmethod
  def validate_input_dict(cls, input_dict):
    assert isinstance(input_dict["name"], str)

  @classmethod
  def create_object(cls, **input_dict):
    return LocalMetric(**input_dict)

  @classmethod
  def validate_object(cls, metric):
    if metric.is_constraint and metric.threshold is None:
      raise ValueError("Constraint metrics must have the threshold field defined")


class LocalConditionalBuilder(BuilderBase):
  @classmethod
  def validate_input_dict(cls, input_dict):
    assert set(input_dict.keys()) == {"name", "values"}
    assert isinstance(input_dict["name"], str)
    assert isinstance(input_dict["values"], list)
    if not len(input_dict["values"]) > 1:
      raise ValueError(f"Conditional {input_dict['name']} must have at least two values")

  @classmethod
  def create_object(cls, **input_dict):
    input_dict["values"] = [LocalConditionalValue(enum_index=i + 1, name=v) for i, v in enumerate(input_dict["values"])]
    return LocalConditional(**input_dict)


class LocalTaskBuilder(BuilderBase):
  @classmethod
  def validate_input_dict(cls, input_dict):
    assert set(input_dict.keys()) == {"name", "cost"}
    assert isinstance(input_dict["name"], str)
    assert isinstance(input_dict["cost"], (int, float))

  @classmethod
  def create_object(cls, **input_dict):
    return LocalTask(**input_dict)

  @classmethod
  def validate_object(cls, task):
    if not (0 < task.cost <= 1):
      raise ValueError(f"{task} costs must be positve and less than or equal to 1.")


class LocalLinearConstraintBuilder(BuilderBase):
  @classmethod
  def validate_input_dict(cls, input_dict):
    assert input_dict.keys() == {"type", "terms", "threshold"}
    assert input_dict["type"] in ["less_than", "greater_than"]
    assert isinstance(input_dict["threshold"], (int, float))

  @classmethod
  def create_object(cls, **input_dict):
    cls.set_list_of_objects(input_dict, field="terms", local_class=LocalConstraintTermBuilder)
    return LocalLinearConstraint(**input_dict)


class LocalBoundsBuilder(BuilderBase):
  @classmethod
  def validate_input_dict(cls, input_dict):
    assert set(input_dict.keys()) == {"min", "max"}
    assert isinstance(input_dict["min"], (int, float))
    assert isinstance(input_dict["max"], (int, float))

  @classmethod
  def create_object(cls, **input_dict):
    return LocalBounds(**input_dict)

  @classmethod
  def validate_object(cls, bounds):
    if bounds.min >= bounds.max:
      raise ValueError(f"{bounds}: min must be less than max")


class LocalParameterPriorBuilder(BuilderBase):
  @classmethod
  def validate_input_dict(cls, input_dict):
    assert input_dict["name"] in ["beta", "normal"]

  @classmethod
  def create_object(cls, **input_dict):
    return LocalParameterPrior(**input_dict)

  @classmethod
  def validate_object(cls, parameter_prior):
    if parameter_prior.is_beta:
      if (parameter_prior.shape_a is None) or (parameter_prior.shape_b is None):
        raise ValueError(f"{parameter_prior} must have shape_a and shape_b")
      if parameter_prior.shape_a <= 0:
        raise ValueError(f"{parameter_prior} shape_a must be positive")
      if parameter_prior.shape_b <= 0:
        raise ValueError(f"{parameter_prior} shape_b must be positive")
    if parameter_prior.is_normal:
      if (parameter_prior.mean is None) or (parameter_prior.scale is None):
        raise ValueError(f"{parameter_prior} must provide mean and scale")
      if parameter_prior.scale <= 0:
        raise ValueError(f"{parameter_prior} scale must be positive")


class LocalConstraintTermBuilder(BuilderBase):
  @classmethod
  def validate_input_dict(cls, input_dict):
    assert set(input_dict.keys()) == {"name", "weight"}
    assert isinstance(input_dict["name"], str)
    assert isinstance(input_dict["weight"], (int, float))

  @classmethod
  def create_object(cls, **input_dict):
    return LocalConstraintTerm(**input_dict)


class LocalObservationBuilder(BuilderBase):
  @classmethod
  def validate_input_dict(cls, input_dict):
    validate_against_schema(input_dict, OBSERVATION_CREATE_SCHEMA)

  @classmethod
  def create_object(cls, **input_dict):
    cls.set_object(input_dict, "assignments", LocalAssignments)
    cls.set_list_of_objects(input_dict, field="values", local_class=MetricEvaluationBuilder)
    values = input_dict.get("values")
    if values:
      input_dict["metric_evaluations"] = {me.name: me for me in values}
    input_dict.pop("values", None)
    cls.set_object(input_dict, "task", LocalTaskBuilder)
    return LocalObservation(**input_dict)

  @classmethod
  def validate_object(cls, observation, experiment):
    validate_observation(observation, experiment)


class MetricEvaluationBuilder(BuilderBase):
  @classmethod
  def validate_input_dict(cls, input_dict):
    assert isinstance(input_dict["name"], str)
    assert isinstance(input_dict["value"], (int, float))
    if "value_stddev" in input_dict:
      assert isinstance(input_dict["value_stddev"], (int, float))
      assert input_dict["value_stddev"] >= 0
      assert set(input_dict.keys()) == {"name", "value", "value_stddev"}
    else:
      assert set(input_dict.keys()) == {"name", "value"}

  @classmethod
  def create_object(cls, **input_dict):
    return MetricEvaluation(**input_dict)
