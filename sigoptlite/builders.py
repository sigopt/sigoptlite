# Copyright © 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache License 2.0
import itertools

import numpy


from libsigopt.aux.constant import (
  DOUBLE_EXPERIMENT_PARAMETER_NAME,
  INT_EXPERIMENT_PARAMETER_NAME,
  ConstraintType,
  ParameterTransformationNames,
)

from libsigopt.aux.validate_schema import validate as validate_against_schema
from libsigopt.aux.errors import SigoptValidationError

from sigoptlite.models import *
from sigoptlite.validators import EXPERIMENT_CREATE_SCHEMA, OBSERVATION_CREATE_SCHEMA, check_all_conditional_values_satisfied, check_constraint_feasibility, validate_constraints, validate_tasks


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
        cls.validate_parameters(experiment)
        cls.validate_metrics(experiment)

        if not experiment.parallel_bandwidth == 1:
            raise ValueError(f"{cls.cls_name} must have parallel_bandwidth == 1")

        observation_budget = experiment.observation_budget
        if observation_budget is None:
            if experiment.num_solutions > 1:
                raise ValueError(f"observation_budget is required for a {cls.cls_name} with multiple solutions")
            if experiment.requires_pareto_frontier_optimization:
                raise ValueError(f"observation_budget is required for a {cls.cls_name} with more than one optimized metric")
            if experiment.has_constraint_metrics:
                raise ValueError(f"observation_budget is required for a {cls.cls_name} with constraint metrics")
            if experiment.is_multitask:
                raise ValueError(f"observation_budget is required for a {cls.cls_name} with tasks (multitask)")

        if not (experiment.optimized_metrics or experiment.constraint_metrics):
            raise ValueError(f"{cls.cls_name} must have optimized or constraint metrics")

        if experiment.optimized_metrics:
            if not len(experiment.optimized_metrics) in [1, 2]:
                raise ValueError(f"{cls.cls_name} must have one or two optimized metrics")
            elif len(experiment.optimized_metrics) == 1 and experiment.optimized_metrics[0].threshold is not None:
                raise ValueError(
                  "Thresholds are only supported for experiments with more than one optimized metric."
                  " Try an All-Constraint experiment instead by setting `strategy` to `constraint`."
                )

        # Check feature viability of multisolution experiments
        num_solutions = experiment.num_solutions
        if num_solutions and num_solutions > 1:
            if num_solutions > observation_budget:
                raise ValueError("observation_budget needs to be larger than the number of solutions")
            if not len(experiment.optimized_metrics) == 1:
                raise ValueError(f"{cls.cls_name} with multiple solutions require exactly one optimized metric")

        # Check conditional limitation
        parameters_have_conditions = any(parameter.conditions for parameter in experiment.parameters)
        if parameters_have_conditions ^ experiment.is_conditional:
            raise ValueError(
              f"For conditional {cls.cls_name}, need both conditions defined in parameters and conditionals variables"
              " defined in experiment"
            )
        if experiment.is_conditional:
            if num_solutions and num_solutions > 1:
                raise ValueError(f"{cls.cls_name} with multiple solutions does not support conditional parameters")
            if experiment.is_search:
                raise ValueError(f"All-Constraint {cls.cls_name} does not support conditional parameters")
            cls.validate_conditionals(experiment)

        # Check feature viability of multitask
        tasks = experiment.tasks
        if tasks:
            if experiment.requires_pareto_frontier_optimization:
                raise ValueError(f"{cls.cls_name} cannot have both tasks and multiple optimized metrics")
            if experiment.has_constraint_metrics:
                raise ValueError(f"{cls.cls_name} cannot have both tasks and constraint metrics")
            if num_solutions and num_solutions > 1:
                raise ValueError(f"{cls.cls_name} with multiple solutions cannot be multitask")
            cls.validate_tasks(experiment, cls.cls_name)

        if experiment.linear_constraints:
            validate_constraints(experiment)
            check_constraint_feasibility(experiment)

    @classmethod
    def validate_parameters(cls, experiment):
        param_names = [p.name for p in experiment.parameters]
        if not len(param_names) == cls.get_num_distinct_elements(param_names):
            raise ValueError(f"No duplicate parameters are allowed: {param_names}")

    @classmethod
    def validate_metrics(cls, experiment):
        metric_names = [m.name for m in experiment.metrics]
        if not len(metric_names) == cls.get_num_distinct_elements(metric_names):
            raise ValueError(f"No duplicate metrics are allowed: {metric_names}")

    @classmethod
    def validate_conditionals(cls, experiment):
        conditional_names = [c.name for c in experiment.conditionals]
        if not len(conditional_names) == cls.get_num_distinct_elements(conditional_names):
            raise ValueError(f"No duplicate conditionals are allowed: {conditional_names}")

        for parameter in experiment.parameters:
            if parameter.conditions and any(c.name not in conditional_names for c in parameter.conditions):
                unsatisfied_condition_names = [c.name for c in parameter.conditions if c.name not in conditional_names]
                raise ValueError(
                  f"The parameter {parameter.name} has conditions {unsatisfied_condition_names} that are not part of"
                  " the conditionals"
                )

        check_all_conditional_values_satisfied(experiment)


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
        # categorical parameter
        if parameter.is_categorical:
            if not len(parameter.categorical_values) > 1:
                raise ValueError(
                  f"Categorical parameter {parameter.name} must have more than one categorical value. "
                  f"Current values are {parameter.categorical_values}"
                )
            if parameter.grid:
                raise ValueError("Categorical parameter does not support grid values")
            if parameter.bounds:
                raise ValueError(f"Categorical parameter should not have bounds: {parameter.bounds}")

        # parameter with grid
        if parameter.grid:
            if not len(parameter.grid) > 1:
                raise ValueError(
                  f"Grid parameter {parameter.name} must have more than one value. Current values are {parameter.grid}"
                )
            if parameter.bounds:
                raise ValueError(f"Grid parameter should not have bounds: {parameter.bounds}")
            if not cls.get_num_distinct_elements(parameter.grid) == len(parameter.grid):
                raise ValueError(f"Grid values should be unique: {parameter.grid}")

        # log transformation
        if parameter.has_transformation:
            if not parameter.is_double:
                raise ValueError("Transformation only applies to parameters type of double")
            if parameter.bounds and parameter.bounds.min <= 0:
                raise ValueError("Invalid bounds for log-transformation: bounds must be positive")
            if parameter.grid and min(parameter.grid) <= 0:
                raise ValueError("Invalid grid values for log-transformation: values must be positive")

        # parameter priors
        if parameter.has_prior:
            if not parameter.is_double:
                raise ValueError("Prior only applies to parameters type of double")
            if parameter.grid:
                raise ValueError("Grid parameters cannot have priors")
            if parameter.has_transformation:
                raise ValueError("Parameters with log transformation cannot have priors")
            if parameter.prior.is_normal:
                if not parameter.bounds.is_value_within(parameter.prior.mean):
                    raise ValueError(f"parameter.prior.mean {parameter.prior.mean} must be within bounds {parameter.bounds}")
            if not (parameter.prior.is_normal ^ parameter.prior.is_beta):
                raise ValueError(f"{parameter.prior} must be either normal or beta")


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
        for parameter in experiment.parameters:
            if parameter_conditions_satisfied(parameter, observation.assignments):
                cls.observation_must_have_parameter(observation, parameter)
            else:
                cls.observation_does_not_have_parameter(observation, parameter)

        cls.validate_observation_conditionals(observation, experiment.conditionals)

        if experiment.is_multitask:
            cls.validate_observation_tasks(observation, experiment.tasks)

        if not experiment.is_multitask and observation.task:
            raise ValueError("Observation with task is not expected for this experiment")

        if observation.failed and observation.metric_evaluations:
            raise ValueError(
              f"Observation marked as failure ({observation.failed}) should not have values. "
              f"Observation metrics are: {observation.metric_evaluations}."
            )

        if not observation.failed:
            num_reported_metrics = len(observation.metric_evaluations)
            if num_reported_metrics != len(experiment.metrics):
                raise ValueError("The number of observation values and experiment metrics must be equal.")
            for m in experiment.metrics:
                if observation.get_metric_evaluation_by_name(m.name) is None:
                    raise ValueError(
                      f"Values must have metric names defined in experiment: {[m.name for m in experiment.metrics]}."
                    )

    @staticmethod
    def observation_must_have_parameter(observation, parameter):
        if parameter.name not in observation.assignments:
            raise ValueError(
              f"Parameter {parameter.name} is required for this experiment, "
              f"and is missing from this observation: {observation.assignments}"
            )
        parameter_value = observation.assignments[parameter.name]
        if parameter.is_categorical:
            expected_categories = [cv.name for cv in parameter.categorical_values]
            if parameter_value not in expected_categories:
                raise ValueError(
                  f"Categorical parameter {parameter.name} must have one of following categories: "
                  f"{expected_categories} instead of {parameter_value}"
                )
        if parameter.grid and parameter_value not in parameter.grid:
            raise ValueError(
              f"Grid parameter {parameter.name} must have one of following grid values: "
              f"{parameter.grid} instead of {parameter_value}"
            )
        if parameter.has_transformation:
            if not (parameter_value > 0):
                raise ValueError(f"Assignment must be positive for log-transformed parameter {parameter.name}")

    @staticmethod
    def observation_does_not_have_parameter(observation, parameter):
        if parameter.name in observation.assignments:
            raise ValueError(
              f"Parameter {parameter.name} does not satisfy conditions. "
              f"Observation assignments: {observation.assignments} is invalid."
            )

    @staticmethod
    def validate_observation_conditionals(observation, conditionals):
        for conditional in conditionals:
            if conditional.name not in observation.assignments:
                raise ValueError(f"Conditional parameter {conditional.name} must be in {observation}")
            conditional_value = observation.assignments[conditional.name]
            expected_conditional_options = [cv.name for cv in conditional.values]
            if conditional_value not in expected_conditional_options:
                raise ValueError(
                  f"Conditional parameter {conditional.name} must have one of following options: "
                  f"{expected_conditional_options} instead of {conditional_value}"
                )

    @staticmethod
    def validate_observation_tasks(observation, tasks):
        if not observation.task:
            raise ValueError("Observation must have a task field for this experiment")
        obs_task_name = observation.task.name
        if obs_task_name not in [t.name for t in tasks]:
            raise ValueError(
              f"Task {obs_task_name} is not a valid task for this experiment. Must be one of the following: {tasks}"
            )
        obs_task_costs = observation.task.cost
        expected_task_costs = [t.cost for t in tasks]
        if obs_task_costs not in expected_task_costs:
            raise ValueError(
              f"Task cost {obs_task_costs} is not a valid cost for this experiment. Must be one of the following:"
              f" {expected_task_costs}"
            )


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
