# Copyright © 2023 Intel Corporation
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
from libsigopt.aux.geometry_utils import find_interior_point

from sigoptlite.models import parameter_conditions_satisfied


EXPERIMENT_CREATE_SCHEMA = {
  "definitions": {
    "name": {
      "type": "string",
      "minLength": 1,
      "maxLength": 100,
    },
    "opt_name": {
      "type": ["string", "null"],
      "minLength": 1,
      "maxLength": 100,
    },
    "term": {
      "type": "object",
      "required": ["name", "weight"],
      "properties": {
        "name": {"$ref": "#/definitions/opt_name"},
        "weight": {"type": "number"},
      },
    },
    "task": {
      "type": "object",
      "required": ["name", "cost"],
      "properties": {
        "name": {"$ref": "#/definitions/opt_name"},
        "cost": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
      },
    },
    "constraints": {
      "linear_ineq": {
        "type": "object",
        "required": ["type", "terms", "threshold"],
        "properties": {
          "type": {
            "type": "string",
            "enum": ["greater_than", "less_than"],
          },
          "terms": {"type": "array", "items": {"$ref": "#/definitions/term"}},
          "threshold": {"type": "number"},
        },
      }
    },
  },
  "type": "object",
  "properties": {
    "name": {"$ref": "#/definitions/name"},
    "project": {
      "type": "string",
    },
    "type": {
      "type": ["string", "null"],
      "enum": [None, "offline", "random"],
    },
    "observation_budget": {
      "type": ["integer", "null"],
    },
    "metrics": {
      "type": ["array"],
      "minItems": 1,
      "items": {
        "type": ["string", "object"],
        "properties": {
          "name": {"$ref": "#/definitions/opt_name"},
          "objective": {
            "type": ["string", "null"],
            "enum": [None, "maximize", "minimize"],
          },
          "strategy": {
            "type": ["string", "null"],
            "enum": [None, "optimize", "store", "constraint"],
          },
          "threshold": {"type": ["number", "null"]},
          "object": {
            "type": ["string"],
            "enum": ["metric"],
          },
        },
      },
    },
    "parameters": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["name", "type"],
        "properties": {
          "name": {"$ref": "#/definitions/name"},
          "type": {
            "type": "string",
            "enum": ["double", "int", "categorical"],
          },
          "conditions": {
            "type": "object",
            "additionalProperties": {
              "type": ["array", "string"],
              "items": {"type": "string"},
            },
          },
        },
      },
    },
    "conditionals": {
      "type": ["array", "null"],
      "items": {
        "type": "object",
        "required": ["name", "values"],
        "properties": {
          "name": {"$ref": "#/definitions/name"},
          "values": {
            "type": "array",
            "items": {"type": "string"},
          },
        },
      },
    },
    "linear_constraints": {
      "type": ["array", "null"],
      "items": {
        "type": "object",
        "required": ["type"],
        "oneOf": [{"$ref": "#/definitions/constraints/linear_ineq"}],
      },
    },
    "tasks": {
      "type": ["array", "null"],
      "items": {"type": "object", "required": ["name", "cost"], "oneOf": [{"$ref": "#/definitions/task"}]},
    },
    "metadata": {
      "type": ["object", "null"],
    },
    "num_solutions": {
      "type": ["integer", "null"],
      "minimum": 1,
    },
    "parallel_bandwidth": {
      "type": "integer",
      "minimum": 1,
    },
  },
  "required": ["parameters", "metrics"],
}

OBSERVATION_CREATE_SCHEMA = {
  "properties": {
    "suggestion": {"type": "integer", "minimum": 1},
    "assignments": {},
    "values": {
      "items": {
        "type": ["array", "null", "object"],
        "required": ["value"],
        "properties": {
          "name": {"type": ["string", "null"]},
          "value": {
            "type": ["number"],
          },
          "value_stddev": {
            "type": ["number", "null"],
            "minimum": 0.0,
          },
        },
      }
    },
    "failed": {
      "type": ["boolean", "null"],
    },
    "metadata": {
      "type": ["object", "null"],
    },
    "task": {
      "type": ["object", "string", "null"],
    },
  },
}


def get_num_distinct_elements(lst):
  return len(set(lst))



def validate_experiment(experiment, cls_name):
  validate_parameters_for_experiment(experiment)
  validate_metrics_for_experiment(experiment)

  if not experiment.parallel_bandwidth == 1:
    raise ValueError(f"{cls_name} must have parallel_bandwidth == 1")

  observation_budget = experiment.observation_budget
  if observation_budget is None:
    if experiment.num_solutions > 1:
      raise ValueError(f"observation_budget is required for a {cls_name} with multiple solutions")
    if experiment.requires_pareto_frontier_optimization:
      raise ValueError(f"observation_budget is required for a {cls_name} with more than one optimized metric")
    if experiment.has_constraint_metrics:
      raise ValueError(f"observation_budget is required for a {cls_name} with constraint metrics")
    if experiment.is_multitask:
      raise ValueError(f"observation_budget is required for a {cls_name} with tasks (multitask)")

  if not (experiment.optimized_metrics or experiment.constraint_metrics):
    raise ValueError(f"{cls_name} must have optimized or constraint metrics")

  if experiment.optimized_metrics:
    if not len(experiment.optimized_metrics) in [1, 2]:
      raise ValueError(f"{cls_name} must have one or two optimized metrics")
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
      raise ValueError(f"{cls_name} with multiple solutions require exactly one optimized metric")

  # Check conditional limitation
  parameters_have_conditions = any(parameter.conditions for parameter in experiment.parameters)
  if parameters_have_conditions ^ experiment.is_conditional:
    raise ValueError(
      f"For conditional {cls_name}, need both conditions defined in parameters and conditionals variables"
      " defined in experiment"
    )
  if experiment.is_conditional:
    if num_solutions and num_solutions > 1:
      raise ValueError(f"{cls_name} with multiple solutions does not support conditional parameters")
    if experiment.is_search:
      raise ValueError(f"All-Constraint {cls_name} does not support conditional parameters")
    validate_conditionals_for_experiment(experiment)

  # Check feature viability of multitask
  tasks = experiment.tasks
  if tasks:
    if experiment.requires_pareto_frontier_optimization:
      raise ValueError(f"{cls_name} cannot have both tasks and multiple optimized metrics")
    if experiment.has_constraint_metrics:
      raise ValueError(f"{cls_name} cannot have both tasks and constraint metrics")
    if num_solutions and num_solutions > 1:
      raise ValueError(f"{cls_name} with multiple solutions cannot be multitask")
    validate_tasks_for_experiment(experiment, cls_name)

  if experiment.linear_constraints:
    validate_constraints_for_experiment(experiment)
    check_constraint_feasibility_for_experiment(experiment)



def validate_parameter(parameter):
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
    if not get_num_distinct_elements(parameter.grid) == len(parameter.grid):
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



def validate_observation(observation, experiment):
  for parameter in experiment.parameters:
    if parameter_conditions_satisfied(parameter, observation.assignments):
      observation_must_have_parameter(observation, parameter)
    else:
      observation_does_not_have_parameter(observation, parameter)

  validate_observation_conditionals(observation, experiment.conditionals)

  if experiment.is_multitask:
    validate_observation_tasks(observation, experiment.tasks)

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


def check_all_conditional_values_satisfied_for_experiment(experiment):
  num_conditional_values = numpy.product([len(c.values) for c in experiment.conditionals])
  satisfied_parameter_configurations = set([])
  for parameter in experiment.parameters:
    conditional_values = []
    for conditional in experiment.conditionals:
      parameter_conditions = {x.name: x.values for x in parameter.conditions}
      if conditional.name in parameter_conditions:
        conditional_values.append(parameter_conditions[conditional.name])
      else:  # If that conditional is not present for a parameter, then add all values
        conditional_values.append([x.name for x in conditional.values])
    for selected_conditionals in itertools.product(*conditional_values):
      satisfied_parameter_configurations.add(selected_conditionals)
  if len(satisfied_parameter_configurations) != num_conditional_values:
    raise ValueError("Need at least one parameter that satisfies each conditional value")



def check_constraint_feasibility_for_experiment(experiment):
  def parse_constraints_to_halfspaces(constraints, parameters):
    constrained_parameters_names = []
    for constraint in constraints:
      constrained_parameters_names += [term.name for term in constraint.terms]
    constrained_parameters_names = list(set(constrained_parameters_names))

    constrained_parameters = [p for p in parameters if p.name in constrained_parameters_names]
    dim = len(constrained_parameters)
    num_explicit_constraints = len(constraints)
    n_halfspaces = 2 * dim + num_explicit_constraints
    halfspaces = numpy.zeros((n_halfspaces, dim + 1))

    for ic, constraint in enumerate(constraints):
      # Invert less_than constraints
      assert constraint.type in (ConstraintType.greater_than, ConstraintType.less_than)
      sign = -1 if constraint.type == ConstraintType.less_than else 1

      halfspaces[ic, -1] = sign * constraint.threshold
      nonzero_coef_map = {a.name: a.weight for a in constraint.terms}
      for ip, p in enumerate(constrained_parameters):
        if p.name in nonzero_coef_map:
          halfspaces[ic, ip] = -sign * nonzero_coef_map[p.name]

    for index, p in enumerate(constrained_parameters):
      imin = num_explicit_constraints + 2 * index
      imax = num_explicit_constraints + 2 * index + 1

      halfspaces[imin, -1] = p.bounds.min
      halfspaces[imax, -1] = -p.bounds.max
      halfspaces[imin, index] = -1
      halfspaces[imax, index] = 1

    return halfspaces

  halfspaces = parse_constraints_to_halfspaces(experiment.linear_constraints, experiment.parameters)
  _, _, feasibility = find_interior_point(halfspaces)
  if not feasibility:
    raise ValueError("Infeasible constraints")



def validate_constraints_for_experiment(experiment):
  parameter_names = []
  double_params_names = []
  integer_params_names = []
  unconditioned_params_names = []
  log_transform_params_names = []
  grid_param_names = []
  for p in experiment.parameters:
    parameter_names.append(p.name)
    if p.grid:
      grid_param_names.append(p.name)
    if p.type == DOUBLE_EXPERIMENT_PARAMETER_NAME:
      double_params_names.append(p.name)
    if p.type == INT_EXPERIMENT_PARAMETER_NAME:
      integer_params_names.append(p.name)
    if p.type in [DOUBLE_EXPERIMENT_PARAMETER_NAME, INT_EXPERIMENT_PARAMETER_NAME]:
      if not p.conditions:
        unconditioned_params_names.append(p.name)
      if p.transformation == ParameterTransformationNames.LOG:
        log_transform_params_names.append(p.name)

  constrained_integer_variables = set()
  for c in experiment.linear_constraints:
    terms = c.terms
    constraint_var_set = set()
    if len(terms) <= 1:
      raise ValueError("Constraint must have more than one term")

    term_types = []
    for term in terms:
      coeff = term.weight
      if coeff == 0:
        continue
      name = term.name
      if name in integer_params_names:
        constrained_integer_variables.add(name)
      if name not in parameter_names:
        raise ValueError(f"Variable {name} is not a known parameter")
      if name not in double_params_names and name not in integer_params_names:
        raise ValueError(f"Variable {name} is not a parameter of type `double` or type `int`")
      else:
        term_types.append(
          DOUBLE_EXPERIMENT_PARAMETER_NAME if name in double_params_names else INT_EXPERIMENT_PARAMETER_NAME
        )
      if name not in unconditioned_params_names:
        raise ValueError(f"Constraint cannot be defined on a conditioned parameter {name}")
      if name in log_transform_params_names:
        raise ValueError(f"Constraint cannot be defined on a log-transformed parameter {name}")
      if name in grid_param_names:
        raise ValueError(f"Constraint cannot be defined on a grid parameter {name}")
      if name in constraint_var_set:
        raise ValueError(f"Duplicate constrained variable name: {name}")
      else:
        constraint_var_set.add(name)

    if len(set(term_types)) > 1:
      raise ValueError("Constraint functions cannot mix integers and doubles. One or the other only.")



def validate_tasks_for_experiment(experiment, class_name):
  if len(experiment.tasks) < 2:
    raise ValueError(f"For multitask {class_name}, at least 2 tasks must be present")
  costs = [t.cost for t in experiment.tasks]
  num_distinct_task = get_num_distinct_elements([t.name for t in experiment.tasks])
  num_distinct_costs = get_num_distinct_elements(costs)
  if not num_distinct_task == len(experiment.tasks):
    raise ValueError(f"For multitask {class_name}, all task names must be distinct")
  if not num_distinct_costs == len(experiment.tasks):
    raise ValueError(f"For multitask {class_name}, all task costs must be distinct")
  if 1 not in costs:
    raise ValueError(f"For multitask {class_name}, exactly one task must have cost == 1 (none present).")



def validate_conditionals_for_experiment(experiment):
  conditional_names = [c.name for c in experiment.conditionals]
  if not len(conditional_names) == get_num_distinct_elements(conditional_names):
    raise ValueError(f"No duplicate conditionals are allowed: {conditional_names}")

  for parameter in experiment.parameters:
    if parameter.conditions and any(c.name not in conditional_names for c in parameter.conditions):
      unsatisfied_condition_names = [c.name for c in parameter.conditions if c.name not in conditional_names]
      raise ValueError(
        f"The parameter {parameter.name} has conditions {unsatisfied_condition_names} that are not part of"
        " the conditionals"
      )

  check_all_conditional_values_satisfied_for_experiment(experiment)



def validate_metrics_for_experiment(experiment):
  metric_names = [m.name for m in experiment.metrics]
  if not len(metric_names) == get_num_distinct_elements(metric_names):
    raise ValueError(f"No duplicate metrics are allowed: {metric_names}")



def validate_parameters_for_experiment(experiment):
  param_names = [p.name for p in experiment.parameters]
  if not len(param_names) == get_num_distinct_elements(param_names):
    raise ValueError(f"No duplicate parameters are allowed: {param_names}")



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



def observation_does_not_have_parameter(observation, parameter):
  if parameter.name in observation.assignments:
    raise ValueError(
      f"Parameter {parameter.name} does not satisfy conditions. "
      f"Observation assignments: {observation.assignments} is invalid."
    )



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
