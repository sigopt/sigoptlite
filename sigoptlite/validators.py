import numpy

from libsigopt.aux.geometry_utils import find_interior_point

from libsigopt.aux.constant import (
  DOUBLE_EXPERIMENT_PARAMETER_NAME,
  INT_EXPERIMENT_PARAMETER_NAME,
  ConstraintType,
  ParameterTransformationNames,
)

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

def check_all_conditional_values_satisfied(experiment):
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

def check_constraint_feasibility(experiment):
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

def validate_constraints(experiment):
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

def validate_tasks(experiment, class_name):
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
