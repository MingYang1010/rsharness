from .schemas import BudgetCounter, BudgetCounters, BudgetSpec


def _counter(limit: int, used: int = 0) -> BudgetCounter:
    bounded_used = min(max(used, 0), limit)
    return BudgetCounter(
        limit=limit,
        used=bounded_used,
        remaining=max(limit - bounded_used, 0),
    )


def initial_budget(specification: BudgetSpec) -> BudgetCounters:
    return BudgetCounters(
        steps=_counter(specification.max_steps),
        tool_calls=_counter(specification.max_tool_calls),
        wall_time_ms=_counter(specification.max_wall_time_ms),
        input_bytes=_counter(specification.max_input_bytes),
        artifact_bytes=_counter(specification.max_artifact_bytes),
    )


def update_budget(
    current: BudgetCounters,
    elapsed_wall_time_ms: int,
    step_increment: int = 0,
    tool_call_increment: int = 0,
    input_byte_increment: int = 0,
    artifact_byte_increment: int = 0,
) -> BudgetCounters:
    return BudgetCounters(
        steps=_counter(
            current.steps.limit,
            current.steps.used + step_increment,
        ),
        tool_calls=_counter(
            current.tool_calls.limit,
            current.tool_calls.used + tool_call_increment,
        ),
        wall_time_ms=_counter(current.wall_time_ms.limit, elapsed_wall_time_ms),
        input_bytes=_counter(
            current.input_bytes.limit,
            current.input_bytes.used + input_byte_increment,
        ),
        artifact_bytes=_counter(
            current.artifact_bytes.limit,
            current.artifact_bytes.used + artifact_byte_increment,
        ),
    )


def exhausted(budget: BudgetCounters) -> bool:
    return budget.steps.remaining == 0 or budget.wall_time_ms.remaining == 0
