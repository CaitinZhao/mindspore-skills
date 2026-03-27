# Consistency Validation

After building an accuracy profile, validate the most likely consistency causes
against real evidence.

Typical groups:

- data consistency
- config consistency
- model consistency
- checkpoint consistency
- dtype and precision consistency
- framework or platform consistency
- metric and evaluation consistency

Every root-cause claim should carry:

- confidence
- supporting evidence
- a validation check
- a small next experiment

## Operator-Level Note

If framework or platform consistency checks narrow the first stable mismatch to
one operator, load `references/operator-accuracy-triage.md` and finish that
operator-specific validation before claiming the operator is the root cause.

Before escalating from a mismatching module to an operator claim, verify that
the module already has aligned:

- inputs
- weights
- dtype
- API parameters
- actual device placement

Use evidence to justify the claim rather than intuition. Intuition may help you
choose the next check, but it is not evidence on its own.
