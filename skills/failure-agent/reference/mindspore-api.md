# MindSpore API Diagnosis Notes

When stack is `ms`, focus on:

- API call site and parameter validity
- Graph vs PyNative behavior differences
- Unsupported operator or backend-specific constraints
- Operator registration or dispatch mismatches

If issue is confirmed as MindSpore operator implementation gap, route to builder skills in this repo.
