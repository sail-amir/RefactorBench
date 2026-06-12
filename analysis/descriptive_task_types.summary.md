# Descriptive RefactorBench tasks — Fowler refactoring categories

- Tasks: **100**  | Single-type: **87**  | Multi-type: **13**  | Label instances: **113**


## Count by category

| Category | Tasks |
|---|---|
| Add Parameter (Change Fn Declaration) | 19 |
| Move Function | 20 |
| Rename Function/Method | 16 |
| Inline / Merge Module | 14 |
| Remove Dead Code | 13 |
| Rename Variable/Constant/Attribute | 7 |
| Rename File/Module | 6 |
| Combine Functions into Class | 5 |
| Rename Class | 4 |
| Introduce Parameter Object | 4 |
| Move Class | 2 |
| Extract Class | 1 |
| Split Function/Phase | 1 |
| Other / Non-structural | 1 |

## Count by repository

| Repo | Tasks |
|---|---|
| ansible_refactor | 11 |
| celery_refactor | 12 |
| django_refactor | 18 |
| fastapi_refactor | 6 |
| flask_refactor | 6 |
| requests_refactor | 10 |
| salt_refactor | 15 |
| scrapy_refactor | 13 |
| tornado_refactor | 9 |

## Notes on taxonomy fit

- The approved 12-category set was extended with **two** clearly-flagged labels during the meticulous pass:
  - **Rename Class** (4 tasks) — class renames are distinct from function/variable/file renames; the fine-grained set needed it.
  - **Other / Non-structural** (1 task: add-none-handling-duration-string) — adds None-handling behavior; not a structural Fowler refactor.
- **Inline / Merge Module** also covers the two function-level combines (combine-unpickle, combine-from-key-to-key) and the import-canonicalization task (object-mro-lookup); noted per-task in `evidence`.
- Multi-label was used only for whole-file merges that also delete the source (**Inline / Merge Module + Remove Dead Code**) and the one move-then-delete-file task (**Move Function + Remove Dead Code**).
