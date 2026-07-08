# PathTraversal Extract Class Comparison

This note explains the evidence behind the visual:

- `extract-class-task-pathtraversal-pangu-7b-vs-glm-5-1-vs-gpt-5-2-vs-opus-4-7-comparison.png`
- `extract-class-task-pathtraversal-pangu-7b-vs-glm-5-1-vs-gpt-5-2-vs-opus-4-7-comparison.svg`

## Task

The descriptive RefactorBench task asks the model to create a new Django exception class named `PathTraversal` in `django/core/exceptions.py`. The new class should inherit from `SuspiciousOperation`. The patch must then update Django storage path-validation behavior, documentation, and functional tests so path traversal cases use the new exception.

The relevant refactoring type is `Extract Class`: a specific path-traversal failure concept is extracted out of broader file-operation error handling into its own exception class.

## Checker Expectations

For this task, passing requires more than adding a class. The official checker validates several surfaces:

- `PathTraversal` exists in `django/core/exceptions.py`.
- `PathTraversal` directly extends `SuspiciousOperation`.
- Storage path-validation methods raise `PathTraversal` where needed.
- `django/utils/_os.py::safe_join()` raises `PathTraversal` for traversal.
- Existing functional tests are updated, especially `tests/utils_tests/test_os_utils.py`.
- Documentation, including `docs/ref/exceptions.txt`, reflects the new exception.

The important point for the comparison: all failed patches applied successfully. They failed semantic checker requirements, not patch application.

## Summary

| Model | Result | Main reason |
| --- | --- | --- |
| Pangu-7B | PASS | Covered the class, runtime behavior, docs, and the existing official test file. |
| GLM-5.1 | FAIL | Missed `tests/utils_tests/test_os_utils.py`, even though it changed much of the runtime code. |
| GPT-5.2 | FAIL | Only made a partial class/docs/import change; missed `safe_join()` behavior and `test_os_utils.py`. |
| Opus 4.7 | FAIL | Used the wrong direct base class and missed `safe_join()` behavior. |

## Pangu-7B: Why It Passed

Pangu-7B covered the checker-required surfaces end to end.

It added the new exception class with the expected base:

```python
class PathTraversal(SuspiciousOperation):
    ...
```

It updated runtime behavior in the storage and path utility code, including:

- `django/core/files/storage/base.py`
- `django/core/files/utils.py`
- `django/utils/_os.py`
- related template/file handling call sites

Most importantly, it updated the existing test file that the checker explicitly inspects:

```diff
diff --git a/tests/utils_tests/test_os_utils.py b/tests/utils_tests/test_os_utils.py
-from django.core.exceptions import SuspiciousFileOperation
+from django.core.exceptions import PathTraversal
...
-        with self.assertRaises(SuspiciousFileOperation):
+        with self.assertRaises(PathTraversal):
             safe_join("/abc/", "../def")
```

That is the key difference from the failed models. Pangu-7B did not just implement the runtime change; it also updated the existing official test surface that validates the refactor.

## GLM-5.1: Why It Failed

GLM-5.1 made a substantial patch, and the patch applied cleanly. It did several things correctly:

- Added `PathTraversal(SuspiciousOperation)`.
- Updated storage-related code to raise `PathTraversal`.
- Updated `django/utils/_os.py::safe_join()` to raise `PathTraversal`.
- Updated some docs.

However, it did not modify:

```text
tests/utils_tests/test_os_utils.py
```

The checker failure was:

```text
PathTraversal is not imported in /tests/utils_tests/test_os_utils.py
```

GLM-5.1 also added several root-level ad-hoc test scripts such as `test_path_traversal.py` and `test_final_comprehensive.py`. Those scripts may show that the model tried to verify behavior, but they do not satisfy RefactorBench's official checker. The checker expects the existing Django test file to be updated, not unrelated new scripts.

So GLM-5.1 failed because it solved much of the runtime refactor but missed an existing checker-required test surface.

## GPT-5.2: Why It Failed

GPT-5.2 produced a much smaller patch. The patch applied cleanly, but it was incomplete.

It added the class with the correct base:

```python
class PathTraversal(SuspiciousOperation):
    ...
```

It also updated `docs/ref/exceptions.txt` and added imports in a couple of files. But it did not complete the behavior change. In particular:

- It did not update `tests/utils_tests/test_os_utils.py`.
- It did not touch `django/utils/_os.py`.
- Therefore `safe_join()` still did not raise `PathTraversal`.
- It did not substantially update the runtime raise sites needed for the full task.

The checker failures included:

```text
PathTraversal is not imported in /tests/utils_tests/test_os_utils.py
PathTraversal exception is not raised in safe_join
```

So GPT-5.2 failed because it recognized the new class name and documentation requirement, but did not propagate the refactor through the required runtime behavior and official test surface.

## Opus 4.7: Why It Failed

Opus 4.7 did more runtime work than GPT-5.2, but it made a critical class hierarchy mistake and still missed `safe_join()`.

It added:

```python
class PathTraversal(SuspiciousFileOperation):
    ...
```

The task and checker expected:

```python
class PathTraversal(SuspiciousOperation):
    ...
```

The checker failure was:

```text
'SuspiciousOperation' not found in ['SuspiciousFileOperation'] : PathTraversal does not extend SuspiciousOperation
```

Opus 4.7 did update some storage methods and file-storage tests, including `Storage.get_available_name()` and `Storage.generate_filename()` cases. But it did not update:

```text
django/utils/_os.py
```

As a result, `safe_join()` still did not raise `PathTraversal`, producing another checker failure:

```text
PathTraversal exception is not raised in safe_join
```

So Opus 4.7 failed for two independent reasons: the direct base class was wrong, and one required runtime path, `safe_join()`, was still incomplete.

## Interpretation

This task rewards complete repository-wide propagation. The class definition alone is not enough. The successful patch had to connect four pieces:

1. Add `PathTraversal`.
2. Raise it in all relevant path-traversal runtime paths.
3. Update existing tests that encode the expected behavior.
4. Update docs.

Pangu-7B passed because it connected those pieces. GLM-5.1, GPT-5.2, and Opus 4.7 each left at least one checker-required surface incomplete.
