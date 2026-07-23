# CNPort test results

`scripts/run_platform_tests.py` 将每次测试写入：

```text
tests/results/<platform>/<YYYYMMDDHHMMSS>/
```

运行产物默认不提交 Git。本文件和 `summary.schema.json` 用于说明稳定的公开格式；需要随发布保留的结果应先脱敏，再整理为独立验证摘要。
