# 上游源码快照

`code/` 由 `scripts/refresh_curated_sources.py` 从 GitHub 默认分支临时浅克隆并生成。脚本完成后会自动删除 clone；本目录不长期保存 `upstream/`。

| 项目 | GitHub | 分支 | 快照 commit | commit 时间 |
|---|---|---|---|---|
| FlashInfer | <https://github.com/flashinfer-ai/flashinfer> | `main` | `08ddfbcd2e89b2f4b68391825817909e30d445e2` | 2026-08-03 12:22 +08:00 |
| FLA | <https://github.com/fla-org/flash-linear-attention> | `main` | `0b346347379476548be0678ec597f9e14f148bf7` | 2026-08-03 11:05 +08:00 |
| FlashKDA | <https://github.com/MoonshotAI/FlashKDA> | `master` | `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` | 2026-07-29 02:21 UTC |
| cuLA | <https://github.com/inclusionAI/cuLA> | `main` | `43bcbcf30643a23fe3e0e0dcaa22e1d3d7970e74` | 2026-08-03 11:49 +08:00 |

每个 `code/<project>/` 目录还包含机器可读的：

- `UPSTREAM_REVISION.md`：仓库、commit 与 commit 时间；
- `MANIFEST.sha256`：精选文件校验值；
- 原项目 `LICENSE`（FlashInfer 另含 `NOTICE`）。

## 刷新快照

确保系统安装了 `git`；推荐安装并登录 GitHub CLI `gh`，然后运行：

```bash
python scripts/refresh_curated_sources.py
```

脚本会：

1. 在 `liner_attn/.source-refresh-*` 临时目录浅克隆四个仓库；
2. 生成新的精选源码、revision 和 checksum；
3. 所有仓库都成功后才替换 `code/`；
4. 自动清理临时 clone。

若没有 `gh`，脚本回退到 HTTPS `git clone`。刷新后应根据各项目的 `UPSTREAM_REVISION.md` 同步本表，并重新检查 API 和约束。

## 构建完整上游项目

精选快照用于阅读，不保证包含完整导入依赖。需要构建或运行上游测试时，在工作区之外单独 clone：

```bash
# 示例：FlashKDA 和 cuLA 都需要递归初始化 CUTLASS 等子模块
git clone --recurse-submodules https://github.com/MoonshotAI/FlashKDA.git /tmp/FlashKDA
git clone --recurse-submodules https://github.com/inclusionAI/cuLA.git /tmp/cuLA
```

具体环境与安装命令以各项目当前 README 为准。

## 许可证

- FlashInfer：Apache-2.0，另见其 `NOTICE`；
- FLA：MIT；
- FlashKDA：MIT；
- cuLA：Apache-2.0。

复用源码时不要移除原版权头，并遵守对应许可证。
