# 基于  Github  Action 的 CI/CD 流程

在开发`Python`代码时，程序员会经常因为代码格式不统一等各种问题，采用`pre-commit`可以解决此类问题。

## [Git hooks](https://githooks.com)
### 什么是 Git hooks

Git hooks 是Git在事件之前或之后执行的脚本，例如：commit，push和receive。Git hooks 是一个内置功能 - 无需下载任何东西。Git hooks 在本地运行。

一些 sample hooks 脚本如下：

- pre-commit: 检查提交消息是否有拼写错误
- pre-receive: Enforce project coding standards.
- post-commit: Email/SMS team members of a new commit.
- post-receive: Push the code to production.

### Git hooks如何工作？

每个Git存储库都有一个.git/hooks文件夹，其中包含可以绑定到的每个钩子的脚本。您可以根据需要随意更改或更新这些脚本，Git将在这些事件发生时执行它们。

#### [pre-commit](https://pre-commit.com) 简介
pre-commit 是 git hooks 的一个子集，实现在提交代码审查之前，Git钩子脚本可用于处理简单问题。我们在每次提交时运行我们的钩子，以自动指出代码中的问题，例如缺少分号，尾随空格和调试语句。本文以python 项目为例。

##### 安装
```python
## 使用 pip 安装:
pip install pre-commit
```
##### 配置
在项目根目录填加 .pre-commit-config.yaml 文件, 这里以 `mmdetection` 的配置文件为例来做说明：
```python
repos:
  - repo: https://gitlab.com/pycqa/flake8.git
    rev: 3.8.3
    hooks:
      - id: flake8
  - repo: https://github.com/asottile/seed-isort-config
    rev: v2.2.0
    hooks:
      - id: seed-isort-config
  - repo: https://github.com/timothycrosley/isort
    rev: 4.3.21
    hooks:
      - id: isort
  - repo: https://github.com/pre-commit/mirrors-yapf
    rev: v0.30.0
    hooks:
      - id: yapf
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v3.1.0
    hooks:
      - id: trailing-whitespace
      - id: check-yaml
      - id: end-of-file-fixer
      - id: requirements-txt-fixer
      - id: double-quote-string-fixer
      - id: check-merge-conflict
      - id: fix-encoding-pragma
        args: ["--remove"]
      - id: mixed-line-ending
        args: ["--fix=lf"]
  - repo: https://github.com/myint/docformatter
    rev: v1.3.1
    hooks:
      - id: docformatter
        args: ["--in-place", "--wrap-descriptions", "79"]
```

其中 flake8 根据 flake8 给出的代码规则检查代码.

##### pre-commit 使用

```python
# pre-commit install
pre-commit install
# pre-commit installed at .git/hooks/pre-commit
```
pre-commit 将会在每次提交前执行，每次 clone 代码后，都需要执行 pre-commit install


#####  手动触发
第一次，需要触发全部：
```python
pre-commit run --all-files
```
第一次 pre-commit 运行时，将会自动下载、安装并且运行 hook

##### 支持语言
```python
docker
docker_image
fail
golang
node
python
python_venv
ruby
rust
swift
pcre
pygrep
script
system
```

## CI
## CD
## Github Action
