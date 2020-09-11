# 基于  Github Action 的 CI/CD 流程

## 前言
在大型的开源项目或者软件开发过程中， 很多开发者都会去提交`PR`或者进行代码的 `push`操作。如果对于每次代码合并都需要项目的核心维护者进行 `code review`，这项工作是及其困难而且耗时的。因此许多团队都会指定一套代码规范, 然后编写测试用例严格的检查每次代码修改， 这样能够非常有效的减少后期代码维护的成本。

现在，基于 $Github  Action$, 我们可以自动化的完成代码的 `CI/CD' 工作流。$Github Ac­tions$是 GitHub 推出的持续集成 (Con­tin­u­ous in­te­gra­tion，简称 CI) 服务，它提供了配置非常不错的虚拟服务器环境，基于它可以进行构建、测试、打包、部署项目。简单来讲就是将软件开发中的一些流程交给云服务器自动化处理，比方说开发者把代码 push 到 GitHub 后它会自动测试、编译、发布。有了持续集成服务开发者就可以专心于写代码，其它乱七八糟的事情就不用管了，这样可以大大提高开发效率。本篇文章将介绍 GitHub Ac­tions 的基本使用方法。

## [Git hooks](https://githooks.com)
### 1. 什么是 Git hooks


Git hooks 是 Git 在事件之前或之后执行的脚本, 用于控制 git 工作的流程。`Git hooks` 脚本对于我们提交`code review` 之前识别一些简单的问题很有用。 我们在每次提交代码时都会触发这些 hooks，以自动指出代码中的问题，例如缺少分号，尾随空白和调试语句。通过在`code review` 之前指出这些问题，代码审阅者可以专注于代码结构和功能的更改，而不需要浪费时间来审查这些格式问题。

Git hooks 分为客户端钩子和服务端钩子。客户端钩子由诸如提交和合并这样的操作所调用，而服务器端钩子作用于诸如接收被推送的提交这样的联网操作。

客户端钩子：`pre-commit`、`prepare-commit-msg`、`commit-msg`、`post-commit`等，主要用于控制客户端 git 的提交和合并这样的操作。

服务端钩子：pre-receive、post-receive、update，主要在服务端接收提交对象时、推送到服务器之前调用。

- pre-commit: Check the commit message for spelling errors.
- pre-receive: Enforce project coding standards.
- post-commit: Email/SMS team members of a new commit.
- post-receive: Push the code to production.

### 2. Git hooks 如何工作？

每个Git存储库都有一个`.git/hooks`文件夹，其中包含可以绑定到的每个钩子的脚本。您可以根据需要随意更改或更新这些脚本，Git将在这些事件发生时执行它们。进去`.git/hooks` 后会看到一些 `hooks` 的官方示例，他们都是以`.sample`结尾的文件名。

注意这些以`.sample`结尾的示例脚本是不会执行的，如果你想启用它们，得先移除这个后缀。

例如下面的文件列表是 `git init` 在 `.git/hooks` 文件夹下自动创建的 `hooks` 方法。

```python
-rwxrwxr-x 1 robin robin  478 Jun  1 17:54 applypatch-msg.sample*
-rwxrwxr-x 1 robin robin  896 Jun  1 17:54 commit-msg.sample*
-rwxrwxr-x 1 robin robin  189 Jun  1 17:54 post-update.sample*
-rwxrwxr-x 1 robin robin  424 Jun  1 17:54 pre-applypatch.sample*
-rwxrwxr-x 1 robin robin 1642 Jun  1 17:54 pre-commit.sample*
-rwxrwxr-x 1 robin robin 1239 Jun  1 17:54 prepare-commit-msg.sample*
-rwxrwxr-x 1 robin robin 1348 Jun  1 17:54 pre-push.sample*
-rwxrwxr-x 1 robin robin 4898 Jun  1 17:54 pre-rebase.sample*
-rwxrwxr-x 1 robin robin 3610 Jun  1 17:54 update.sample*
```

例如下面的文件列表中多了一个 `pre-commit` 的文件， 是我添加的用于代码检查的 `pre-commit hook`方法。

```python
-rwxrwxr-x 1 robin robin  478 Sep  7 10:25 applypatch-msg.sample*
-rwxrwxr-x 1 robin robin  896 Sep  7 10:25 commit-msg.sample*
-rwxrwxr-x 1 robin robin  189 Sep  7 10:25 post-update.sample*
-rwxrwxr-x 1 robin robin  424 Sep  7 10:25 pre-applypatch.sample*
-rwxrwxr-x 1 robin robin 1475 Sep  8 20:13 pre-commit*
-rwxrwxr-x 1 robin robin 1642 Sep  7 10:25 pre-commit.sample*
-rwxrwxr-x 1 robin robin 1239 Sep  7 10:25 prepare-commit-msg.sample*
-rwxrwxr-x 1 robin robin 1348 Sep  7 10:25 pre-push.sample*
-rwxrwxr-x 1 robin robin 4898 Sep  7 10:25 pre-rebase.sample*
-rwxrwxr-x 1 robin robin 3610 Sep  7 10:25 update.sample*
```

### 3.[pre-commit](https://pre-commit.com) 简介

`pre-commit`是客户端hooks之一，`pre-commit` 在 `git add`提交之后，然后执行 `git commit` 时执行, 实现对代码的审查。 脚本执行没报错就继续提交，反之就驳回提交的操作。

这个钩子脚本可用于处理简单问题：对将要提交的代码进行检查、优化代码格式, 例如检查是字符数是否超出限制，尾随空格和调试语句等。

Git钩子。我们在每次提交时运行我们的钩子，以。本文以python 项目为例。



#### (1).安装
```python
## 使用 pip 安装:
pip install pre-commit
```
#### (2). 配置
在项目根目录填加 `.pre-commit-config.yaml` 文件, 这里以 `mmdetection` 的配置文件为例来做说明：
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

#### (3). 安装 git hook scripts
```python
# pre-commit install
pre-commit install
# pre-commit installed at .git/hooks/pre-commit
```
第一次 pre-commit 运行时，将会自动下载、安装并且运行 hook。安装完成之后，`pre-commit` 将会在每次运行`git commit`命令时自动执行。 注意： 每次 clone 代码后，都需要执行 `pre-commit install`。

#### (4). 手动触发
第一次，需要触发全部：

```python
(pytorch) robin@robin-Z390-UD:~/jianzh/ApulisVision$ pre-commit run --all-files
flake8...................................................................Passed
seed isort known_third_party.............................................Passed
isort....................................................................Passed
yapf.....................................................................Passed
Trim Trailing Whitespace.................................................Failed
- hook id: trailing-whitespace
- exit code: 1
- files were modified by this hook

Fixing docs/GithubAction.md

Check Yaml...............................................................Passed
Fix End of Files.........................................................Passed
Fix requirements.txt.....................................................Passed
Fix double quoted strings................................................Passed
Check for merge conflicts................................................Passed
Fix python encoding pragma...............................................Passed
Mixed line ending........................................................Passed
docformatter.............................................................Passed
```
可以看到， 我的文档中， 存在一些 `trailing whitespace`


##
#### (5). 支持语言
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
## $Github Action$

### Let’s get started

#### 1. Create Actions (workflow)

$$Github Action$s$ 必须在`.github/workflow` 文件夹中创建，以便Github可以访问它. 创建文件夹结构，请在代码根目录下运行以下命令.

```python
mkdir .github/
mkdir .github/workflows/
```
