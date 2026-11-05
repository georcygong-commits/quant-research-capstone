# -*- coding: utf-8 -*-
"""
容器化与 CI/CD（M6.7）：
生成 Dockerfile、docker-compose 与 CI 流水线配置（本地无需真实 Docker 即可产出骨架）。
"""


def generate_dockerfile(python_version="3.10", entrypoint="run_all.py"):
    """生成可直接构建的 Dockerfile 文本。"""
    return "\n".join([
        "FROM python:%s-slim" % python_version,
        "WORKDIR /app",
        "COPY requirements.txt .",
        "RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple",
        "COPY . .",
        'CMD ["python", "%s"]' % entrypoint,
    ])


def generate_compose(services):
    """生成 docker-compose 服务编排骨架。"""
    lines = ["services:"]
    for name, cmd in services.items():
        lines.append("  %s:" % name)
        lines.append("    build: .")
        lines.append('    command: ["python", "%s"]' % cmd)
        lines.append("    restart: unless-stopped")
    return "\n".join(lines)


def generate_pipeline(stages=("lint", "test", "build", "deploy")):
    """生成 CI/CD 流水线配置（GitHub Actions 风格）。"""
    lines = ["name: ci", "on: [push]", "jobs:", "  build:",
             "    runs-on: ubuntu-latest", "    steps:",
             "      - uses: actions/checkout@v4",
             "      - uses: actions/setup-python@v5",
             "        with:", "          python-version: '3.10'"]
    for s in stages:
        lines.append("      - name: %s" % s)
        if s == "lint":
            lines.append("        run: python -m flake8 .")
        elif s == "test":
            lines.append("        run: python -m pytest -q")
        elif s == "build":
            lines.append("        run: docker build -t quant-app .")
        else:
            lines.append("        run: echo deploy")
    return "\n".join(lines)


class DeployManifest:
    """部署清单：登记服务并校验入口文件非空。"""
    def __init__(self):
        self.services = {}

    def add(self, name, entrypoint):
        self.services[name] = entrypoint

    def validate(self):
        bad = [n for n, c in self.services.items() if not c]
        return {"valid": len(bad) == 0, "bad": bad, "n": len(self.services)}
