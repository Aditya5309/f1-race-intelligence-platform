from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="f1-race-winner-prediction",
    version="0.1.0",
    description="Production ML project for F1 race winner prediction",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=read_requirements(),
)
