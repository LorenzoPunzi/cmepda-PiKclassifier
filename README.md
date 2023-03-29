# cmepda-PiKclassifier

A brief description of what this project does and who it's for

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/LorenzoPunzi/cmepda-PiKclassifier/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/LorenzoPunzi/cmepda-PiKclassifier/tree/main)

[![ReadTheDocs](https://readthedocs.org/projects/docs/badge/?version=latest)](https://cmepda-pikclassifier.readthedocs.io/en/latest/index.html)


## Authors

- [@lorenzopunzi](https://github.com/LorenzoPunzi)
- [@rubenforti](https://github.com/rubenforti)


## Documentation

[Documentation](https://cmepda-pikclassifier.readthedocs.io/en/latest/index.html)


## Run Locally

Clone the project

```bash
  git clone https://github.com/LorenzoPunzi/cmepda-PiKclassifier.git
```

Install dependencies from inside the project directory

```bash
  pip install -r cmepda-PiKclassifier/requirements.txt
```
!!!PLEASE NOTE!!! : ROOT is not in requirements.txt file but is needed to run the template fit module. The package works as of ROOT v6.26/10.
Also, the requirements are NOT strict, meaning that former versions could potentially work.

To run the package as a whole using the main with parser

```bash
  python cmepda-Pikclassifier [generic options] <subparser name> [subparser options]
```
for example "python cmepda-Pikclassifier -fig analysis -m dnn dtc -ld"

## Running Tests

To run tests, run the following command inside the project directory

```bash
  python -m unittest discover -s tests
```
