DIRS = \
	notebooks \
	data \
	src \
	src/data_processing \
	src/model_pipeline \
	src/model_pipeline/training_configs \
	src/model_pipeline/training_results \
	src/final_files \
	src/utils

FILES = \
	src/ __init__.py \
	src/utils/__init__.py \
	src/final_files/__init__.py \
	.gitattributes \
	README.md \
	requirements.txt

.PHONY: all clean tree

all: $(DIRS) $(FILES) .gitignore .gitattributes tree

$(DIRS):
	mkdir -p $@

$(FILES): | $(DIRS)
	touch $@

.gitignore: | $(DIRS)
	@echo "Creating .gitignore with Python template..."
	@echo "# Byte-compiled / optimized / DLL files" > .gitignore
	@echo "__pycache__/" >> .gitignore
	@echo "*.py[cod]" >> .gitignore
	@echo "*$py.class" >> .gitignore
	@echo "" >> .gitignore
	@echo "# C extensions" >> .gitignore
	@echo "*.so" >> .gitignore
	@echo "" >> .gitignore
	@echo "# Distribution / packaging" >> .gitignore
	@echo ".Python" >> .gitignore
	@echo "build/" >> .gitignore
	@echo "develop-eggs/" >> .gitignore
	@echo "dist/" >> .gitignore
	@echo "downloads/" >> .gitignore
	@echo "eggs/" >> .gitignore
	@echo ".eggs/" >> .gitignore
	@echo "lib/" >> .gitignore
	@echo "lib64/" >> .gitignore
	@echo "parts/" >> .gitignore
	@echo "sdist/" >> .gitignore
	@echo "var/" >> .gitignore
	@echo "wheels/" >> .gitignore
	@echo "share/python-wheels/" >> .gitignore
	@echo "*.egg-info/" >> .gitignore
	@echo ".installed.cfg" >> .gitignore
	@echo "*.egg" >> .gitignore
	@echo "MANIFEST" >> .gitignore
	@echo "" >> .gitignore
	@echo "# PyInstaller" >> .gitignore
	@echo "#  Usually these files are written by a python script from a template" >> .gitignore
	@echo "#  before PyInstaller builds the exe, so as to inject date/other infos into it." >> .gitignore
	@echo "*.manifest" >> .gitignore
	@echo "*.spec" >> .gitignore
	@echo "" >> .gitignore
	@echo "# Installer logs" >> .gitignore
	@echo "pip-log.txt" >> .gitignore
	@echo "pip-delete-this-directory.txt" >> .gitignore
	@echo "" >> .gitignore
	@echo "# Unit test / coverage reports" >> .gitignore
	@echo "htmlcov/" >> .gitignore
	@echo ".tox/" >> .gitignore
	@echo ".nox/" >> .gitignore
	@echo ".coverage" >> .gitignore
	@echo ".coverage.*" >> .gitignore
	@echo ".cache" >> .gitignore
	@echo "nosetests.xml" >> .gitignore
	@echo "coverage.xml" >> .gitignore
	@echo "*.cover" >> .gitignore
	@echo "*.py,cover" >> .gitignore
	@echo ".hypothesis/" >> .gitignore
	@echo ".pytest_cache/" >> .gitignore
	@echo "cover/" >> .gitignore
	@echo "" >> .gitignore
	@echo "# Translations" >> .gitignore
	@echo "*.mo" >> .gitignore
	@echo "*.pot" >> .gitignore
	@echo "" >> .gitignore
	@echo "# Django stuff:" >> .gitignore
	@echo "*.log" >> .gitignore
	@echo "local_settings.py" >> .gitignore
	@echo "db.sqlite3" >> .gitignore
	@echo "db.sqlite3-journal" >> .gitignore
	@echo "" >> .gitignore
	@echo "# Flask stuff:" >> .gitignore
	@echo "instance/" >> .gitignore
	@echo ".webassets-cache" >> .gitignore
	@echo "" >> .gitignore
	@echo "# Scrapy stuff:" >> .gitignore
	@echo ".scrapy" >> .gitignore
	@echo "" >> .gitignore
	@echo "# Sphinx documentation" >> .gitignore
	@echo "docs/_build/" >> .gitignore
	@echo "" >> .gitignore
	@echo "# PyBuilder" >> .gitignore
	@echo ".pybuilder/" >> .gitignore
	@echo "target/" >> .gitignore
	@echo "" >> .gitignore
	@echo "# Jupyter Notebook" >> .gitignore
	@echo ".ipynb_checkpoints" >> .gitignore
	@echo "" >> .gitignore
	@echo "# IPython" >> .gitignore
	@echo "profile_default/" >> .gitignore
	@echo "ipython_config.py" >> .gitignore
	@echo "" >> .gitignore
	@echo "# pyenv" >> .gitignore
	@echo "#   For a library or package, you might want to ignore these files since the code is" >> .gitignore
	@echo "#   intended to run in multiple environments; otherwise, check them in:" >> .gitignore
	@echo "# .python-version" >> .gitignore
	@echo "" >> .gitignore
	@echo "# pipenv" >> .gitignore
	@echo "#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control." >> .gitignore
	@echo "#   However, in case of collaboration, if having platform-specific dependencies or dependencies" >> .gitignore
	@echo "#   having no cross-platform support, pipenv may install dependencies that don't work, or not" >> .gitignore
	@echo "#   install all needed dependencies." >> .gitignore
	@echo "#Pipfile.lock" >> .gitignore
	@echo "" >> .gitignore
	@echo "# poetry" >> .gitignore
	@echo "#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control." >> .gitignore
	@echo "#   This is especially recommended for binary packages to ensure reproducibility, and is more" >> .gitignore
	@echo "#   commonly ignored for libraries." >> .gitignore
	@echo "#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control" >> .gitignore
	@echo "#poetry.lock" >> .gitignore
	@echo "" >> .gitignore
	@echo "# pdm" >> .gitignore
	@echo "#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control." >> .gitignore
	@echo "#pdm.lock" >> .gitignore
	@echo "#   pdm stores project-wide configurations in .pdm.toml, but it is recommended to not include it" >> .gitignore
	@echo "#   in version control." >> .gitignore
	@echo "#   https://pdm.fming.dev/#use-with-ide" >> .gitignore
	@echo ".pdm.toml" >> .gitignore
	@echo "" >> .gitignore
	@echo "# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm" >> .gitignore
	@echo "__pypackages__/" >> .gitignore
	@echo "" >> .gitignore
	@echo "# Celery stuff" >> .gitignore
	@echo "celerybeat-schedule" >> .gitignore
	@echo "celerybeat.pid" >> .gitignore
	@echo "" >> .gitignore
	@echo "# SageMath parsed files" >> .gitignore
	@echo "*.sage.py" >> .gitignore
	@echo "" >> .gitignore
	@echo "# Environments" >> .gitignore
	@echo ".env" >> .gitignore
	@echo ".venv" >> .gitignore
	@echo "env/" >> .gitignore
	@echo "venv/" >> .gitignore
	@echo "ENV/" >> .gitignore
	@echo "env.bak/" >> .gitignore
	@echo "venv.bak/" >> .gitignore
	@echo "" >> .gitignore
	@echo "# Spyder project settings" >> .gitignore
	@echo ".spyderproject" >> .gitignore
	@echo ".spyproject" >> .gitignore
	@echo "" >> .gitignore
	@echo "# Rope project settings" >> .gitignore
	@echo ".ropeproject" >> .gitignore
	@echo "" >> .gitignore
	@echo "# mkdocs documentation" >> .gitignore
	@echo "/site" >> .gitignore
	@echo "" >> .gitignore
	@echo "# mypy" >> .gitignore
	@echo ".mypy_cache/" >> .gitignore
	@echo ".dmypy.json" >> .gitignore
	@echo "dmypy.json" >> .gitignore
	@echo "" >> .gitignore
	@echo "# Pyre type checker" >> .gitignore
	@echo ".pyre/" >> .gitignore
	@echo "" >> .gitignore
	@echo "# pytype static type analyzer" >> .gitignore
	@echo ".pytype/" >> .gitignore
	@echo "" >> .gitignore
	@echo "# Cython debug symbols" >> .gitignore
	@echo "cython_debug/" >> .gitignore
	@echo "" >> .gitignore
	@echo "# PyCharm" >> .gitignore
	@echo "#  JetBrains specific template is maintained in a separate JetBrains.gitignore that can" >> .gitignore
	@echo "#  be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore" >> .gitignore
	@echo "#  and can be added to the global gitignore or merged into this file.  For a more nuclear" >> .gitignore
	@echo "#  option (not recommended) you can uncomment the following to ignore the entire idea folder." >> .gitignore
	@echo ".idea/" >> .gitignore

tree:
	@echo "Project structure:"
	@tree .

clean:
	rm -f $(FILES) .gitignore .gitattributes
	rm -rf $(DIRS)
