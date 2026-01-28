# Makefile variables set automatically
plugin_id=`cat plugin.json | python -c "import sys, json; print(str(json.load(sys.stdin)['id']).replace('/',''))"`
plugin_version=`cat plugin.json | python -c "import sys, json; print(str(json.load(sys.stdin)['version']).replace('/',''))"`
archive_file_name="dss-plugin-${plugin_id}-${plugin_version}.zip"
remote_url=`git config --get remote.origin.url`
last_commit_id=`git rev-parse HEAD`


plugin:
	@echo "[START] Archiving plugin to dist/ folder..."
	@cat plugin.json | json_pp > /dev/null
	@rm -rf dist
	@mkdir dist
	@echo "{\"remote_url\":\"${remote_url}\",\"last_commit_id\":\"${last_commit_id}\"}" > release_info.json
	@git archive -v -9 --format zip -o dist/${archive_file_name} HEAD
	@zip --delete dist/${archive_file_name} "tests/*"
	@zip -u dist/${archive_file_name} release_info.json
	@rm release_info.json
	@echo "[SUCCESS] Archiving plugin to dist/ folder: Done!"

unit-tests:
	@echo "Running unit tests..."
	@( \
		PYTHON_VERSION=`python3 -c "import sys; print('PYTHON{}{}'.format(sys.version_info.major, sys.version_info.minor))"`; \
		PYTHON_VERSION_IS_CORRECT=`cat code-env/python/desc.json | python3 -c "import sys, json; print('$$PYTHON_VERSION' in json.load(sys.stdin)['acceptedPythonInterpreters']);"`; \
		if [ $$PYTHON_VERSION_IS_CORRECT == "False" ]; then echo "Python version $$PYTHON_VERSION is not in acceptedPythonInterpreters"; exit 1; else echo "Python version $$PYTHON_VERSION is in acceptedPythonInterpreters"; fi; \
	)
	@( \
		rm -rf ./env/; \
		python3 -m venv env/; \
		source env/bin/activate; \
		pip install --upgrade pip;\
		pip install --no-cache-dir -r tests/python/unit/requirements.txt; \
		pip install --no-cache-dir -r code-env/python/spec/requirements.txt; \
		export PYTHONPATH="$(PYTHONPATH):$(PWD)/python-lib"; \
		export DICTIONARY_FOLDER_PATH="$(PWD)/resource/dictionaries"; \
		export STOPWORDS_FOLDER_PATH="$(PWD)/resource/stopwords"; \
		pytest tests/python/unit --alluredir=tests/allure_report || ret=$$?; exit $$ret \
	)

integration-tests:
	@echo "Running integration tests..."
	@( \
		rm -rf ./env/; \
		python3 -m venv env/; \
		source env/bin/activate; \
		pip3 install --upgrade pip;\
		pip install --no-cache-dir -r tests/python/integration/requirements.txt; \
		pytest tests/python/integration --alluredir=tests/allure_report || ret=$$?; exit $$ret \
	)

tests: unit-tests integration-tests

dist-clean:
	rm -rf dist

# Docker-based unit tests for Linux environment
# Uses --platform linux/amd64 to ensure consistent behavior on Apple Silicon

DOCKER_IMAGE_NAME=nlp-preparation-test
DOCKER_PLATFORM=linux/amd64

define run-docker-test
	@echo "[START] Running unit tests in Docker with Python $(1)..."
	@docker build \
		--platform $(DOCKER_PLATFORM) \
		--build-arg PYTHON_VERSION=$(1) \
		-t $(DOCKER_IMAGE_NAME):py$(1) \
		-f tests/docker/Dockerfile \
		. && \
	docker run --rm --platform $(DOCKER_PLATFORM) $(DOCKER_IMAGE_NAME):py$(1)
	@echo "[DONE] Python $(1) tests completed"
endef

docker-test-py36:
	$(call run-docker-test,3.6)

docker-test-py37:
	$(call run-docker-test,3.7)

docker-test-py38:
	$(call run-docker-test,3.8)

docker-test-py39:
	$(call run-docker-test,3.9)

docker-test-py310:
	$(call run-docker-test,3.10)

docker-test-py311:
	$(call run-docker-test,3.11)

docker-test-py312:
	$(call run-docker-test,3.12)

docker-test-py313:
	$(call run-docker-test,3.13)

docker-test-all: docker-test-py36 docker-test-py37 docker-test-py38 docker-test-py39 docker-test-py310 docker-test-py311 docker-test-py312 docker-test-py313
	@echo "[SUCCESS] All Docker tests completed"

docker-clean:
	@echo "Removing Docker test images..."
	@docker rmi -f $(DOCKER_IMAGE_NAME):py3.6 $(DOCKER_IMAGE_NAME):py3.7 $(DOCKER_IMAGE_NAME):py3.8 $(DOCKER_IMAGE_NAME):py3.9 $(DOCKER_IMAGE_NAME):py3.10 $(DOCKER_IMAGE_NAME):py3.11 $(DOCKER_IMAGE_NAME):py3.12 $(DOCKER_IMAGE_NAME):py3.13 2>/dev/null || true
	@echo "Docker images cleaned"
