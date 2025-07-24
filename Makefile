
IMAGE_NAME ?= gcr.io/twistsystems/tsp

.ONESHELL:

.PHONY: build-image push-image build-and-push-image

build-image:
	@if ! git diff --quiet HEAD; then \
		SUFFIX="-local"; \
	fi

	IMAGE_VERSION=$$(git log -1 --format="%at" | xargs -I{} date -d @{} +%Y%m%d.%H%M).$$(git rev-parse --short HEAD)$${SUFFIX}
	@echo Building image ${IMAGE_NAME}:$${IMAGE_VERSION}
	docker build -t ${IMAGE_NAME}:$${IMAGE_VERSION} .

push-image:
	echo docker push ${IMAGE_NAME}

build-and-push-image: build-image push-image

docs:
	cd docs && make html

test:
	pytest

test-image:
	@if ! git diff --quiet HEAD; then \
		SUFFIX="-local"; \
	fi
	IMAGE_VERSION=$$(git log -1 --format="%at" | xargs -I{} date -d @{} +%Y%m%d.%H%M).$$(git rev-parse --short HEAD)${SUFFIX}
	docker run --rm -it ${IMAGE_NAME}:$${IMAGE_VERSION} pytest
