FROM python:3.13.3-slim-bookworm AS base

ARG USERNAME="twist"
ARG USER_UID="1000"
ARG USER_GID=$USER_UID

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBUG=False

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir /app \
    && chown $USERNAME:$USERNAME /app

COPY --chown=$USERNAME:$USERNAME ./requirements.txt /home/$USERNAME/

RUN apt-get -q update \
    && apt-get install -yq --no-install-recommends build-essential \
    && pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r /home/$USERNAME/requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

######################################################################
# Development
######################################################################
FROM base AS dev

# Install dev dependencies

WORKDIR /app

######################################################################
# Final
######################################################################
FROM base AS final

COPY --chown=$USERNAME:$USERNAME . /app

WORKDIR /app

# Set the entrypoint or command to run. For example, for a gunicorn server:
# CMD gunicorn siteconfig.wsgi:application \
#     -b :$PORT \
#     -w $GUNICORN_WORKERS \
#     -t $GUNICORN_TIMEOUT \
#     --forwarded-allow-ips="*" \
#     --log-level=debug
