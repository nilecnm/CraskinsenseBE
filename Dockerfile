FROM python:3.10 as base

# Setup env
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1

FROM base AS python-deps

# Install pipenv and compilation dependencies
RUN pip install pipenv

# Install python dependencies in /.venv
COPY Pipfile .
COPY Pipfile.lock .
RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy

RUN apt-get update && apt-get install -y --no-install-recommends gcc ffmpeg libsm6 libxext6 python3-opencv libgl1 libgl1-mesa-dev

FROM base AS runtime

COPY --from=python-deps /.venv /.venv
ENV PATH="/.venv/bin:$PATH"

# Create and switch to a new user
RUN useradd --create-home appuser
WORKDIR /home/appuser
RUN pip install "fastapi[standard]"
USER appuser

# Install application into container
COPY . .

# Run the application
CMD ["fastapi", "run", "main.py", "--port", "80"]