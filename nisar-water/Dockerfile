FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY checkpoints ./checkpoints
COPY benchmarks ./benchmarks
RUN mkdir /results && useradd --create-home worker && chown worker:worker /results
USER worker
EXPOSE 8010
CMD ["nisar-water", "serve", "--root", "/results", "--host", "0.0.0.0", "--port", "8010", "--assessment", "/app/benchmarks/round5/assessment.json"]
