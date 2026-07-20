FROM python:3.11-slim

# Quarto CLI
ARG QUARTO_VERSION=1.7.33
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gdebi-core \
    && curl -LO https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb \
    && gdebi --non-interactive quarto-${QUARTO_VERSION}-linux-amd64.deb \
    && rm quarto-${QUARTO_VERSION}-linux-amd64.deb \
    && apt-get purge -y gdebi-core && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /book

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["quarto", "render"]
