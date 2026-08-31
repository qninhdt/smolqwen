FROM vllm/vllm-openai:v0.26.0

USER root
WORKDIR /opt/smolqwen

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY serving ./serving

# vLLM and torch come from the version-pinned base image. Install only this
# package so pip cannot silently replace the shared CUDA/torch ABI.
RUN python -m pip install --no-deps . \
    && chmod +x /opt/smolqwen/serving/entrypoint.sh

ENTRYPOINT ["/opt/smolqwen/serving/entrypoint.sh"]
