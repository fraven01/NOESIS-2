# RAG Audit Notes

## 1) Vector/embedding setup
- docs/rag/schema.sql: `CREATE TABLE embeddings` —
  ```sql
  CREATE TABLE IF NOT EXISTS {{SCHEMA_NAME}}.embeddings (
      id UUID PRIMARY KEY,
      chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
      embedding vector({{VECTOR_DIM}}) NOT NULL
  );
  CREATE INDEX IF NOT EXISTS embeddings_embedding_hnsw
      ON {{SCHEMA_NAME}}.embeddings USING hnsw (embedding vector_cosine_ops)
      WITH (m = 32, ef_construction = 200);
  ```
- ai_core/management/commands/rebuild_rag_index.py: `handle` —
  ```python
        index_kind = str(getattr(settings, "RAG_INDEX_KIND", "HNSW")).upper()
        hnsw_m = int(getattr(settings, "RAG_HNSW_M", 32))
        hnsw_ef = int(getattr(settings, "RAG_HNSW_EF_CONSTRUCTION", 200))
        ivf_lists = int(getattr(settings, "RAG_IVF_LISTS", 2048))
        row = self._rebuild_index(
            schema_name, index_kind, hnsw_m, hnsw_ef, ivf_lists, expected_index,
  ```
- ai_core/rag/vector_client.py: `hybrid_search` —
  ```python
        index_kind = str(_get_setting("RAG_INDEX_KIND", "HNSW")).upper()
        ef_search = int(_get_setting("RAG_HNSW_EF_SEARCH", 80))
        probes = int(_get_setting("RAG_IVF_PROBES", 64))
  ```
- ai_core/rag/vector_client.py: `_get_distance_operator` —
  ```python
    def _get_distance_operator(self, conn, index_kind: str) -> str:
        key = index_kind.upper()
        with conn.cursor() as cur:
            operator = resolve_distance_operator(cur, key)
        if operator is None:
            raise RuntimeError(
                "No compatible pgvector operator class available for queries."
            )
  ```

## 2) Chunk/ingestion contracts
- ai_core/rag/ingestion_contracts.py: `ChunkMeta` —
  ```python
  class ChunkMeta(BaseModel):
      tenant_id: str
      case_id: str
      source: str
      hash: str
      external_id: str
      content_hash: str
      embedding_profile: str | None = None
      vector_space_id: str | None = None
      process: str | None = None
      doc_class: str | None = None
      parent_ids: list[str] | None = None
  ```
- ai_core/rag/schemas.py: `Chunk` —
  ```python
  @dataclass
  class Chunk:
      content: str
      meta: Dict[str, Any]
      embedding: Optional[List[float]] = None
      parents: Optional[Dict[str, Dict[str, Any]]] = None
  ```
- ai_core/tasks.py: `chunk` —
  ```python
      target_tokens = int(getattr(settings, "RAG_CHUNK_TARGET_TOKENS", 450))
      if profile_limit is not None:
          hard_limit = profile_limit
          target_tokens = min(target_tokens, hard_limit)
      else:
          hard_limit = max(target_tokens, fallback_limit)
      overlap_tokens = _resolve_overlap_tokens(
          text, meta, target_tokens=target_tokens, hard_limit=hard_limit,
      )
      chunk_meta = {
          "tenant_id": meta["tenant_id"],
          "case_id": meta.get("case_id"),
          "source": text_path,
          "hash": content_hash,
          "external_id": external_id,
          "content_hash": content_hash,
          "parent_ids": parent_ids,
      }
  ```
- ai_core/tasks.py: `_resolve_overlap_tokens` —
  ```python
  configured_limit = getattr(settings, "RAG_CHUNK_OVERLAP_TOKENS", None)
  if configured_value is not None and configured_value <= 0:
      return 0
  overlap = int(round(target_tokens * ratio))
  overlap = min(overlap, max(0, target_tokens - 1))
  return max(0, min(overlap, hard_limit))
  ```

## 3) Embedding profiles and routing
- noesis2/settings/base.py: `RAG_EMBEDDING_PROFILES` —
  ```python
  RAG_EMBEDDING_PROFILES = {
      "standard": {
          "model": EMBEDDINGS_MODEL_PRIMARY,
          "dimension": DEFAULT_EMBEDDING_DIMENSION,
          "vector_space": "global",
          "chunk_hard_limit": 512,
      },
      "demo": {
          "model": DEMO_EMBEDDINGS_MODEL,
          "dimension": DEMO_EMBEDDING_DIMENSION,
          "vector_space": "demo",
          "chunk_hard_limit": 1024,
      },
  }
  ```
- ai_core/rag/embedding_config.py: `EmbeddingProfileConfig` & coercion —
  ```python
  @dataclass(frozen=True, slots=True)
  class EmbeddingProfileConfig:
      id: str
      model: str
      dimension: int
      vector_space: str
      chunk_hard_limit: int
  def _coerce_chunk_limit(raw: object, *, context: str) -> int:
      if raw is None:
          return _DEFAULT_PROFILE_CHUNK_LIMIT

      try:
          limit = int(raw)  # type: ignore[arg-type]
      except (TypeError, ValueError) as exc:
          raise EmbeddingConfigurationError(
              _format_error(
                  EmbeddingConfigErrorCode.PROFILE_CHUNK_LIMIT_INVALID,
                  f"{context} chunk_hard_limit must be a positive integer",
              )
          ) from exc

      if limit <= 0:
          raise EmbeddingConfigurationError(
              _format_error(
                  EmbeddingConfigErrorCode.PROFILE_CHUNK_LIMIT_INVALID,
                  f"{context} chunk_hard_limit must be positive",
              )
          )

      return limit
  ```
- ai_core/rag/embedding_config.py: `get_embedding_profile` —
  ```python
  def get_embedding_profile(profile_id: str) -> EmbeddingProfileConfig:
      configuration = get_embedding_configuration()
      config = configuration.embedding_profiles
      try:
          return config[profile_id]
      except KeyError as exc:
          default_profile_id = str(getattr(settings, "RAG_DEFAULT_EMBEDDING_PROFILE", "")).strip()
          if default_profile_id and default_profile_id in config:
              return config[default_profile_id]
          fallback_profile = next(iter(config.values()), None)
          return EmbeddingProfileConfig(
              id=fallback_profile.id,
              model=fallback_profile.model,
              dimension=fallback_profile.dimension,
              vector_space=fallback_profile.vector_space,
              chunk_hard_limit=_DEFAULT_PROFILE_CHUNK_LIMIT,
          )
  ```
- ai_core/rag/profile_resolver.py: `resolve_embedding_profile` —
  ```python
  profile_id = get_routing_table().resolve(
      tenant=tenant,
      process=sanitized_process,
      doc_class=sanitized_doc_class,
  )
  configuration = get_embedding_configuration().embedding_profiles
  if profile_id not in configuration:
      raise ProfileResolverError(
          ProfileResolverErrorCode.UNKNOWN_PROFILE,
          f"Resolved profile '{profile_id}' is not configured",
      )
  ```
- ai_core/rag/routing_rules.py: `get_routing_table` —
  ```python
  default_profile = str(raw_config["default_profile"]).strip()
  if not default_profile:
      raise RoutingConfigurationError(
          _format_error(
              RoutingErrorCode.DEFAULT_PROFILE_EMPTY,
              "default_profile cannot be empty",
          )
      )
  rules = _build_rules(raw_rules)
  table = RoutingTable(default_profile=default_profile, rules=rules)
  _ensure_profiles_exist(table=table, available_profiles=config.embedding_profiles.keys())
  ```
- ai_core/rag/vector_client.py: ingestion summary —
  ```python
      metadata = doc.get("metadata", {})
      embedding_profile = metadata.get("embedding_profile")
      if embedding_profile:
          doc_payload["embedding_profile"] = embedding_profile
      vector_space_id = metadata.get("vector_space_id")
      if vector_space_id:
          doc_payload["vector_space_id"] = vector_space_id
  ```

## 4) Document storage
- ai_core/ingestion.py: `_upload_dir` & `_meta_store_path` —
  ```python
  def _upload_dir(tenant: str, case: str) -> Path:
      return (
          object_store.BASE_PATH
          / object_store.sanitize_identifier(tenant)
          / object_store.sanitize_identifier(case)
          / "uploads"
      )
  def _meta_store_path(tenant: str, case: str, document_id: str) -> str:
      return "/".join(
          (
              object_store.sanitize_identifier(tenant),
              object_store.sanitize_identifier(case),
              "uploads",
              f"{document_id}.meta.json",
          )
      )
  ```
- ai_core/tasks.py: `_build_path` —
  ```python
  def _build_path(meta: Dict[str, str], *parts: str) -> str:
      tenant = object_store.sanitize_identifier(meta["tenant_id"])
      case = object_store.sanitize_identifier(meta["case_id"])
      return "/".join([tenant, case, *parts])
  ```
- ai_core/tasks.py: chunk output packaging —
  ```python
  payload = {"chunks": chunks, "parents": limited_parents}
  out_path = _build_path(meta, "embeddings", "chunks.json")
  object_store.write_json(out_path, payload)
  ```
- ai_core/rag/vector_client.py: `_ensure_documents` —
  ```python
      metadata_dict = dict(doc.get("metadata", {}))
      metadata_dict.setdefault("hash", content_hash)
      if self._near_duplicate_strategy == "replace":
          metadata_dict["near_duplicate_of"] = str(near_duplicate_details.get("external_id"))
      if isinstance(parents_map, Mapping) and parents_map:
          metadata_dict["parent_nodes"] = limit_parent_payload(parents_map)
      cur.execute(
          """
          INSERT INTO documents (id, tenant_id, external_id, source, hash, metadata)
          VALUES (%s, %s, %s, %s, %s, %s)
          ON CONFLICT (tenant_id, external_id) DO UPDATE
              SET source = EXCLUDED.source,
                  hash = EXCLUDED.hash,
                  metadata = EXCLUDED.metadata,
                  deleted_at = NULL
          RETURNING id, hash
          """,
          (
              document_id,
              str(tenant_uuid),
              external_id,
              doc["source"],
              storage_hash,
              metadata,
          ),
      )
      )
  ```

## 5) Near-duplicate detection
- ai_core/rag/vector_client.py: constructor settings —
  ```python
  near_strategy = str(_get_setting("RAG_NEAR_DUPLICATE_STRATEGY", "skip")).lower()
  threshold_setting = _get_setting("RAG_NEAR_DUPLICATE_THRESHOLD", 0.97)
  probe_setting = _get_setting("RAG_NEAR_DUPLICATE_PROBE_LIMIT", 8)
  self._require_unit_norm_for_l2 = _get_bool_setting(
      "RAG_NEAR_DUPLICATE_REQUIRE_UNIT_NORM", False
  )
  ```
- ai_core/rag/vector_client.py: `_get_distance_operator` & support —
  ```python
  operator = resolve_distance_operator(cur, key)
  if operator == "<=>":
      supported = True
  elif operator == "<->":
      supported = True
      if self._require_unit_norm_for_l2:
          logger.info(
              "ingestion.doc.near_duplicate_l2_enabled",
              extra={"index_kind": key, "requires_unit_normalised": True},
          )
      else:
          logger.info(
              "ingestion.doc.near_duplicate_l2_distance_mode",
              extra={"index_kind": key, "requires_unit_normalised": False},
          )
  else:
      logger.info(
          "ingestion.doc.near_duplicate_operator_unsupported",
          extra={"index_kind": key, "operator": operator},
      )
  ```
- ai_core/rag/vector_client.py: `_find_near_duplicate` —
  ```python
  index_kind = str(_get_setting("RAG_INDEX_KIND", "HNSW")).upper()
  if operator not in {"<=>", "<->"}:
      self._disable_near_duplicate_for_operator(
          index_kind=index_kind,
          operator=operator,
          tenant_uuid=tenant_uuid,
      )
      return None
  if operator == "<->":
      if self._require_unit_norm_for_l2 and not embedding_is_unit_normalised:
          self._near_duplicate_enabled = False
          return None
      else:
          distance_cutoff = math.sqrt(max(0.0, 2.0 * (1.0 - self._near_duplicate_threshold)))
  query = sql.SQL(
      """
      WITH base AS (
          SELECT
              d.id,
              d.external_id,
              {sim} AS similarity,
              {distance} AS chunk_distance
          FROM documents d
          JOIN chunks c ON c.document_id = d.id
          JOIN embeddings e ON e.chunk_id = c.id
          WHERE d.tenant_id = %s
            AND d.deleted_at IS NULL
            AND d.external_id <> %s
          ORDER BY chunk_distance ASC
          LIMIT %s
      )
      SELECT id, external_id, similarity
      FROM (
          SELECT
              id,
              external_id,
              similarity,
              ROW_NUMBER() OVER (
                  PARTITION BY id
                  ORDER BY chunk_distance ASC
              ) AS chunk_rank
          FROM base
      ) AS ranked
      WHERE chunk_rank = 1
      ORDER BY similarity {global_order}
      LIMIT %s
      """
  )
  ```

## 6) Retrieval fusion & MMR
- ai_core/rag/vector_client.py: fusion scoring —
  ```python
      has_vector_signal = bool(vector_rows) and (query_vec is not None) and (not query_embedding_empty)
      for entry in candidates.values():
          vector_preview = float(entry.get("vscore", 0.0))
          lexical_preview = float(entry.get("lscore", 0.0))
          if query_embedding_empty:
              fused_preview = max(0.0, min(1.0, lexical_preview))
          elif has_vector_signal:
              fused_preview = max(0.0, min(1.0, alpha_value * vector_preview + (1.0 - alpha_value) * lexical_preview))
          else:
              fused_preview = max(0.0, min(1.0, lexical_preview))
          meta["fused"] = fused
          meta["score"] = fused
  ```
- ai_core/rag/vector_store.py: hybrid parameter resolution —
  ```python
  alpha_default = float(get_limit_setting("RAG_HYBRID_ALPHA", 0.7))
  min_sim_default = float(get_limit_setting("RAG_MIN_SIM", 0.15))
  trgm_default = float(get_limit_setting("RAG_TRGM_LIMIT", 0.30))
  alpha_value, alpha_source = clamp_fraction(alpha, default=alpha_default, return_source=True)
  min_sim_value, min_sim_source = clamp_fraction(min_sim, default=min_sim_default, return_source=True)
  trgm_value, trgm_source = clamp_fraction(trgm_requested, default=trgm_default, return_source=True)
  ```
- ai_core/nodes/retrieve.py: `_tokenise` & MMR diversification —
  ```python
  _TOKEN_PATTERN = re.compile(r"[\w\u00C0-\u024F]+", re.UNICODE)
  def _tokenise(text: str | None) -> set[str]:
      if not isinstance(text, str) or not text:
          return set()
      tokens = {match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)}
      return tokens
  def _apply_diversification(matches: list[Dict[str, Any]], *, top_k: int, strength: float) -> list[Dict[str, Any]]:
      lambda_param = 1.0 - normalised_strength
      relevance_scores = [float(match.get("score", 0.0)) for match in matches]
      token_sets = [_tokenise(match.get("text") or match.get("content") or "") for match in matches]
      mmr_score = lambda_param * relevance_scores[idx] - ((1.0 - lambda_param) * diversity_penalty)
  ```

## 7) Settings/constants
- noesis2/settings/base.py: RAG defaults —
  ```python
  RAG_INDEX_KIND = env("RAG_INDEX_KIND", default="HNSW").upper()
  RAG_HNSW_M = env.int("RAG_HNSW_M", default=32)
  RAG_HNSW_EF_SEARCH = env.int("RAG_HNSW_EF_SEARCH", default=80)
  RAG_IVF_LISTS = env.int("RAG_IVF_LISTS", default=2048)
  RAG_IVF_PROBES = env.int("RAG_IVF_PROBES", default=64)
  RAG_MIN_SIM = env.float("RAG_MIN_SIM", default=0.15)
  RAG_TRGM_LIMIT = env.float("RAG_TRGM_LIMIT", default=0.1)
  RAG_HYBRID_ALPHA = env.float("RAG_HYBRID_ALPHA", default=0.7)
  RAG_MAX_CANDIDATES = env.int("RAG_MAX_CANDIDATES", default=200)
  RAG_CHUNK_TARGET_TOKENS = env.int("RAG_CHUNK_TARGET_TOKENS", default=450)
  RAG_CHUNK_OVERLAP_TOKENS = env.int("RAG_CHUNK_OVERLAP_TOKENS", default=80)
  ```
- ai_core/tasks.py: RAG parent capture controls —
  ```python
  def _resolve_parent_capture_max_depth() -> int:
      value = getattr(settings, "RAG_PARENT_CAPTURE_MAX_DEPTH", 0)
      depth = int(value) if value else 0
      return depth if depth > 0 else 0
  def _resolve_parent_capture_max_bytes() -> int:
      value = getattr(settings, "RAG_PARENT_CAPTURE_MAX_BYTES", 0)
      byte_limit = int(value)
      return byte_limit if byte_limit > 0 else 0
  ```
- ai_core/rag/parents.py: payload cap —
  ```python
  value = getattr(settings, "RAG_PARENT_MAX_BYTES", 0)
  cap = int(value)
  return cap if cap > 0 else 0
  ```
- ai_core/tasks.py: near-duplicate normalisation flag —
  ```python
  env_value = os.getenv("RAG_NEAR_DUPLICATE_REQUIRE_UNIT_NORM")
  try:
      return bool(getattr(settings, "RAG_NEAR_DUPLICATE_REQUIRE_UNIT_NORM"))
  except Exception:
      return False
  ```
- ai_core/rag/vector_client.py: timeout & retry config —
  ```python
        env_timeout = int(os.getenv("RAG_STATEMENT_TIMEOUT_MS", str(FALLBACK_STATEMENT_TIMEOUT_MS)))
        env_retries = int(os.getenv("RAG_RETRY_ATTEMPTS", str(FALLBACK_RETRY_ATTEMPTS)))
        env_retry_delay = int(os.getenv("RAG_RETRY_BASE_DELAY_MS", str(FALLBACK_RETRY_BASE_DELAY_MS)))
        self._statement_timeout_ms = timeout_value
        self._retries = max(1, retries_value)
        self._retry_base_delay = max(0, retry_delay_value) / 1000.0
  ```
- ai_core/rag/vector_client.py: near-duplicate probes —
  ```python
  probe_setting = _get_setting("RAG_NEAR_DUPLICATE_PROBE_LIMIT", 8)
  self._near_duplicate_probe_limit = max(1, probe_limit)
  self._near_duplicate_enabled = (
      near_strategy in {"skip", "replace"} and self._near_duplicate_threshold > 0.0
  )
  ```
