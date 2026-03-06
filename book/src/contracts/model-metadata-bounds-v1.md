# model-metadata-bounds-v1

**Version:** 1.0.0

Valid ranges for model configuration dimensions

## References

- enforce-provable-DbC.md Section 4 Gap 2
- realizar/src/gguf/config.rs ValidatedModelConfig::validate()

## Equations

### config_bounds_check

$$
validate(cfg) = all(field_min <= cfg.field <= field_max for field in required_fields)
$$

**Domain:** $cfg: ModelConfig with hidden_dim, num_layers, num_heads, num_kv_heads, vocab_size, intermediate_dim, context_length, rope_theta, eps$

**Codomain:** $Result<ValidatedModelConfig, ConfigError>$

**Invariants:**

- $hidden_dim \in [1, 65536] and divisible by num_heads$
- $num_kv_heads divides num_heads evenly (GQA ratio)$
- $eps \in [1e-10, 0.01] when configured$
- $rope_theta \in [1.0, 1e8] when configured$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | All required fields within min/max bounds | $\forall field \in required: field.min \leq cfg[field] \leq field.max$ |
| 2 | invariant | hidden_dim divisible by num_heads | $cfg.hidden_dim \% cfg.num_heads == 0$ |
| 3 | invariant | num_kv_heads divides num_heads | $cfg.num_heads \% cfg.num_kv_heads == 0$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-MMB-001 | Bounds rejection | validate() rejects hidden_dim = 0 | Lower bound check missing for hidden_dim |
| FALSIFY-MMB-002 | Bounds rejection | validate() rejects hidden_dim = 131072 | Upper bound check missing for hidden_dim |
| FALSIFY-MMB-003 | GQA divisibility | validate() rejects num_heads=7, num_kv_heads=3 | Missing GQA divisibility check |

