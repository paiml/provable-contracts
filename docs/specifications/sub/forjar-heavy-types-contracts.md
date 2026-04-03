# Section 33: Forjar Heavy Types Contracts

> Parent: [pv-spec.md](../pv-spec.md) §33

## 33.1 Motivation

Forjar is a Rust-native IaC tool with 236+ types across 50+ modules. It currently has 5 contracts
(blake3-state, dag-ordering, execution-safety, recipe-determinism, codegen-dispatch) covering the
execution kernel. The **store module** (content-addressed derivation, purity classification, FAR
archive), **task/pipeline system**, **event rulebooks**, **Copia delta sync**, and **sandbox
isolation** are entirely uncontracted despite being correctness-critical.

Forjar's design philosophy is **verifiable determinism** — every operation is content-addressed,
every execution order is deterministic, every state change is auditable. Contracts must enforce
these guarantees formally.

## 33.2 Domain Inventory

| ID | Domain | Priority | Contracts | Surface |
|----|--------|----------|-----------|---------|
| FJ-HVY-1 | Store / Content-Addressed | P0 | store-cas-v1 | Core storage invariant |
| FJ-HVY-2 | OCI / Container Images | P0 | oci-manifest-v1 | Build reproducibility |
| FJ-HVY-3 | Task / Pipeline | P0 | task-pipeline-v1 | Execution correctness |
| FJ-HVY-4 | Event / Rulebook | P1 | event-rulebook-v1 | Automation safety |
| FJ-HVY-5 | Plugin Lifecycle | P1 | plugin-lifecycle-v1 | Extension safety |
| FJ-HVY-6 | Secret Provider | P1 | secret-provider-v1 | Credential safety |
| FJ-HVY-7 | Copia Delta Sync | P0 | copia-delta-v1 | Data integrity |
| FJ-HVY-8 | Sandbox Isolation | P1 | sandbox-isolation-v1 | Security boundary |

## 33.3 Store / Content-Addressed Storage (store-cas-v1)

### Type Surface

```rust
// Core CAS types
struct StorePath { hash: Blake3Hash, name: String }
struct Derivation { inputs: Vec<StorePath>, builder: String, env: BTreeMap<String, String> }
struct StoreEntry { path: StorePath, references: Vec<StorePath>, registration_time: u64 }
enum PurityLevel { Pure(0), NetworkAccess(1), Impure(2), Unrestricted(3) }
```

### Equations

**derivation_determinism**: Same inputs always produce same store path.

```
derive: Derivation -> StorePath
  derive(d₁) = derive(d₂) when d₁.inputs = d₂.inputs ∧ d₁.builder = d₂.builder ∧ d₁.env = d₂.env
  StorePath.hash = BLAKE3(canonical_serialize(derivation))
```

**closure_completeness**: A store path's closure contains all transitive references.

```
closure: StorePath -> Set<StorePath>
  ∀ ref in entry.references: ref ∈ closure(entry)
  ∀ ref in entry.references: closure(ref) ⊆ closure(entry)
  Closure is the least fixed point of the reference relation
```

**purity_monotonicity**: Higher purity level is more restrictive.

```
purity: Derivation -> PurityLevel
  Pure(0): no network, no impure inputs, hermetic sandbox
  NetworkAccess(1): network allowed, inputs still pure
  Impure(2): system state may affect output
  Unrestricted(3): no guarantees
  purity(d) = max(purity(input) for input in d.inputs) ∪ d.own_purity
```

**far_archive_roundtrip**: FAR (Forjar Archive) pack/unpack is identity.

```
far: Directory -> bool
  unpack(pack(dir)) = dir
  ∀ file in dir: hash(file_before) = hash(file_after)
  Preserves: permissions, ownership, symlinks, timestamps
```

**gc_safety**: Garbage collection never removes live store paths.

```
gc: (Store, RootSet) -> Store
  ∀ path in closure(root) for root in RootSet: path ∈ gc(store)
  Only removes: paths not reachable from any root
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| determinism | Derivation deterministic | d₁ = d₂ → derive(d₁) = derive(d₂) |
| completeness | Closure is transitive | ref ∈ closure(x) ∧ ref' ∈ closure(ref) → ref' ∈ closure(x) |
| monotonicity | Purity propagates upward | purity(d) ≥ max(purity(input_i)) |
| roundtrip | FAR identity | unpack(pack(dir)) = dir |
| conservation | GC preserves live paths | ∀ live path: path ∈ gc(store) |
| precondition | Store path hash valid | StorePath.hash = BLAKE3(canonical_serialize(derivation)) |

### Falsification Tests

- **FALSIFY-CAS-001**: Two derivations with same inputs produce identical store path
- **FALSIFY-CAS-002**: Two derivations with different env produce different store path
- **FALSIFY-CAS-003**: Closure of a path with no references is {self}
- **FALSIFY-CAS-004**: Closure of A→B→C includes all three paths
- **FALSIFY-CAS-005**: FAR roundtrip preserves file permissions (0o755 vs 0o644)
- **FALSIFY-CAS-006**: FAR roundtrip preserves symlink targets
- **FALSIFY-CAS-007**: GC with empty root set removes all paths
- **FALSIFY-CAS-008**: GC with full root set removes nothing
- **FALSIFY-CAS-009**: Pure derivation with impure input escalates to impure

## 33.4 OCI / Container Images (oci-manifest-v1)

### Type Surface

```rust
struct OciManifest { schema_version: u8, media_type: String, config: Descriptor, layers: Vec<Descriptor> }
struct Descriptor { media_type: String, digest: String, size: u64 }
struct ImageConfig { architecture: String, os: String, rootfs: RootFs, history: Vec<History> }
struct LayerCache { digest_to_path: HashMap<String, PathBuf> }
```

### Equations

**manifest_digest_consistency**: Manifest digest matches content hash.

```
digest: OciManifest -> String
  digest = "sha256:" ++ hex(SHA256(canonical_json(manifest)))
  len(digest) = 7 + 64  (prefix + hex SHA256)
```

**layer_ordering**: Layers are applied in order, later layers override earlier.

```
apply_layers: Vec<Layer> -> Filesystem
  apply(layers[0..n]) = apply(layers[0..n-1]) ∪ layers[n]
  Where ∪ means: files in layers[n] override files in layers[0..n-1]
```

**layer_cache_hit**: Unchanged layers reuse cached digests.

```
cache: (LayerCache, Layer) -> (Descriptor, bool)
  digest(layer) ∈ cache → (cached_descriptor, hit=true)
  digest(layer) ∉ cache → (build_descriptor(layer), hit=false)
  cache_hit → no rebuild, no re-upload
```

**reproducible_build**: Same Dockerfile + context → same image digest.

```
build: (Dockerfile, Context) -> OciManifest
  build(df, ctx₁) = build(df, ctx₂) when file_hashes(ctx₁) = file_hashes(ctx₂)
  Requires: deterministic timestamps, sorted directory entries
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| determinism | Manifest digest deterministic | same content → same digest |
| ordering | Layer application order | later layers override earlier |
| equivalence | Cache hit avoids rebuild | cached layer = built layer (by digest) |
| determinism | Reproducible build | same inputs → same manifest |
| bound | Digest format | len("sha256:") + 64 hex chars |

### Falsification Tests

- **FALSIFY-OCI-001**: Manifest with reordered JSON keys produces same digest (canonical JSON)
- **FALSIFY-OCI-002**: Image with 3 layers, file in layer 3 overrides layer 1
- **FALSIFY-OCI-003**: Cached layer digest matches freshly built layer digest
- **FALSIFY-OCI-004**: Two builds from identical context produce identical manifest digest
- **FALSIFY-OCI-005**: Empty layer produces valid descriptor with size=0

## 33.5 Task / Pipeline (task-pipeline-v1)

### Type Surface

```rust
enum TaskMode { Batch, Pipeline, Service, Dispatch }
struct PipelineStage { name: String, steps: Vec<Step>, depends_on: Vec<String>, quality_gate: Option<QualityGate> }
struct QualityGate { metric: String, threshold: f64, operator: CompareOp }
enum TaskStatus { Pending, Running, Succeeded, Failed, Skipped, Cancelled }
struct HealthCheck { command: String, interval: Duration, retries: u32, timeout: Duration }
```

### Equations

**pipeline_dag_execution**: Stages execute in dependency order.

```
execute: Vec<PipelineStage> -> Vec<(StageId, TaskStatus)>
  ∀ stage S with depends_on = [A, B]:
    start_time(S) > end_time(A) ∧ start_time(S) > end_time(B)
  Independent stages may execute in parallel
```

**quality_gate_enforcement**: Failed quality gate blocks downstream stages.

```
gate: (QualityGate, MetricValue) -> bool
  gate.operator.eval(value, gate.threshold) = true → stage passes
  gate.operator.eval(value, gate.threshold) = false → stage FAILS, all dependents SKIPPED
```

**task_status_terminal**: Terminal states are final.

```
terminal: TaskStatus -> bool
  Succeeded → terminal (no further transitions)
  Failed    → terminal
  Skipped   → terminal
  Cancelled → terminal
  Pending   → non-terminal
  Running   → non-terminal
```

**health_check_retry**: Health check retries with backoff before declaring failure.

```
health: (HealthCheck, Attempts) -> TaskStatus
  attempts ≤ retries → retry after interval
  attempts > retries → Failed
  Total wall clock ≤ retries * (timeout + interval)
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| ordering | DAG execution order | depends_on satisfied before start |
| postcondition | Quality gate blocks dependents | gate fails → all dependents skipped |
| state_machine | Terminal states are final | no transition from Succeeded/Failed/Skipped/Cancelled |
| termination | Health check terminates | attempts ≤ retries + 1 |
| bound | Wall clock bounded | total ≤ retries × (timeout + interval) |

### Falsification Tests

- **FALSIFY-PIPE-001**: Stage with failed dependency is skipped, not run
- **FALSIFY-PIPE-002**: Quality gate with threshold 0.8 blocks stage scoring 0.7
- **FALSIFY-PIPE-003**: Quality gate with threshold 0.8 passes stage scoring 0.9
- **FALSIFY-PIPE-004**: Task in Succeeded state cannot transition to Failed
- **FALSIFY-PIPE-005**: Health check with 0 retries fails immediately on first failure
- **FALSIFY-PIPE-006**: Pipeline with diamond dependency (A→B, A→C, B→D, C→D) executes D last
- **FALSIFY-PIPE-007**: Independent stages B and C run in parallel (wall clock < sequential)

## 33.6 Event / Rulebook (event-rulebook-v1)

### Type Surface

```rust
enum TriggerKind { File, Process, Cron, Webhook, Metric, Manual }
struct Rule { trigger: TriggerKind, condition: Condition, actions: Vec<Action>, cooldown: Duration }
struct Condition { expression: String }  // CEL or similar
struct Action { resource: String, operation: Operation }
enum Operation { Apply, Destroy, Restart, Notify }
```

### Equations

**trigger_dispatch_completeness**: Every trigger kind has a handler.

```
dispatch: TriggerKind -> Handler
  ∀ kind in TriggerKind: handler(kind) exists ∧ handler(kind) ≠ no-op
```

**cooldown_deduplication**: Events within cooldown window are deduplicated.

```
dedup: (Rule, Event, LastFired) -> bool
  now - last_fired < cooldown → event suppressed
  now - last_fired >= cooldown → event fires
```

**action_ordering**: Actions within a rule execute sequentially.

```
actions: Rule -> Vec<ActionResult>
  ∀ i < j: end_time(actions[i]) < start_time(actions[j])
  action[i] fails → actions[i+1..] skipped (fail-fast)
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| completeness | All triggers handled | ∀ kind: handler(kind) exists |
| idempotency | Cooldown dedup | events within cooldown → 1 execution |
| ordering | Sequential actions | action[i] completes before action[i+1] starts |
| soundness | Fail-fast | action failure → remaining skipped |

### Falsification Tests

- **FALSIFY-EVT-001**: File trigger on modified path fires exactly once
- **FALSIFY-EVT-002**: Two events within 10s cooldown fire only once
- **FALSIFY-EVT-003**: Event after cooldown expires fires normally
- **FALSIFY-EVT-004**: Second action not executed when first action fails
- **FALSIFY-EVT-005**: Manual trigger with no condition always fires

## 33.7 Plugin Lifecycle (plugin-lifecycle-v1)

### Type Surface

```rust
enum PluginState { Discovered, Loaded, Initialized, Running, Stopped, Error }
struct PluginManifest { name: String, version: Version, permissions: Vec<Permission> }
enum Permission { ReadFs, WriteFs, Network, Exec, Secrets }
struct PluginSchema { inputs: Vec<ParamDef>, outputs: Vec<ParamDef> }
```

### Equations

**lifecycle_state_machine**: Plugin follows valid state transitions.

```
transition: (PluginState, Event) -> Result<PluginState, InvalidTransition>
  Discovered → Loaded → Initialized → Running → Stopped
  Any → Error (on failure)
  Error → Discovered (on reload)
  No skip: Discovered → Running is INVALID
```

**permission_scoping**: Plugin cannot exceed declared permissions.

```
enforce: (Plugin, Operation) -> Result<(), PermissionDenied>
  operation.requires ⊆ plugin.manifest.permissions → Ok
  operation.requires ⊄ plugin.manifest.permissions → Err(PermissionDenied)
```

**schema_validation**: Plugin inputs validated against manifest schema.

```
validate: (PluginSchema, Inputs) -> Result<ValidInputs, SchemaError>
  ∀ required in schema.inputs: required.name ∈ inputs
  ∀ input in inputs: type(input.value) matches schema.inputs[input.name].type
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| state_machine | Valid transitions | No skip, Error recoverable via reload |
| precondition | Permission enforcement | operation.requires ⊆ declared |
| soundness | Schema validation | required inputs present, types match |

### Falsification Tests

- **FALSIFY-PLG-001**: Plugin without Network permission cannot make HTTP request
- **FALSIFY-PLG-002**: Plugin in Discovered state cannot execute (must initialize first)
- **FALSIFY-PLG-003**: Plugin with missing required input returns SchemaError
- **FALSIFY-PLG-004**: Plugin failure transitions to Error, not crash

## 33.8 Secret Provider (secret-provider-v1)

### Type Surface

```rust
enum SecretProvider { Env, File, Sops, OnePassword }
struct SecretRef { provider: SecretProvider, key: String }
struct ResolvedSecret { value: String, ephemeral: bool }
```

### Equations

**provider_dispatch**: Every provider kind resolves or fails explicitly.

```
resolve: SecretRef -> Result<ResolvedSecret, SecretError>
  Env → env::var(key).map_err(|_| SecretError::NotFound)
  File → fs::read_to_string(key).map_err(|_| SecretError::NotFound)
  Sops → sops_decrypt(key).map_err(|e| SecretError::DecryptFailed(e))
  OnePassword → op_read(key).map_err(|e| SecretError::ProviderFailed(e))
```

**ephemeral_cleanup**: Ephemeral secrets are zeroed after use.

```
ephemeral: ResolvedSecret -> ()
  drop(secret) when ephemeral=true → memory zeroed (zeroize crate)
  Secret does not appear in: logs, error messages, debug output
```

**drift_detection**: Secret hash change detected between runs.

```
drift: (SecretRef, StoredHash, CurrentHash) -> DriftStatus
  stored_hash = current_hash → NoDrift
  stored_hash ≠ current_hash → Drifted(old, new)
  stored_hash = None → NewSecret
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| completeness | All providers handled | ∀ provider: resolve(provider, key) returns Result |
| frame | Ephemeral cleanup | drop(secret) zeroes memory, no log leakage |
| determinism | Drift detection | same secret → NoDrift, changed → Drifted |

### Falsification Tests

- **FALSIFY-SEC-001**: Env provider returns NotFound for absent variable
- **FALSIFY-SEC-002**: Ephemeral secret not present in panic backtrace
- **FALSIFY-SEC-003**: Changed secret file detected as Drifted on next run
- **FALSIFY-SEC-004**: Sops provider with wrong key returns DecryptFailed, not garbage

## 33.9 Copia Delta Sync (copia-delta-v1)

### Type Surface

```rust
struct BlockIndex { block_size: usize, hashes: Vec<Blake3Hash> }
struct Delta { new_blocks: Vec<(usize, Vec<u8>)>, removed_blocks: Vec<usize> }
struct SyncResult { bytes_transferred: u64, blocks_reused: u64, blocks_new: u64 }
```

### Equations

**delta_correctness**: Applying delta to old file produces new file.

```
apply_delta: (OldFile, Delta) -> NewFile
  apply(old, compute_delta(old, new)) = new
  byte-for-byte equality
```

**block_reuse**: Unchanged blocks are not retransferred.

```
reuse: (BlockIndex_old, BlockIndex_new) -> SyncResult
  ∀ i: hash_old[i] = hash_new[i] → block[i] reused (0 bytes transferred)
  bytes_transferred = sum(size(block) for block in delta.new_blocks)
```

**transfer_minimality**: Delta is minimal (no redundant blocks).

```
minimal: Delta -> bool
  ∀ (i, data) in new_blocks: hash(data) ≠ old_hashes[i]
  No block is included in new_blocks if it already matches
```

**identity_sync**: Identical files produce empty delta.

```
identity: (File, File) -> Delta
  compute_delta(f, f) = Delta { new_blocks: [], removed_blocks: [] }
  bytes_transferred = 0
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| equivalence | Delta correctness | apply(old, delta(old, new)) = new |
| conservation | Block reuse | unchanged blocks → 0 transfer |
| bound | Transfer minimality | no redundant blocks in delta |
| idempotency | Identity sync | delta(f, f) = empty |

### Falsification Tests

- **FALSIFY-COPIA-001**: Single changed block in 1MB file transfers only that block
- **FALSIFY-COPIA-002**: Identical files produce 0 bytes transferred
- **FALSIFY-COPIA-003**: Appended data produces delta with new blocks only at end
- **FALSIFY-COPIA-004**: Delta apply on 100MB file matches byte-for-byte
- **FALSIFY-COPIA-005**: Empty file delta to 1MB file transfers all blocks
- **FALSIFY-COPIA-006**: Block size change forces full resync (no hash reuse)

## 33.10 Sandbox Isolation (sandbox-isolation-v1)

### Type Surface

```rust
enum SandboxBackend { Seccomp, Overlayfs, Netns, Combined }
struct SandboxConfig { backend: SandboxBackend, allowed_paths: Vec<PathBuf>, network: bool }
struct SandboxResult { exit_code: i32, stdout: String, stderr: String, fs_changes: Vec<FsChange> }
enum FsChange { Created(PathBuf), Modified(PathBuf), Deleted(PathBuf) }
```

### Equations

**filesystem_isolation**: Sandboxed process cannot write outside allowed paths.

```
isolate: (SandboxConfig, Process) -> SandboxResult
  ∀ write in process.writes: write.path ∈ config.allowed_paths ∨ write.path ∈ overlay
  Host filesystem unmodified outside allowed_paths
```

**network_isolation**: Sandboxed process without network permission cannot connect.

```
network: (SandboxConfig, Process) -> bool
  config.network = false → all connect() calls fail with ENETUNREACH
  config.network = true → normal network access
```

**overlay_capture**: All filesystem mutations captured in overlay.

```
overlay: (Process, OverlayFs) -> Vec<FsChange>
  ∀ mutation by process: mutation ∈ overlay.upper_dir
  overlay.lower_dir unchanged (read-only)
  merge(lower, upper) = final filesystem state
```

### Proof Obligations

| Type | Property | Formal |
|------|----------|--------|
| frame | FS isolation | writes only to allowed_paths ∪ overlay |
| precondition | Network isolation | network=false → no outbound connections |
| completeness | Overlay captures all mutations | every write in overlay.upper |
| conservation | Lower dir read-only | overlay.lower unchanged after execution |

### Falsification Tests

- **FALSIFY-SBX-001**: Sandboxed `touch /etc/test` fails (outside allowed paths)
- **FALSIFY-SBX-002**: Sandboxed `curl` with network=false gets ENETUNREACH
- **FALSIFY-SBX-003**: Sandboxed file creation appears in overlay upper dir
- **FALSIFY-SBX-004**: Host /tmp unmodified after sandboxed write to /tmp
- **FALSIFY-SBX-005**: Overlay lower dir is byte-identical before and after execution

## 33.11 Summary Matrix

| Contract | Equations | Obligations | Falsification | Kani |
|----------|-----------|-------------|---------------|------|
| store-cas-v1 | 5 | 6 | 9 | 6 |
| oci-manifest-v1 | 4 | 5 | 5 | 5 |
| task-pipeline-v1 | 4 | 5 | 7 | 5 |
| event-rulebook-v1 | 3 | 4 | 5 | 4 |
| plugin-lifecycle-v1 | 3 | 3 | 4 | 3 |
| secret-provider-v1 | 3 | 3 | 4 | 3 |
| copia-delta-v1 | 4 | 4 | 6 | 4 |
| sandbox-isolation-v1 | 3 | 4 | 5 | 4 |
| **Total** | **29** | **34** | **45** | **34** |

This brings Forjar from 5 contracts (blake3, dag, execution, recipe, codegen) to **13 contracts**
covering all heavy type domains. Combined with the existing contracts, Forjar would have formal
verification across its entire critical path: hash → store → build → execute → sync → isolate.

## 33.12 Cross-Domain Contract Dependencies

```
blake3-state-v1 ←── store-cas-v1 (store paths use BLAKE3)
                ←── copia-delta-v1 (block hashes use BLAKE3)
                ←── oci-manifest-v1 (layer digests)

dag-ordering-v1 ←── task-pipeline-v1 (stage dependency order)

execution-safety-v1 ←── sandbox-isolation-v1 (sandboxed execution)
                    ←── task-pipeline-v1 (jidoka failure policy)

recipe-determinism-v1 ←── store-cas-v1 (recipe expansion → derivation)

codegen-dispatch-v1 ←── sandbox-isolation-v1 (generated scripts run in sandbox)
```
