import { FormEvent, startTransition, useEffect, useMemo, useState } from "react";

type ResourceData = {
  backends: Record<string, unknown>;
  models: Record<string, unknown>;
  deployments: Record<string, unknown>;
  gpu_devices: Record<string, unknown>;
  gpu_groups: Record<string, unknown>;
};

type RuntimeData = {
  queued_requests: Array<Record<string, unknown>>;
  active_requests: Record<string, unknown>;
  batch_jobs: Record<string, unknown>;
  drains: Record<string, unknown>;
  deployments: Record<string, unknown>;
  gpu_observations: Record<string, unknown>;
  gpu_groups: Record<string, unknown>;
  backend_health: Record<string, unknown>;
};

type EnvironmentStatus = {
  hf_token_configured: boolean;
  hf_home: string | null;
  environment_file: string | null;
};

type ModelOnboardingForm = {
  modelId: string;
  source: string;
  displayName: string;
  tokenizer: string;
  tasks: Array<
    "chat" | "completion" | "embedding" | "audio_speech" | "audio_transcription" | "audio_translation"
  >;
  modelCapabilities: Array<"text" | "audio" | "image" | "video">;
  backendCompatibility: string[];
  createDeployment: boolean;
  deploymentId: string;
  apiModelName: string;
  deploymentDisplayName: string;
  backendRuntimeId: string;
  gpuGroupId: string;
  memoryFraction: string;
};

type DeploymentOnboardingForm = {
  deploymentId: string;
  modelId: string;
  backendRuntimeId: string;
  gpuGroupId: string;
  apiModelName: string;
  displayName: string;
  memoryFraction: string;
  tasks: Array<
    "chat" | "completion" | "embedding" | "audio_speech" | "audio_transcription" | "audio_translation"
  >;
};

const collections = [
  ["backends", "backends"],
  ["models", "models"],
  ["deployments", "deployments"],
  ["gpu-groups", "gpu_groups"],
  ["gpu-devices", "gpu_devices"],
] as const;

async function api<T>(path: string, token: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as T;
}

export default function App() {
  const [token, setToken] = useState<string>(() => localStorage.getItem("shardon_admin_token") ?? "");
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("admin");
  const [resources, setResources] = useState<ResourceData | null>(null);
  const [runtime, setRuntime] = useState<RuntimeData | null>(null);
  const [environmentStatus, setEnvironmentStatus] = useState<EnvironmentStatus | null>(null);
  const [apiKeys, setApiKeys] = useState<Array<Record<string, unknown>>>([]);
  const [events, setEvents] = useState<string[]>([]);
  const [message, setMessage] = useState<string>("");
  const [collection, setCollection] = useState<(typeof collections)[number][0]>("deployments");
  const [selectedId, setSelectedId] = useState("");
  const [editorId, setEditorId] = useState("");
  const [editorValue, setEditorValue] = useState("{}");
  const [drainGroupId, setDrainGroupId] = useState("");
  const [loadDeploymentId, setLoadDeploymentId] = useState("chat-a");
  const [unloadDeploymentId, setUnloadDeploymentId] = useState("chat-a");
  const [newKeyId, setNewKeyId] = useState("demo-key");
  const [newUserName, setNewUserName] = useState("demo-user");
  const [createdSecret, setCreatedSecret] = useState("");
  const [modelForm, setModelForm] = useState<ModelOnboardingForm>({
    modelId: "llama-3-1-8b-instruct",
    source: "meta-llama/Llama-3.1-8B-Instruct",
    displayName: "Llama 3.1 8B Instruct",
    tokenizer: "",
    tasks: ["chat", "completion"],
    modelCapabilities: ["text"],
    backendCompatibility: [],
    createDeployment: true,
    deploymentId: "llama-3-1-8b-instruct-a",
    apiModelName: "llama-3.1-8b",
    deploymentDisplayName: "Llama 3.1 8B / Group A",
    backendRuntimeId: "",
    gpuGroupId: "",
    memoryFraction: "0.90",
  });
  const [deploymentForm, setDeploymentForm] = useState<DeploymentOnboardingForm>({
    deploymentId: "llama-3-1-8b-instruct-b",
    modelId: "",
    backendRuntimeId: "",
    gpuGroupId: "",
    apiModelName: "llama-3.1-8b",
    displayName: "Llama 3.1 8B / Group B",
    memoryFraction: "0.90",
    tasks: ["chat", "completion"],
  });

  const collectionKey = useMemo(() => {
    return collections.find(([id]) => id === collection)?.[1] ?? "deployments";
  }, [collection]);

  async function refresh(currentToken = token) {
    if (!currentToken) return;
    const [resourceData, runtimeData, keyData, eventData, envData] = await Promise.all([
      api<ResourceData>("/resources", currentToken),
      api<RuntimeData>("/runtime/status", currentToken),
      api<Array<Record<string, unknown>>>("/api-keys", currentToken),
      api<{ lines: string[] }>("/runtime/events", currentToken),
      api<EnvironmentStatus>("/runtime/environment", currentToken),
    ]);
    startTransition(() => {
      setResources(resourceData);
      setRuntime(runtimeData);
      setApiKeys(keyData);
      setEvents(eventData.lines);
      setEnvironmentStatus(envData);
      setMessage("State refreshed.");
    });
  }

  useEffect(() => {
    if (token) {
      refresh(token).catch((error: Error) => setMessage(error.message));
    }
  }, [token]);

  useEffect(() => {
    if (!resources) return;
    const bucket = (resources[collectionKey as keyof ResourceData] ?? {}) as Record<string, unknown>;
    if (selectedId && bucket[selectedId]) {
      setEditorId(selectedId);
      setEditorValue(JSON.stringify(bucket[selectedId], null, 2));
      return;
    }
    const firstId = Object.keys(bucket)[0] ?? "";
    setSelectedId(firstId);
    setEditorId(firstId);
    setEditorValue(firstId ? JSON.stringify(bucket[firstId], null, 2) : "{}");
  }, [collection, collectionKey, resources, selectedId]);

  useEffect(() => {
    if (!resources) return;
    const backendIds = Object.keys(resources.backends ?? {});
    const backendTypes = Array.from(
      new Set(
        Object.values(resources.backends ?? {}).map(
          (backend) => String((backend as { backend_type?: string }).backend_type ?? ""),
        ),
      ),
    ).filter(Boolean);
    const gpuGroupIds = Object.keys(resources.gpu_groups ?? {});
    setModelForm((current) => ({
      ...current,
      backendCompatibility:
        current.backendCompatibility.length > 0
          ? current.backendCompatibility
          : backendTypes.slice(0, 1),
      backendRuntimeId: current.backendRuntimeId || backendIds[0] || "",
      gpuGroupId: current.gpuGroupId || gpuGroupIds[0] || "",
    }));
  }, [resources]);

  useEffect(() => {
    if (!resources) return;
    const backendIds = Object.keys(resources.backends ?? {});
    const gpuGroupIds = Object.keys(resources.gpu_groups ?? {});
    const modelIds = Object.keys(resources.models ?? {});
    setDeploymentForm((current) => ({
      ...current,
      modelId: current.modelId || modelIds[0] || "",
      backendRuntimeId: current.backendRuntimeId || backendIds[0] || "",
      gpuGroupId: current.gpuGroupId || gpuGroupIds[0] || "",
    }));
  }, [resources]);

  async function login(event: FormEvent) {
    event.preventDefault();
    const response = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    if (!response.ok) {
      setMessage(await response.text());
      return;
    }
    const payload = (await response.json()) as { access_token: string };
    localStorage.setItem("shardon_admin_token", payload.access_token);
    setToken(payload.access_token);
  }

  async function saveResource() {
    if (!editorId) return;
    await api(`/resources/${collection}/${editorId}`, token, {
      method: "PUT",
      body: editorValue,
    });
    setSelectedId(editorId);
    setMessage(`Saved ${collection}/${editorId}.`);
    await refresh();
  }

  async function deleteResource() {
    if (!selectedId) return;
    await api(`/resources/${collection}/${selectedId}`, token, { method: "DELETE" });
    setMessage(`Deleted ${collection}/${selectedId}.`);
    setSelectedId("");
    await refresh();
  }

  async function createKey() {
    const payload = await api<{ secret: string }>("/api-keys", token, {
      method: "POST",
      body: JSON.stringify({
        key_id: newKeyId,
        user_name: newUserName,
        priority: 100,
        permissions: ["inference"],
      }),
    });
    setCreatedSecret(payload.secret);
    await refresh();
  }

  async function drainGroup() {
    if (!drainGroupId) return;
    await api(`/runtime/drain/${drainGroupId}`, token, {
      method: "POST",
      body: JSON.stringify({ timeout_seconds: 120 }),
    });
    setMessage(`Drain finished for ${drainGroupId}.`);
    await refresh();
  }

  async function loadDeployment() {
    if (!loadDeploymentId) return;
    await api(`/runtime/load/${loadDeploymentId}`, token, { method: "POST" });
    setMessage(`Loaded ${loadDeploymentId}.`);
    await refresh();
  }

  async function unloadDeployment() {
    if (!unloadDeploymentId) return;
    await api(`/runtime/unload/${unloadDeploymentId}`, token, { method: "POST" });
    setMessage(`Unloaded ${unloadDeploymentId}.`);
    await refresh();
  }

  function toggleTask(
    task: "chat" | "completion" | "embedding" | "audio_speech" | "audio_transcription" | "audio_translation",
  ) {
    setModelForm((current) => ({
      ...current,
      tasks: current.tasks.includes(task)
        ? current.tasks.filter((item) => item !== task)
        : [...current.tasks, task],
    }));
  }

  function toggleModelCapability(capability: "text" | "audio" | "image" | "video") {
    setModelForm((current) => ({
      ...current,
      modelCapabilities: current.modelCapabilities.includes(capability)
        ? current.modelCapabilities.filter((item) => item !== capability)
        : [...current.modelCapabilities, capability],
    }));
  }

  function toggleBackendCompatibility(backendId: string) {
    setModelForm((current) => ({
      ...current,
      backendCompatibility: current.backendCompatibility.includes(backendId)
        ? current.backendCompatibility.filter((item) => item !== backendId)
        : [...current.backendCompatibility, backendId],
    }));
  }

  async function submitModelOnboarding() {
    await api("/workflows/model-onboarding", token, {
      method: "POST",
      body: JSON.stringify({
        model_id: modelForm.modelId,
        source: modelForm.source,
        display_name: modelForm.displayName,
        backend_compatibility: modelForm.backendCompatibility,
        tasks: modelForm.tasks,
        model_capabilities: modelForm.modelCapabilities,
        tokenizer: modelForm.tokenizer || null,
        create_deployment: modelForm.createDeployment,
        deployment_id: modelForm.createDeployment ? modelForm.deploymentId : null,
        api_model_name: modelForm.createDeployment ? modelForm.apiModelName : null,
        deployment_display_name: modelForm.createDeployment ? modelForm.deploymentDisplayName : null,
        backend_runtime_id: modelForm.createDeployment ? modelForm.backendRuntimeId : null,
        gpu_group_id: modelForm.createDeployment ? modelForm.gpuGroupId : null,
        memory_fraction: Number(modelForm.memoryFraction),
      }),
    });
    setMessage(`Onboarded model ${modelForm.modelId}.`);
    setCollection("models");
    setSelectedId(modelForm.modelId);
    await refresh();
  }

  function toggleDeploymentTask(
    task: "chat" | "completion" | "embedding" | "audio_speech" | "audio_transcription" | "audio_translation",
  ) {
    setDeploymentForm((current) => ({
      ...current,
      tasks: current.tasks.includes(task)
        ? current.tasks.filter((item) => item !== task)
        : [...current.tasks, task],
    }));
  }

  async function submitDeploymentOnboarding() {
    await api(`/resources/deployments/${deploymentForm.deploymentId}`, token, {
      method: "PUT",
      body: JSON.stringify({
        id: deploymentForm.deploymentId,
        model_id: deploymentForm.modelId,
        backend_runtime_id: deploymentForm.backendRuntimeId,
        gpu_group_id: deploymentForm.gpuGroupId,
        api_model_name: deploymentForm.apiModelName,
        display_name: deploymentForm.displayName,
        memory_fraction: Number(deploymentForm.memoryFraction),
        enabled: true,
        priority_weight: 100,
        tasks: deploymentForm.tasks,
        extra: {},
      }),
    });
    setMessage(`Added deployment ${deploymentForm.deploymentId}.`);
    setCollection("deployments");
    setSelectedId(deploymentForm.deploymentId);
    await refresh();
  }

  if (!token) {
    return (
      <main className="shell shell-login">
        <section className="login-card">
          <p className="eyebrow">Shardon</p>
          <h1>Pardon, one model at a time.</h1>
          <p className="lede">Sign in to the control plane and manage deployments, queues, drains, keys, and runtime health.</p>
          <form onSubmit={login} className="login-form">
            <label>
              Username
              <input value={username} onChange={(event) => setUsername(event.target.value)} />
            </label>
            <label>
              Password
              <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} />
            </label>
            <button type="submit">Sign In</button>
          </form>
          <p className="message">{message}</p>
        </section>
      </main>
    );
  }

  const resourceBucket = (resources?.[collectionKey as keyof ResourceData] ?? {}) as Record<string, unknown>;
  const backendIds = Object.keys(resources?.backends ?? {});
  const gpuGroupIds = Object.keys(resources?.gpu_groups ?? {});
  const backendTypes = Array.from(
    new Set(
      Object.values(resources?.backends ?? {}).map(
        (backend) => String((backend as { backend_type?: string }).backend_type ?? ""),
      ),
    ),
  ).filter(Boolean);

  return (
    <main className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Shardon Admin</p>
          <h1>Operational control for constrained GPU routing.</h1>
          <p className="lede">Separate control plane, file-backed desired state, live runtime visibility, and aggressive keep-free enforcement for shared hardware.</p>
        </div>
        <div className="hero-actions">
          <button onClick={() => refresh().catch((error: Error) => setMessage(error.message))}>Refresh</button>
          <button
            className="secondary"
            onClick={() => {
              localStorage.removeItem("shardon_admin_token");
              setToken("");
            }}
          >
            Sign Out
          </button>
        </div>
      </header>

      <section className="message-bar">{message}</section>

      <section className="metrics-grid">
        <article className="metric-card">
          <span>Queued</span>
          <strong>{runtime?.queued_requests.length ?? 0}</strong>
        </article>
        <article className="metric-card">
          <span>Active</span>
          <strong>{Object.keys(runtime?.active_requests ?? {}).length}</strong>
        </article>
        <article className="metric-card">
          <span>Batch Jobs</span>
          <strong>{Object.keys(runtime?.batch_jobs ?? {}).length}</strong>
        </article>
        <article className="metric-card">
          <span>Loaded Deployments</span>
          <strong>
            {
              Object.values(runtime?.deployments ?? {}).filter((item) => (item as { loaded?: boolean }).loaded)
                .length
            }
          </strong>
        </article>
      </section>

      <section className="dashboard-grid">
        <article className="panel">
          <div className="panel-header">
            <h2>Add Model</h2>
            <p>Add a model and, optionally, create its first deployment in one step.</p>
          </div>
          <div className="stack">
            <div className="hint-box">
              <strong>HF downloads</strong>
              <span>
                `HF_TOKEN` is read from the process environment or repo `.env`, never from this UI.
              </span>
              <span>
                Status: {environmentStatus?.hf_token_configured ? "configured" : "not configured"}
              </span>
              <span>Env file: {environmentStatus?.environment_file ?? "none detected"}</span>
            </div>
            <label>
              Model ID
              <input
                value={modelForm.modelId}
                onChange={(event) => setModelForm((current) => ({ ...current, modelId: event.target.value }))}
              />
            </label>
            <label>
              Hugging Face repo or local path
              <input
                value={modelForm.source}
                onChange={(event) => setModelForm((current) => ({ ...current, source: event.target.value }))}
              />
            </label>
            <label>
              Display Name
              <input
                value={modelForm.displayName}
                onChange={(event) => setModelForm((current) => ({ ...current, displayName: event.target.value }))}
              />
            </label>
            <label>
              Tokenizer
              <input
                value={modelForm.tokenizer}
                onChange={(event) => setModelForm((current) => ({ ...current, tokenizer: event.target.value }))}
                placeholder="optional tokenizer override"
              />
            </label>
            <div className="check-grid">
              {(
                ["chat", "completion", "embedding", "audio_speech", "audio_transcription", "audio_translation"] as const
              ).map((task) => (
                <label key={task} className="check-item">
                  <input
                    type="checkbox"
                    checked={modelForm.tasks.includes(task)}
                    onChange={() => toggleTask(task)}
                  />
                  <span>{task}</span>
                </label>
              ))}
            </div>
            <div className="subhead">Model Capabilities</div>
            <div className="check-grid">
              {(["text", "audio", "image", "video"] as const).map((capability) => (
                <label key={capability} className="check-item">
                  <input
                    type="checkbox"
                    checked={modelForm.modelCapabilities.includes(capability)}
                    onChange={() => toggleModelCapability(capability)}
                  />
                  <span>{capability}</span>
                </label>
              ))}
            </div>
            <div className="subhead">Compatible Backends</div>
            <div className="check-grid">
              {backendTypes.map((backendType) => (
                <label key={backendType} className="check-item">
                  <input
                    type="checkbox"
                    checked={modelForm.backendCompatibility.includes(backendType)}
                    onChange={() => toggleBackendCompatibility(backendType)}
                  />
                  <span>{backendType}</span>
                </label>
              ))}
            </div>
            <label className="check-item">
              <input
                type="checkbox"
                checked={modelForm.createDeployment}
                onChange={(event) =>
                  setModelForm((current) => ({ ...current, createDeployment: event.target.checked }))
                }
              />
              <span>Create first deployment now</span>
            </label>
            {modelForm.createDeployment ? (
              <>
                <label>
                  Deployment ID
                  <input
                    value={modelForm.deploymentId}
                    onChange={(event) =>
                      setModelForm((current) => ({ ...current, deploymentId: event.target.value }))
                    }
                  />
                </label>
                <label>
                  API Model Alias
                  <input
                    value={modelForm.apiModelName}
                    onChange={(event) =>
                      setModelForm((current) => ({ ...current, apiModelName: event.target.value }))
                    }
                  />
                </label>
                <label>
                  Deployment Display Name
                  <input
                    value={modelForm.deploymentDisplayName}
                    onChange={(event) =>
                      setModelForm((current) => ({ ...current, deploymentDisplayName: event.target.value }))
                    }
                  />
                </label>
                <label>
                  Backend Runtime
                  <select
                    value={modelForm.backendRuntimeId}
                    onChange={(event) =>
                      setModelForm((current) => ({ ...current, backendRuntimeId: event.target.value }))
                    }
                  >
                    {backendIds.map((backendId) => (
                      <option key={backendId} value={backendId}>
                        {backendId}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  GPU Group
                  <select
                    value={modelForm.gpuGroupId}
                    onChange={(event) =>
                      setModelForm((current) => ({ ...current, gpuGroupId: event.target.value }))
                    }
                  >
                    {gpuGroupIds.map((gpuGroupId) => (
                      <option key={gpuGroupId} value={gpuGroupId}>
                        {gpuGroupId}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Memory Fraction
                  <input
                    value={modelForm.memoryFraction}
                    onChange={(event) =>
                      setModelForm((current) => ({ ...current, memoryFraction: event.target.value }))
                    }
                  />
                </label>
              </>
            ) : null}
            <button onClick={() => submitModelOnboarding().catch((error: Error) => setMessage(error.message))}>
              Save Model
            </button>
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <h2>Add Deployment</h2>
            <p>Place an existing logical model onto another backend runtime or GPU group.</p>
          </div>
          <div className="stack">
            <label>
              Deployment ID
              <input
                value={deploymentForm.deploymentId}
                onChange={(event) =>
                  setDeploymentForm((current) => ({ ...current, deploymentId: event.target.value }))
                }
              />
            </label>
            <label>
              Model
              <select
                value={deploymentForm.modelId}
                onChange={(event) =>
                  setDeploymentForm((current) => ({ ...current, modelId: event.target.value }))
                }
              >
                {Object.keys(resources?.models ?? {}).map((modelId) => (
                  <option key={modelId} value={modelId}>
                    {modelId}
                  </option>
                ))}
              </select>
            </label>
            <label>
              API Model Alias
              <input
                value={deploymentForm.apiModelName}
                onChange={(event) =>
                  setDeploymentForm((current) => ({ ...current, apiModelName: event.target.value }))
                }
              />
            </label>
            <label>
              Display Name
              <input
                value={deploymentForm.displayName}
                onChange={(event) =>
                  setDeploymentForm((current) => ({ ...current, displayName: event.target.value }))
                }
              />
            </label>
            <label>
              Backend Runtime
              <select
                value={deploymentForm.backendRuntimeId}
                onChange={(event) =>
                  setDeploymentForm((current) => ({ ...current, backendRuntimeId: event.target.value }))
                }
              >
                {backendIds.map((backendId) => (
                  <option key={backendId} value={backendId}>
                    {backendId}
                  </option>
                ))}
              </select>
            </label>
            <label>
              GPU Group
              <select
                value={deploymentForm.gpuGroupId}
                onChange={(event) =>
                  setDeploymentForm((current) => ({ ...current, gpuGroupId: event.target.value }))
                }
              >
                {gpuGroupIds.map((gpuGroupId) => (
                  <option key={gpuGroupId} value={gpuGroupId}>
                    {gpuGroupId}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Memory Fraction
              <input
                value={deploymentForm.memoryFraction}
                onChange={(event) =>
                  setDeploymentForm((current) => ({ ...current, memoryFraction: event.target.value }))
                }
              />
            </label>
            <div className="check-grid">
              {(
                ["chat", "completion", "embedding", "audio_speech", "audio_transcription", "audio_translation"] as const
              ).map((task) => (
                <label key={task} className="check-item">
                  <input
                    type="checkbox"
                    checked={deploymentForm.tasks.includes(task)}
                    onChange={() => toggleDeploymentTask(task)}
                  />
                  <span>{task}</span>
                </label>
              ))}
            </div>
            <button onClick={() => submitDeploymentOnboarding().catch((error: Error) => setMessage(error.message))}>
              Save Deployment
            </button>
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <h2>Config Editor</h2>
            <p>Raw editor for backends, models, deployments, GPU groups, and GPU mapping.</p>
          </div>
          <div className="editor-controls">
            <select value={collection} onChange={(event) => setCollection(event.target.value as (typeof collections)[number][0])}>
              {collections.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
            <select value={selectedId} onChange={(event) => setSelectedId(event.target.value)}>
              <option value="">New item</option>
              {Object.keys(resourceBucket).map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </div>
          <label>
            Item ID
            <input value={editorId} onChange={(event) => setEditorId(event.target.value)} placeholder="chat-c" />
          </label>
          <textarea value={editorValue} onChange={(event) => setEditorValue(event.target.value)} rows={18} />
          <div className="row-actions">
            <button onClick={() => saveResource().catch((error: Error) => setMessage(error.message))}>Save</button>
            <button className="secondary" onClick={() => deleteResource().catch((error: Error) => setMessage(error.message))}>
              Delete
            </button>
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <h2>Keys and Runtime Control</h2>
            <p>Issue inference keys, load or unload deployments, and drain a GPU group on demand.</p>
          </div>
          <div className="stack">
            <label>
              Key ID
              <input value={newKeyId} onChange={(event) => setNewKeyId(event.target.value)} />
            </label>
            <label>
              User Name
              <input value={newUserName} onChange={(event) => setNewUserName(event.target.value)} />
            </label>
            <button onClick={() => createKey().catch((error: Error) => setMessage(error.message))}>Create API Key</button>
            <code className="secret-box">{createdSecret || "New secret appears here once, after creation."}</code>
            <label>
              GPU Group ID
              <input value={drainGroupId} onChange={(event) => setDrainGroupId(event.target.value)} />
            </label>
            <button className="secondary" onClick={() => drainGroup().catch((error: Error) => setMessage(error.message))}>
              Drain Group
            </button>
            <label>
              Load Deployment
              <input value={loadDeploymentId} onChange={(event) => setLoadDeploymentId(event.target.value)} />
            </label>
            <button className="secondary" onClick={() => loadDeployment().catch((error: Error) => setMessage(error.message))}>
              Load Deployment
            </button>
            <label>
              Unload Deployment
              <input value={unloadDeploymentId} onChange={(event) => setUnloadDeploymentId(event.target.value)} />
            </label>
            <button className="secondary" onClick={() => unloadDeployment().catch((error: Error) => setMessage(error.message))}>
              Unload Deployment
            </button>
          </div>
          <ul className="compact-list">
            {apiKeys.map((item) => (
              <li key={String(item.id)}>
                <strong>{String(item.id)}</strong>
                <span>{String(item.user_name)}</span>
                <span>{String(item.revoked_at ?? "active")}</span>
              </li>
            ))}
          </ul>
        </article>

        <article className="panel panel-wide">
          <div className="panel-header">
            <h2>Queues and Runtime</h2>
            <p>Loaded deployments, group summaries, active requests, batch jobs, drains, and GPU observations.</p>
          </div>
          <pre>{JSON.stringify({ gpu_groups: runtime?.gpu_groups, backend_health: runtime?.backend_health }, null, 2)}</pre>
        </article>

        <article className="panel panel-wide">
          <div className="panel-header">
            <h2>Runtime Details</h2>
            <p>Full raw runtime snapshot for deeper debugging.</p>
          </div>
          <pre>{JSON.stringify(runtime, null, 2)}</pre>
        </article>

        <article className="panel panel-wide">
          <div className="panel-header">
            <h2>Events</h2>
            <p>Recent JSONL event stream for debugging and operations.</p>
          </div>
          <pre>{events.join("\n")}</pre>
        </article>
      </section>
    </main>
  );
}
