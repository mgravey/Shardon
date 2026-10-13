import { useEffect, useState } from "react";

type RequestActivity = {
  id: string; user_name: string; api_key_id: string; endpoint: string; priority: number;
  phase: "waiting" | "loading" | "in_flight"; elapsed_seconds: number; wait_seconds: number;
  wait_reason: string | null; deployment_id: string; gpu_group_id: string;
};
type Activity = {
  generated_at: string; in_flight: number; waiting: number; loading: number;
  models: Array<{
    model_name: string; in_flight: number; waiting: number; loading: number;
    oldest_wait_seconds: number; requests: RequestActivity[];
    deployments: Array<{ id: string; state: string; gpu_group_id: string | null }>;
  }>;
  batches: Array<{
    id: string; user_name: string; api_key_id: string; model_name: string; status: string;
    deployment_id: string | null; gpu_group_id: string | null; total_items: number;
    completed_items: number; failed_items: number; cancelled_items: number;
    current_item: number | null; remaining_items: number;
  }>;
};
type Totals = {
  requests: number; completed: number; failed: number; cancelled: number;
  input_tokens: number | null; output_tokens: number | null; total_tokens: number | null;
  usage_available: number; usage_coverage: number;
};
type Usage = {
  generated_at: string; since: string; until: string; totals: Totals;
  rows: Array<Totals & {
    identity: string; user_names: string[]; models: Array<Totals & { model_name: string }>;
  }>;
};

function usePolling<T>(path: string, interval: number) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const [updated, setUpdated] = useState<Date | null>(null);
  const [refreshing, setRefreshing] = useState(true);
  useEffect(() => {
    let disposed = false;
    let busy = false;
    const controller = new AbortController();
    setRefreshing(true);
    async function refresh() {
      if (disposed || busy || document.hidden) return;
      busy = true;
      try {
        const response = await fetch(`/api${path}`, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`Update failed (${response.status})`);
        const next = await response.json() as T;
        if (!disposed) { setData(next); setError(""); setUpdated(new Date()); }
      } catch (failure) {
        if (!disposed) setError(failure instanceof Error ? failure.message : "Update failed");
      } finally {
        busy = false;
        if (!disposed) setRefreshing(false);
      }
    }
    void refresh();
    const timer = window.setInterval(() => { void refresh(); }, interval);
    document.addEventListener("visibilitychange", refresh);
    return () => {
      disposed = true; controller.abort(); window.clearInterval(timer);
      document.removeEventListener("visibilitychange", refresh);
    };
  }, [path, interval]);
  return { data, error, updated, refreshing };
}

const count = (value: number | null) => value === null ? "Unavailable" : value.toLocaleString();
const duration = (value: number) => value < 60 ? `${Math.floor(value)}s` : `${Math.floor(value / 60)}m ${Math.floor(value % 60)}s`;
const phaseLabel = (phase: string) => ({ in_flight: "In flight", loading: "Loading model", waiting: "Waiting" }[phase] ?? phase);

function Freshness({ error, updated, refreshing }: { error: string; updated: Date | null; refreshing: boolean }) {
  return <span className={error ? "error-text" : "activity-freshness"} role="status">
    {error ? `Stale · ${error}` : refreshing ? "Refreshing…" : "Auto-refresh on"}
    {updated && ` · Updated ${updated.toLocaleTimeString()}`}
  </span>;
}

export function LoadedModelsBanner() {
  // Keep the global loaded-model display independent of the activity filters.
  const activity = usePolling<Activity>("/runtime/activity", 2000);
  const loaded = activity.data?.models.filter(model =>
    model.deployments.some(deployment => deployment.state === "ready"),
  ) ?? [];
  const loading = activity.data?.models.filter(model =>
    model.deployments.some(deployment => deployment.state === "starting"),
  ) ?? [];
  return <div className="loaded-model-banner">
    <p className="eyebrow">Shardon · Currently loaded{loaded.length > 1 ? " models" : " model"}</p>
    <h1 className="loaded-model-name">
      {loaded.length > 0 ? loaded.map(model => <span key={model.model_name}>{model.model_name}</span>)
        : !activity.data ? (activity.error ? "Model status unavailable" : "Checking loaded models…")
        : "No model loaded"}
    </h1>
    {loaded.map(model => <p className="loaded-model-placement" key={model.model_name}>
      <span className="status-pill status-ready">Loaded</span>
      {loaded.length > 1 && <strong>{model.model_name}</strong>}
      {model.deployments.filter(deployment => deployment.state === "ready").map(deployment =>
        <span key={deployment.id}>{deployment.id}{deployment.gpu_group_id ? ` · ${deployment.gpu_group_id}` : ""}</span>,
      )}
    </p>)}
    {loading.length > 0 && <p className="loaded-model-loading">Loading: {loading.map(model => model.model_name).join(", ")}</p>}
    <Freshness {...activity} />
  </div>;
}

function UsageCells({ row }: { row: Totals }) {
  return <><td>{row.requests}</td><td>{row.failed} / {row.cancelled}</td>
    <td>{count(row.input_tokens)}</td><td>{count(row.output_tokens)}</td><td>{count(row.total_tokens)}</td>
    <td title={`${row.usage_available} of ${row.requests} requests reported total tokens`}>
      {Math.round(row.usage_coverage * 100)}% <small>({row.usage_available}/{row.requests})</small>
    </td></>;
}

export default function ActivityBoard() {
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [user, setUser] = useState("");
  const [period, setPeriod] = useState("today");
  const [groupBy, setGroupBy] = useState("api_key");
  const [modelView, setModelView] = useState("actual");
  const filters = new URLSearchParams();
  if (model.trim()) filters.set("model", model.trim());
  if (apiKey.trim()) filters.set("api_key", apiKey.trim());
  if (user.trim()) filters.set("user", user.trim());
  const activity = usePolling<Activity>(`/runtime/activity?${filters}`, 2000);
  const usage = usePolling<Usage>(`/runtime/usage?${filters}&period=${period}&group_by=${groupBy}&model_view=${modelView}`, 30000);
  const filtered = Boolean(model || apiKey || user);
  return <div className="observability">
    <section className="panel activity-panel" aria-labelledby="activity-title">
      <div className="panel-header activity-heading"><div>
        <p className="eyebrow">Right now</p><h2 id="activity-title">Live activity</h2>
        <p>Who is using each model and who is waiting. In flight means dispatched to the backend.</p>
      </div><Freshness {...activity} /></div>
      <div className="activity-filters">
        <label>Model<input aria-label="Filter by model" value={model} onChange={e => setModel(e.target.value)} placeholder="All models · exact alias" /></label>
        <label>API key<input aria-label="Filter by API key" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="All keys · exact ID" /></label>
        <label>User<input aria-label="Filter by user" value={user} onChange={e => setUser(e.target.value)} placeholder="All users · exact name" /></label>
        {filtered && <button className="secondary" onClick={() => { setModel(""); setApiKey(""); setUser(""); }}>Clear filters</button>}
      </div>
      <div className="activity-counters">
        <span><strong>{activity.data?.in_flight ?? "—"}</strong> In flight</span>
        <span><strong>{activity.data?.waiting ?? "—"}</strong> Waiting <small>(includes loading)</small></span>
        <span><strong>{activity.data?.loading ?? "—"}</strong> Loading model</span>
      </div>
      {!activity.data && <p className="empty-state">{activity.error ? "Activity is unavailable. Retrying automatically." : "Loading activity…"}</p>}
      {activity.data?.models.length === 0 && <p className="empty-state">No models or requests match these filters.</p>}
      {activity.data?.models.map(row => <details className="model-activity" key={row.model_name}>
        <summary><strong>{row.model_name}</strong>
          <span className="model-deployments">{row.deployments.map(d => <span key={d.id} className={`status-pill status-${d.state}`}>{d.id}: {d.state}{d.gpu_group_id ? ` · ${d.gpu_group_id}` : ""}</span>)}</span>
          <span className="activity-model-counts"><b>{row.in_flight}</b> in flight · <b>{row.waiting}</b> waiting{row.loading > 0 && ` · ${row.loading} loading`}<small>Oldest wait: {row.waiting ? duration(row.oldest_wait_seconds) : "—"}</small></span>
        </summary>
        {row.requests.length === 0 ? <p className="empty-state">No active or waiting requests.</p> : <div className="table-scroll"><table className="activity-table">
          <thead><tr><th>Owner / key</th><th>Request / endpoint</th><th>Status</th><th>Placement</th><th>Priority</th><th>Elapsed / wait</th><th>Wait reason</th></tr></thead>
          <tbody>{row.requests.map(r => <tr key={r.id}><td>{r.user_name}<small>{r.api_key_id}</small></td><td><code>{r.id}</code><small>{r.endpoint}</small></td>
            <td><span className={`status-pill status-${r.phase === "in_flight" ? "ready" : "starting"}`}>{phaseLabel(r.phase)}</span></td>
            <td>{r.deployment_id || "Unassigned"}<small>{r.gpu_group_id || "GPU not yet selected"}</small></td><td>{r.priority}</td>
            <td>{duration(r.elapsed_seconds)}<small>Wait: {duration(r.wait_seconds)}</small></td><td>{r.wait_reason || "—"}</td></tr>)}</tbody>
        </table></div>}
      </details>)}
      <h3>Batch jobs</h3>
      {!activity.data?.batches.length ? <p className="empty-state">No batch jobs{filtered ? " match these filters" : ""}.</p> : <div className="table-scroll"><table className="activity-table">
        <thead><tr><th>Owner / key</th><th>Batch / model</th><th>Status / GPU</th><th>Completed</th><th>Failed / cancelled</th><th>Current item</th><th>Remaining</th></tr></thead>
        <tbody>{activity.data.batches.map(b => <tr key={b.id}><td>{b.user_name}<small>{b.api_key_id}</small></td><td><code>{b.id}</code><small>{b.model_name}</small></td>
          <td>{b.status}<small>{b.gpu_group_id || "Unassigned"}</small></td><td>{b.completed_items} / {b.total_items}</td><td>{b.failed_items} / {b.cancelled_items}</td><td>{b.current_item ?? "—"}</td><td>{b.remaining_items}</td></tr>)}</tbody>
      </table></div>}
    </section>
    <section className="panel" aria-labelledby="usage-title">
      <div className="panel-header activity-heading"><div><p className="eyebrow">Reported consumption</p><h2 id="usage-title">Token usage leaderboard</h2>
        <p>Ranked by reported total tokens. Missing usage is unavailable; coverage shows how complete the totals are.</p></div><Freshness {...usage} /></div>
      <div className="activity-filters"><label>Period (UTC)<select value={period} onChange={e => setPeriod(e.target.value)}>
        <option value="today">Today</option><option value="7d">7 days</option><option value="30d">30 days</option></select></label>
        <label>Rank by<select value={groupBy} onChange={e => setGroupBy(e.target.value)}><option value="api_key">API key</option><option value="user">User</option></select></label>
        <label>Model names<select value={modelView} onChange={e => setModelView(e.target.value)}>
          <option value="actual">Actual model used</option>
          <option value="requested">Requested model name</option>
        </select></label>
        <p>{modelView === "actual" ? "Model breakdowns and the model filter use the actual model used." : "Model breakdowns and the model filter use the requested name, including currently-loaded."} History starts when tracking is enabled.</p></div>
      {usage.data && <p className="usage-summary">{usage.data.totals.requests.toLocaleString()} requests · {count(usage.data.totals.total_tokens)} total tokens · {Math.round(usage.data.totals.usage_coverage * 100)}% coverage</p>}
      {!usage.data ? <p className="empty-state">{usage.error ? "Usage is unavailable. Retrying automatically." : "Loading usage…"}</p> : usage.data.rows.length === 0 ? <p className="empty-state">No recorded requests in this period{filtered ? " matching these filters" : ""}.</p> :
        <div className="table-scroll"><table className="activity-table usage-table"><thead><tr><th>{groupBy === "api_key" ? "API key / owner" : "User"}</th><th>Requests</th><th>Failed / cancelled</th><th>Input tokens</th><th>Output tokens</th><th>Total tokens</th><th>Coverage</th></tr></thead>
          <tbody>{usage.data.rows.map((row, index) => <tr key={row.identity}><td><details><summary><strong>{index + 1}. {row.identity}</strong>{groupBy === "api_key" && <small>{row.user_names.join(", ")}</small>}</summary>
            <div className="usage-models">{row.models.map(m => <div key={m.model_name}><strong>{m.model_name}</strong><span>{m.requests} requests · {count(m.total_tokens)} total tokens</span><small>Input {count(m.input_tokens)} · Output {count(m.output_tokens)} · Failed {m.failed} · Cancelled {m.cancelled} · Coverage {Math.round(m.usage_coverage * 100)}%</small></div>)}</div>
          </details></td><UsageCells row={row} /></tr>)}</tbody></table></div>}
    </section>
  </div>;
}
