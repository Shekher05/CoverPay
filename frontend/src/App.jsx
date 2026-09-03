/* CoverPay fraud monitor.
 *
 * Design read: a fraud operations console for analysts doing repeat daily
 * triage. Dense and utilitarian, not a marketing page. The screen answers one
 * question on open - what needs looking at right now - so the order is
 * deliberate: headline numbers, the shape of recent traffic, open incidents,
 * then the raw feed. Nothing an analyst needs every visit is behind a tab.
 *
 * The distinguishing idea is the engine column in the feed. Each transaction
 * shows which of the three engines flagged it (M / B / I), because the whole
 * architecture rests on them catching different things, and that is otherwise
 * invisible in a list of risk scores.
 *
 * The visual system lives in styles.css: one type scale, one neutral ramp,
 * three radii, one easing curve. This file spends its attention on hierarchy
 * and on the states an operations tool actually sits in, which are loading,
 * empty and broken far more often than the happy path.
 */
import { useCallback, useEffect, useState } from "react";
import { api, count, elapsed, money } from "./api.js";
import { EngineComparison, TimelineChart } from "./charts.jsx";

const REFRESH_MS = 15000;
const EMPTY = "n/a";

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("coverpay-theme") || "system";
    } catch {
      return "system";
    }
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("coverpay-theme", theme);
    } catch {
      /* private mode: the toggle still works for this session */
    }
  }, [theme]);

  return [theme, setTheme];
}

/** Three bars, longest to shortest: model, behaviour, incident. */
function Brand() {
  return (
    <span className="brand">
      <span className="mark" aria-hidden="true">
        <i />
        <i />
        <i />
      </span>
      <h1>CoverPay</h1>
    </span>
  );
}

function Badge({ value }) {
  if (!value) return <span style={{ color: "var(--ink-muted)" }}>{EMPTY}</span>;
  return (
    <span className="badge" data-tone={String(value).toLowerCase()}>
      {value}
    </span>
  );
}

/** Which engines flagged this row. Letters carry the meaning, colour reinforces. */
function EngineChips({ row }) {
  const on = {
    model: row.recommendation !== "ALLOW",
    behaviour: (row.behaviour_score ?? 0) >= 0.3,
    incident: Boolean(row.incident_id),
  };
  const active = Object.entries(on)
    .filter(([, v]) => v)
    .map(([k]) => k);

  return (
    <span
      className="chips"
      title={active.length ? `Flagged by: ${active.join(", ")}` : "Not flagged"}
    >
      <span className="visually-hidden">
        {active.length ? `Flagged by ${active.join(", ")}` : "Not flagged by any engine"}
      </span>
      {[
        ["model", "M"],
        ["behaviour", "B"],
        ["incident", "I"],
      ].map(([key, letter]) => (
        <span
          key={key}
          className="chip"
          data-engine={key}
          data-on={String(on[key])}
          aria-hidden="true"
        >
          {letter}
        </span>
      ))}
    </span>
  );
}

function Tile({ label, value, note, index }) {
  return (
    <div className="tile rise" style={{ "--i": index }}>
      <dt>{label}</dt>
      <dd>{value}</dd>
      {note && <div className="note">{note}</div>}
    </div>
  );
}

/* A skeleton is only useful if it predicts the layout it stands in for. Four
 * tiles in the real tile grid, then the two panels at the real ratio, so
 * nothing jumps when the data lands. */
function Loading() {
  return (
    <>
      <dl className="tiles">
        {[0, 1, 2, 3].map((i) => (
          <div className="skeleton-tile" key={i}>
            <div className="skeleton" style={{ height: 9, width: "52%" }} />
            <div className="skeleton" style={{ height: 26, width: "68%" }} />
            <div className="skeleton" style={{ height: 9, width: "44%" }} />
          </div>
        ))}
      </dl>
      <div className="grid-2">
        <div className="panel">
          <div style={{ padding: 16 }}>
            <div className="skeleton" style={{ height: 200 }} />
          </div>
        </div>
        <div className="panel">
          <div className="skeleton-rows">
            {[100, 72, 46].map((w) => (
              <div className="skeleton" key={w} style={{ height: 22, width: `${w}%` }} />
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

export default function App() {
  const [theme, setTheme] = useTheme();
  const [data, setData] = useState({});
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [live, setLive] = useState(true);
  const [filter, setFilter] = useState(null);
  const [updatedAt, setUpdatedAt] = useState(null);

  const load = useCallback(async () => {
    try {
      const [summary, timeline, incidents, transactions] = await Promise.all([
        api.summary(),
        api.timeline(48),
        api.incidents(40),
        api.transactions(60, filter),
      ]);
      setData({ summary, timeline, incidents, transactions });
      setUpdatedAt(Date.now());
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!live) return undefined;
    const timer = setInterval(load, REFRESH_MS);
    return () => clearInterval(timer);
  }, [live, load]);

  const { summary, timeline, incidents = [], transactions = [] } = data;
  const empty = !loading && !error && (summary?.transactions ?? 0) === 0;

  return (
    <>
      <a className="skip-link" href="#feed">
        Skip to the transaction feed
      </a>

      <div className="shell">
        <header className="masthead">
          <Brand />
          <span className="sub">
            Fraud intelligence. Advisory only, no payment is ever blocked.
          </span>

          <div className="spacer" />
          <div className="controls">
            <span className="pulse" data-state={live ? "live" : "stale"}>
              {live ? "Live" : "Paused"}
              {updatedAt ? ` · ${new Date(updatedAt).toLocaleTimeString()}` : ""}
            </span>
            <button className="ghost" aria-pressed={live} onClick={() => setLive((v) => !v)}>
              {live ? "Pause" : "Resume"}
            </button>
            <button
              className="ghost"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              {theme === "dark" ? "Light" : "Dark"}
            </button>
          </div>
        </header>

        <main id="main">
          {/* A short statement of what the console does, in the console's own
             type scale. Not a landing-page hero. */}
          <header className="hero-banner rise" style={{ "--i": 0 }}>
            <p className="hero-badge">
              <span className="badge-dot" aria-hidden="true" /> Defensive fraud sentinel
            </p>
            <h2 className="hero-headline">
              Real-time transaction scoring, coordinated-spike detection, and a
              grounded merchant assistant.
            </h2>
            <p className="hero-subtext">
              Three independent engines read every transaction, customer, and
              merchant. Advisory only &mdash; no payment is ever blocked.
            </p>
          </header>

          {loading && <Loading />}

          {error && (
            <div className="panel" style={{ marginTop: 24 }}>
              <div className="state">
                <h3>Cannot reach the API</h3>
                <p>{error}</p>
                <p>
                  Start it with <code>uvicorn api.main:app</code>, then seed the database
                  with <code>python -m simulator.feed --rows 4000 --reset</code>.
                </p>
                <button className="ghost" onClick={load}>
                  Retry
                </button>
              </div>
            </div>
          )}

          {empty && (
            <div className="panel" style={{ marginTop: 24 }}>
              <div className="state">
                <h3>Nothing scored yet</h3>
                <p>
                  The API is up and the database is empty. Seed it with{" "}
                  <code>python -m simulator.feed --rows 4000 --reset</code> and this screen
                  fills in on the next refresh.
                </p>
              </div>
            </div>
          )}

          {!loading && !error && !empty && (
            <>
              <CsvUploadSection />

              <section aria-label="Headline figures">
                <dl className="tiles">
                  <Tile
                    index={0}
                    label="Transactions"
                    value={count.format(summary.transactions)}
                    note={`over ${summary.window.hours.toFixed(1)} hours`}
                  />
                  <Tile
                    index={1}
                    label="Flagged"
                    value={count.format(summary.flagged)}
                    note={`${((summary.flagged / summary.transactions) * 100).toFixed(
                      1
                    )}% of volume`}
                  />
                  <Tile
                    index={2}
                    label="Value at risk"
                    value={money.format(summary.flagged_value)}
                    note={`of ${money.format(summary.total_value)} processed`}
                  />
                  <Tile
                    index={3}
                    label="Open incidents"
                    value={count.format(summary.incidents_open)}
                    note={`${count.format(summary.incidents.RESOLVED ?? 0)} resolved`}
                  />
                </dl>
              </section>

              <div className="grid-2">
                <section className="rise" style={{ "--i": 4 }}>
                  <div className="section-head">
                    <h2>Volume and flagged traffic</h2>
                    <p>equal-width buckets across the stream</p>
                  </div>
                  <div className="panel">
                    <TimelineChart data={timeline} />
                  </div>
                </section>

                <section className="rise" style={{ "--i": 5 }}>
                  <div className="section-head">
                    <h2>What each engine sees</h2>
                  </div>
                  <div className="panel">
                    <EngineComparison
                      flags={summary.engine_flags}
                      total={summary.transactions}
                    />
                    <p className="engine-note">
                      The model reads one transaction, the behaviour engine reads a
                      customer&rsquo;s history, the incident engine reads a merchant&rsquo;s
                      traffic. None of them can do another&rsquo;s job.
                    </p>
                  </div>
                </section>
              </div>

              <section className="rise" style={{ "--i": 6 }}>
                <div className="section-head">
                  <h2>Incidents</h2>
                  <p>coordinated activity at a single merchant</p>
                </div>
                <div className="panel scroll">
                  {incidents.length === 0 ? (
                    <div className="state">
                      <h3>No incidents detected</h3>
                      <p>
                        No merchant in this window shows the coordinated pattern the
                        incident engine looks for.
                      </p>
                    </div>
                  ) : (
                    <table>
                      <caption className="visually-hidden">
                        Detected incidents, most severe first
                      </caption>
                      <thead>
                        <tr>
                          <th scope="col">Incident</th>
                          <th scope="col">Merchant</th>
                          <th scope="col">Severity</th>
                          <th scope="col">Status</th>
                          <th scope="col" className="num">Txns</th>
                          <th scope="col" className="num">Customers</th>
                          <th scope="col" className="num">Devices</th>
                          <th scope="col" className="num">Duration</th>
                          <th scope="col" className="num">Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {incidents.map((incident) => (
                          <tr key={incident.incident_id}>
                            <td className="id">{incident.incident_id}</td>
                            <td className="id">{incident.merchant_id}</td>
                            <td>
                              <Badge value={incident.severity} />
                            </td>
                            <td>
                              <Badge value={incident.status} />
                            </td>
                            <td className="num">{count.format(incident.transactions)}</td>
                            <td className="num">{count.format(incident.customers)}</td>
                            <td className="num">{count.format(incident.devices)}</td>
                            <td className="num">{elapsed(incident.duration_seconds)}</td>
                            <td className="num">{money.format(incident.total_value)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </section>

              <section id="feed" className="rise" style={{ "--i": 7 }}>
                <div className="section-head">
                  <h2>Transaction feed</h2>
                  <p>most recent first</p>
                  <div style={{ marginLeft: "auto" }} />
                  <div className="segmented" role="group" aria-label="Filter by advisory">
                    {[null, "REVIEW", "BLOCK"].map((value) => (
                      <button
                        key={value ?? "all"}
                        aria-pressed={filter === value}
                        onClick={() => setFilter(value)}
                      >
                        {value ?? "All"}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="panel scroll">
                  {transactions.length === 0 ? (
                    <div className="state">
                      <h3>Nothing matches this filter</h3>
                      <p>
                        No transaction in the current window carries a{" "}
                        {filter?.toLowerCase()} advisory.
                      </p>
                    </div>
                  ) : (
                    <table>
                      <caption className="visually-hidden">
                        Recent transactions with each engine&rsquo;s verdict
                      </caption>
                      <thead>
                        <tr>
                          <th scope="col">Transaction</th>
                          <th scope="col">
                            Engines
                            <span className="visually-hidden">
                              {" "}
                              (model, behaviour, incident)
                            </span>
                          </th>
                          <th scope="col">Advisory</th>
                          <th scope="col" className="num">Model</th>
                          <th scope="col" className="num">Behaviour</th>
                          <th scope="col" className="num">Amount</th>
                          <th scope="col">Merchant</th>
                          <th scope="col">Customer</th>
                          <th scope="col">Why</th>
                        </tr>
                      </thead>
                      <tbody>
                        {transactions.map((row) => (
                          <tr key={row.transaction_id}>
                            <td className="id">{row.transaction_id}</td>
                            <td>
                              <EngineChips row={row} />
                            </td>
                            <td>
                              <Badge value={row.recommendation} />
                            </td>
                            <td className="num">{row.risk_score.toFixed(3)}</td>
                            <td className="num">
                              {row.behaviour_score === null
                                ? EMPTY
                                : row.behaviour_score.toFixed(3)}
                            </td>
                            <td className="num">{money.format(row.amount)}</td>
                            <td className="id">{row.merchant_id ?? EMPTY}</td>
                            <td className="id">{row.customer_id ?? EMPTY}</td>
                            <td className="detail">
                              {row.behaviour_detail &&
                              row.behaviour_detail !== "nothing unusual"
                                ? row.behaviour_detail
                                : row.reasons?.[0]?.text ?? EMPTY}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </section>

              {/* How it works: the three engines the rest of the console is
                 built around. Kicker tinted to each engine's series colour. */}
              <section className="landing-feature-section">
                <div className="section-head">
                  <h2>How it works</h2>
                  <p>three engines, each reading a different slice of the stream</p>
                </div>
                <div className="landing-grid">
                  <article className="landing-card" data-engine="model">
                    <p className="card-kicker">Model</p>
                    <h3>Looks like fraud</h3>
                    <p>
                      A gradient-boosted model scores each transaction on its own
                      features and returns an ALLOW, REVIEW, or BLOCK advisory with
                      the reasons behind it.
                    </p>
                  </article>
                  <article className="landing-card" data-engine="behaviour">
                    <p className="card-kicker">Behaviour</p>
                    <h3>Unusual for this customer</h3>
                    <p>
                      Reads a customer&rsquo;s own history &mdash; amounts, merchants,
                      cadence &mdash; and flags a transaction that breaks their
                      established pattern.
                    </p>
                  </article>
                  <article className="landing-card" data-engine="incident">
                    <p className="card-kicker">Incident</p>
                    <h3>Part of a coordinated spike</h3>
                    <p>
                      Watches a merchant&rsquo;s whole traffic for velocity surges and
                      multi-customer rings, and opens an incident when the pattern
                      holds.
                    </p>
                  </article>
                </div>
              </section>
            </>
          )}
        </main>
      </div>

      {/* --- Floating AI Merchant Assistant Chat Drawer --- */}
      <AssistantDrawer />
    </>
  );
}

function AssistantDrawer() {
  const [open, setOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([
    {
      sender: "ai",
      text: "Hello! I am your CoverPay AI Merchant Assistant. Ask me about suspicious transactions, active incidents, or financial exposure.",
    },
  ]);
  const [loading, setLoading] = useState(false);

  const handleSend = async (e) => {
    e?.preventDefault();
    if (!question.trim() || loading) return;

    const userText = question.trim();
    setQuestion("");
    setMessages((prev) => [...prev, { sender: "user", text: userText }]);
    setLoading(true);

    try {
      const res = await api.ask(userText);
      setMessages((prev) => [
        ...prev,
        { sender: "ai", text: res.answer, context: res.context },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          sender: "ai",
          text: `Unable to process request: ${err.message}. Ensure GEMINI_API_KEY is set in .env.`,
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="assistant-fab-container">
      {!open && (
        <button
          className="assistant-fab"
          onClick={() => setOpen(true)}
          aria-label="Open CoverPay AI Assistant"
        >
          Ask CoverPay AI
        </button>
      )}

      {open && (
        <div className="assistant-drawer">
          <div className="assistant-header">
            <div className="assistant-title">
              <span className="mark" aria-hidden="true">
                <i />
                <i />
                <i />
              </span>
              <div>
                <h4>CoverPay Assistant</h4>
                <p>Grounded fraud intelligence</p>
              </div>
            </div>
            <button className="ghost close-btn" onClick={() => setOpen(false)}>
              Close
            </button>
          </div>

          <div className="assistant-messages">
            {messages.map((m, idx) => (
              <div
                key={idx}
                className={`chat-bubble ${m.sender} ${m.isError ? "error" : ""}`}
              >
                <p>{m.text}</p>
                {m.context?.scope && (
                  <div className="context-tag">
                    Window: {m.context.scope.window_hours}h | Volume: ₹
                    {money.format(m.context.total_value || 0)}
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="chat-bubble ai loading">
                <span>Analyzing SQL fraud evidence...</span>
              </div>
            )}
          </div>

          <div className="assistant-quick-prompts">
            {[
              "Why did risk increase?",
              "What active incidents exist?",
              "Which txns should I review?",
            ].map((promptText) => (
              <button
                key={promptText}
                type="button"
                className="prompt-pill"
                onClick={() => {
                  setQuestion(promptText);
                }}
              >
                {promptText}
              </button>
            ))}
          </div>

          <form className="assistant-input-form" onSubmit={handleSend}>
            <input
              type="text"
              placeholder="Ask e.g. Why did risk increase?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
            <button type="submit" disabled={loading || !question.trim()}>
              Send
            </button>
          </form>
        </div>
      )}
    </div>
  );
}

function CsvUploadSection() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
      setResult(null);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await api.uploadCsv(file);
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to analyze CSV");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="csv-upload-section rise" style={{ "--i": 1, marginBottom: 28 }}>
      <div className="section-head">
        <h2>Batch CSV check</h2>
        <p>score an uploaded transaction file against the model and surface the riskiest rows</p>
      </div>

      <div className="panel upload-panel">
        <form onSubmit={handleUpload} className="upload-form">
          <div className="file-input-wrapper">
            <input
              type="file"
              accept=".csv"
              id="csv-file-input"
              onChange={handleFileChange}
              disabled={loading}
            />
            <label htmlFor="csv-file-input" className="file-label">
              {file ? file.name : "Choose a CSV file, or drop one here"}
            </label>
          </div>

          <button
            type="submit"
            className="primary-btn"
            disabled={!file || loading}
          >
            {loading ? "Scoring…" : "Run check"}
          </button>
        </form>

        {loading && (
          <div className="analysis-loading-state">
            <div className="spinner" />
            <div className="loading-text-wrapper">
              <h4>Scoring rows against the model</h4>
              <p>Running the feature pipeline and TreeSHAP contributions for each transaction.</p>
            </div>
          </div>
        )}

        {error && (
          <div className="state error-state" style={{ marginTop: 16 }}>
            <p className="error-text">{error}</p>
          </div>
        )}

        {result && (
          <div className="analysis-results rise" style={{ marginTop: 20 }}>
            <div className="results-summary-bar">
              <div className="res-stat">
                <span className="res-label">Total Transactions</span>
                <span className="res-val">{count.format(result.total_transactions)}</span>
              </div>
              <div className="res-stat alert">
                <span className="res-label">Possible Attacks / Flagged</span>
                <span className="res-val">{count.format(result.possible_attacks_flagged)}</span>
                <span className="res-sub">({result.high_risk_percentage}% of CSV)</span>
              </div>
              <div className="res-stat warning">
                <span className="res-label">Review Recommended</span>
                <span className="res-val">{count.format(result.review_recommended)}</span>
              </div>
              <div className="res-stat critical">
                <span className="res-label">Block Recommended</span>
                <span className="res-val">{count.format(result.block_recommended)}</span>
              </div>
              <div className="res-stat">
                <span className="res-label">Flagged Value</span>
                <span className="res-val">₹{money.format(result.total_value_flagged)}</span>
              </div>
            </div>

            {result.riskiest_transactions?.length > 0 && (
              <div className="top-riskiest-table-wrapper" style={{ marginTop: 16 }}>
                <h4>Highest-risk rows in {result.filename}</h4>
                <table>
                  <thead>
                    <tr>
                      <th>Txn ID</th>
                      <th>Risk Score</th>
                      <th>Level</th>
                      <th>Recommendation</th>
                      <th className="num">Amount</th>
                      <th>Primary Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.riskiest_transactions.map((tx) => (
                      <tr key={tx.transaction_id}>
                        <td className="id">{tx.transaction_id}</td>
                        <td className="num">{tx.risk_score.toFixed(3)}</td>
                        <td>
                          <Badge value={tx.risk_level} />
                        </td>
                        <td>
                          <Badge value={tx.recommendation} />
                        </td>
                        <td className="num">₹{money.format(tx.amount_inr || 0)}</td>
                        <td className="detail">{tx.reasons?.[0]?.text || "High risk feature anomaly"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
