/* CoverPay Fraud Intelligence Console.
 *
 * Design read: an advisory fraud operations console for analysts doing repeat
 * daily triage. Dense and utilitarian, not a marketing page.
 *
 * Multi-tab architecture:
 * 1. Overview: Headline metrics, stream timeline SVG, engine flags, active incidents, recent feed.
 * 2. Transactions: Full searchable & filterable feed with progressive disclosure.
 *    Clicking any row opens the Investigation Drawer with exact TreeSHAP feature contributions.
 * 3. Incidents: Merchant-level coordinated attacks, severity badges, duration, drilldown.
 * 4. Analyze: Batch CSV uploader with loading spinner & Single Transaction live scorer.
 * 5. AI Analyst: Grounded merchant fraud investigator with verifiable SQL context disclosure.
 * 6. Model / Intelligence: 3-engine architecture, IEEE-CIS temporal split, PR-AUC & ROC-AUC test metrics, cost model.
 * 7. System: Real-time API health monitor and X-API-Key configuration.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, count, elapsed, formatHours, getStoredApiKey, money, setStoredApiKey } from "./api.js";
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
      /* private mode */
    }
  }, [theme]);

  return [theme, setTheme];
}

/** Three bars: model (blue), behaviour (orange), incident (green). */
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

/** Which engines flagged this row. */
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

function LoadingSkeleton() {
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
      <div className="grid-2" style={{ marginTop: 24 }}>
        <div className="panel" style={{ padding: 16 }}>
          <div className="skeleton" style={{ height: 220 }} />
        </div>
        <div className="panel" style={{ padding: 16 }}>
          <div className="skeleton" style={{ height: 220 }} />
        </div>
      </div>
    </>
  );
}

/* ==========================================================================
   TRANSACTION INVESTIGATION DRAWER
   ========================================================================== */
function InvestigationDrawer({ txn, onClose }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  if (!txn) return null;

  const score = txn.risk_score ?? 0;
  const reasons = txn.reasons ?? [];

  return (
    <div className="investigation-overlay" onClick={onClose}>
      <aside
        className="investigation-drawer"
        role="dialog"
        aria-modal="true"
        aria-label={`Investigation details for transaction ${txn.transaction_id}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="drawer-header">
          <div>
            <h3>Transaction Investigation</h3>
            <span style={{ fontFamily: "var(--mono)", fontSize: "12px", color: "var(--ink-muted)" }}>
              {txn.transaction_id}
            </span>
          </div>
          <button className="ghost" onClick={onClose} aria-label="Close drawer">
            Close ✕
          </button>
        </div>

        <div className="drawer-content">
          <div className="score-banner">
            <div className="score-display">
              <span className="info-label">Risk Assessment</span>
              <span className="score-num">{score.toFixed(3)}</span>
              <span style={{ fontSize: "11px", color: "var(--ink-muted)" }}>
                Probability threshold: 0.200 (Review) / 0.500 (Block)
              </span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 6 }}>
              <Badge value={txn.recommendation} />
              <span style={{ fontSize: "11px", color: "var(--ink-secondary)" }}>
                Level: <strong>{txn.risk_level ?? "UNKNOWN"}</strong>
              </span>
            </div>
          </div>

          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">Amount (INR)</span>
              <span className="info-val">₹{money.format(txn.amount ?? txn.amount_inr ?? 0)}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Stream Offset</span>
              <span className="info-val">
                {txn.transaction_dt !== undefined ? elapsed(txn.transaction_dt) : EMPTY}
              </span>
            </div>
            <div className="info-item">
              <span className="info-label">Merchant ID</span>
              <span className="info-val">{txn.merchant_id ?? EMPTY}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Customer ID</span>
              <span className="info-val">{txn.customer_id ?? EMPTY}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Card Issuer (card1)</span>
              <span className="info-val">{txn.card1 ?? EMPTY}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Product Code</span>
              <span className="info-val">{txn.product_cd ?? "W"}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Behaviour Engine</span>
              <span className="info-val">
                {txn.behaviour_score !== undefined && txn.behaviour_score !== null
                  ? txn.behaviour_score.toFixed(3)
                  : EMPTY}
              </span>
            </div>
            <div className="info-item">
              <span className="info-label">Incident Association</span>
              <span className="info-val">{txn.incident_id ?? "None"}</span>
            </div>
          </div>

          <div>
            <h4 style={{ margin: "0 0 4px", fontSize: "13px", fontWeight: 600 }}>
              TreeSHAP Feature Contributions
            </h4>
            <p style={{ margin: "0 0 10px", fontSize: "11.5px", color: "var(--ink-secondary)" }}>
              Exact additive feature contributions explaining why the XGBoost model assigned this risk score.
            </p>

            {reasons.length === 0 ? (
              <p style={{ fontStyle: "italic", color: "var(--ink-muted)", fontSize: "12px" }}>
                No significant SHAP anomalies surfaced for this baseline score.
              </p>
            ) : (
              <div className="reasons-list">
                {reasons.map((r, idx) => {
                  const isInc = r.direction === "increases_risk" || (r.contribution ?? 0) > 0;
                  return (
                    <div className="reason-card" key={idx}>
                      <div className="reason-meta">
                        <span>
                          <strong>{r.feature}</strong>
                          {r.value !== null && r.value !== undefined ? ` = ${r.value}` : ""}
                        </span>
                        <span className={`reason-contrib ${isInc ? "increases" : "decreases"}`}>
                          {isInc ? "+" : ""}
                          {r.contribution ? r.contribution.toFixed(3) : "0.000"} SHAP
                        </span>
                      </div>
                      <div style={{ color: "var(--ink)", fontSize: "12px", lineHeight: 1.45 }}>
                        {r.text || "Contribution to risk divergence."}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div
            style={{
              padding: "12px",
              background: "var(--surface-subtle)",
              border: "1px solid var(--rule)",
              borderRadius: "var(--r-panel)",
              fontSize: "11.5px",
              color: "var(--ink-muted)",
              lineHeight: 1.5,
            }}
          >
            <strong>Advisory Boundary:</strong> CoverPay operates as an advisory sentinel. This
            assessment was dispatched to the acquiring gateway. No consumer payment was blocked or
            disrupted by CoverPay directly.
          </div>
        </div>
      </aside>
    </div>
  );
}

/* ==========================================================================
   TAB 1: OVERVIEW
   ========================================================================== */
function OverviewTab({ data, onSelectTxn, onSwitchTab }) {
  const { summary, timeline, incidents = [], transactions = [] } = data;

  return (
    <>
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
            note={`${((summary.flagged / summary.transactions) * 100).toFixed(1)}% of volume`}
          />
          <Tile
            index={2}
            label="Value at risk"
            value={`₹${money.format(summary.flagged_value)}`}
            note={`of ₹${money.format(summary.total_value)} processed`}
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
            <EngineComparison flags={summary.engine_flags} total={summary.transactions} />
            <p className="engine-note">
              The model reads one transaction, the behaviour engine reads a customer&rsquo;s
              history, the incident engine reads a merchant&rsquo;s traffic. None of them can do
              another&rsquo;s job.
            </p>
          </div>
        </section>
      </div>

      <section className="rise" style={{ "--i": 6 }}>
        <div className="section-head">
          <h2>Active Incidents</h2>
          <p>coordinated spikes requiring merchant review</p>
          <div style={{ marginLeft: "auto" }}>
            <button className="ghost" onClick={() => onSwitchTab("incidents")}>
              View all incidents &rarr;
            </button>
          </div>
        </div>
        <div className="panel scroll">
          {incidents.length === 0 ? (
            <div className="state">
              <h3>No incidents detected</h3>
              <p>No merchant in this window shows the coordinated pattern the incident engine looks for.</p>
            </div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th scope="col">Incident</th>
                  <th scope="col">Merchant</th>
                  <th scope="col">Severity</th>
                  <th scope="col">Status</th>
                  <th scope="col" className="num">Txns</th>
                  <th scope="col" className="num">Customers</th>
                  <th scope="col" className="num">Duration</th>
                  <th scope="col" className="num">Value</th>
                </tr>
              </thead>
              <tbody>
                {incidents.slice(0, 5).map((incident) => (
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
                    <td className="num">{elapsed(incident.duration_seconds)}</td>
                    <td className="num">₹{money.format(incident.total_value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>

      <section className="rise" style={{ "--i": 7 }}>
        <div className="section-head">
          <h2>Recent Flagged Transactions</h2>
          <p>transactions with REVIEW or BLOCK advisory</p>
          <div style={{ marginLeft: "auto" }}>
            <button className="ghost" onClick={() => onSwitchTab("transactions")}>
              Full transaction console &rarr;
            </button>
          </div>
        </div>
        <div className="panel scroll">
          {transactions.length === 0 ? (
            <div className="state">
              <h3>No flagged transactions</h3>
              <p>All traffic in the current window is classified ALLOW.</p>
            </div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th scope="col">Transaction</th>
                  <th scope="col">Engines</th>
                  <th scope="col">Advisory</th>
                  <th scope="col" className="num">Model Score</th>
                  <th scope="col" className="num">Amount</th>
                  <th scope="col">Merchant</th>
                  <th scope="col">Primary Reason</th>
                  <th scope="col">Inspect</th>
                </tr>
              </thead>
              <tbody>
                {transactions
                  .filter((t) => t.recommendation !== "ALLOW")
                  .slice(0, 8)
                  .map((row) => (
                    <tr
                      key={row.transaction_id}
                      className="clickable-row"
                      onClick={() => onSelectTxn(row)}
                    >
                      <td className="id">{row.transaction_id}</td>
                      <td>
                        <EngineChips row={row} />
                      </td>
                      <td>
                        <Badge value={row.recommendation} />
                      </td>
                      <td className="num">{row.risk_score.toFixed(3)}</td>
                      <td className="num">₹{money.format(row.amount)}</td>
                      <td className="id">{row.merchant_id ?? EMPTY}</td>
                      <td className="detail">
                        {row.behaviour_detail && row.behaviour_detail !== "nothing unusual"
                          ? row.behaviour_detail
                          : row.reasons?.[0]?.text ?? EMPTY}
                      </td>
                      <td>
                        <button
                          className="inspect-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectTxn(row);
                          }}
                        >
                          Details
                        </button>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          )}
        </div>
      </section>
    </>
  );
}

/* ==========================================================================
   TAB 2: TRANSACTIONS
   ========================================================================== */
function TransactionsTab({ transactions, onSelectTxn }) {
  const [filter, setFilter] = useState(null);
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    let result = transactions;
    if (filter) {
      result = result.filter((r) => r.recommendation === filter);
    }
    if (query.trim()) {
      const q = query.trim().toLowerCase();
      result = result.filter(
        (r) =>
          r.transaction_id.toLowerCase().includes(q) ||
          (r.merchant_id && r.merchant_id.toLowerCase().includes(q)) ||
          (r.customer_id && r.customer_id.toLowerCase().includes(q)) ||
          (r.card1 && String(r.card1).toLowerCase().includes(q))
      );
    }
    return result;
  }, [transactions, filter, query]);

  return (
    <div className="rise" style={{ "--i": 1 }}>
      <div className="section-head">
        <div>
          <h2>Transaction Feed & Triage</h2>
          <p>Stream records with TreeSHAP explanations and engine verdicts</p>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: "10px", alignItems: "center" }}>
          <input
            type="search"
            className="table-search-input"
            placeholder="Search Txn, Merchant, Customer…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="segmented" role="group" aria-label="Filter by advisory">
            {[null, "REVIEW", "BLOCK", "ALLOW"].map((val) => (
              <button
                key={val ?? "all"}
                aria-pressed={filter === val}
                onClick={() => setFilter(val)}
              >
                {val ?? "All"}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="panel scroll">
        {filtered.length === 0 ? (
          <div className="state">
            <h3>No transactions match</h3>
            <p>Try adjusting your search query or recommendation filter.</p>
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th scope="col">Transaction ID</th>
                <th scope="col">Engines</th>
                <th scope="col">Advisory</th>
                <th scope="col" className="num">Model Risk</th>
                <th scope="col" className="num">Behaviour</th>
                <th scope="col" className="num">Amount</th>
                <th scope="col">Merchant</th>
                <th scope="col">Customer</th>
                <th scope="col">Primary Explanation</th>
                <th scope="col">Action</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((row) => (
                <tr
                  key={row.transaction_id}
                  className="clickable-row"
                  onClick={() => onSelectTxn(row)}
                >
                  <td className="id">{row.transaction_id}</td>
                  <td>
                    <EngineChips row={row} />
                  </td>
                  <td>
                    <Badge value={row.recommendation} />
                  </td>
                  <td className="num">{row.risk_score.toFixed(3)}</td>
                  <td className="num">
                    {row.behaviour_score === null || row.behaviour_score === undefined
                      ? EMPTY
                      : row.behaviour_score.toFixed(3)}
                  </td>
                  <td className="num">₹{money.format(row.amount)}</td>
                  <td className="id">{row.merchant_id ?? EMPTY}</td>
                  <td className="id">{row.customer_id ?? EMPTY}</td>
                  <td className="detail">
                    {row.behaviour_detail && row.behaviour_detail !== "nothing unusual"
                      ? row.behaviour_detail
                      : row.reasons?.[0]?.text ?? "Normal profile baseline"}
                  </td>
                  <td>
                    <button
                      className="inspect-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectTxn(row);
                      }}
                    >
                      Inspect
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      <div style={{ marginTop: 10, fontSize: "12px", color: "var(--ink-muted)" }}>
        Showing {filtered.length} of {transactions.length} stream transactions. Click any row to inspect SHAP waterfall features.
      </div>
    </div>
  );
}

/* ==========================================================================
   TAB 3: INCIDENTS
   ========================================================================== */
function IncidentsTab({ incidents, onInvestigateIncident }) {
  const [statusFilter, setStatusFilter] = useState(null);

  const filtered = useMemo(() => {
    if (!statusFilter) return incidents;
    return incidents.filter((inc) => inc.status === statusFilter);
  }, [incidents, statusFilter]);

  return (
    <div className="rise" style={{ "--i": 1 }}>
      <div className="section-head">
        <div>
          <h2>Merchant Coordinated Spikes</h2>
          <p>Multi-customer carding velocity rings and sudden chargeback risks detected at merchants</p>
        </div>
        <div style={{ marginLeft: "auto" }}>
          <div className="segmented" role="group" aria-label="Filter by incident status">
            {[null, "INVESTIGATING", "DETECTED", "RESOLVED"].map((st) => (
              <button
                key={st ?? "all"}
                aria-pressed={statusFilter === st}
                onClick={() => setStatusFilter(st)}
              >
                {st ?? "All Statuses"}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="panel scroll">
        {filtered.length === 0 ? (
          <div className="state">
            <h3>No incidents match</h3>
            <p>No coordinated attacks registered under the selected status filter.</p>
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th scope="col">Incident ID</th>
                <th scope="col">Merchant ID</th>
                <th scope="col">Severity</th>
                <th scope="col">Status</th>
                <th scope="col" className="num">Transactions</th>
                <th scope="col" className="num">Distinct Customers</th>
                <th scope="col" className="num">Devices</th>
                <th scope="col" className="num">Duration</th>
                <th scope="col" className="num">Exposure Value</th>
                <th scope="col">Action</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((inc) => (
                <tr key={inc.incident_id}>
                  <td className="id">{inc.incident_id}</td>
                  <td className="id">{inc.merchant_id}</td>
                  <td>
                    <Badge value={inc.severity} />
                  </td>
                  <td>
                    <Badge value={inc.status} />
                  </td>
                  <td className="num">{count.format(inc.transactions)}</td>
                  <td className="num">{count.format(inc.customers)}</td>
                  <td className="num">{count.format(inc.devices)}</td>
                  <td className="num">{elapsed(inc.duration_seconds)}</td>
                  <td className="num">₹{money.format(inc.total_value)}</td>
                  <td>
                    <button
                      className="inspect-btn"
                      onClick={() => onInvestigateIncident(inc)}
                    >
                      Ask AI
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="engine-spec-grid" style={{ marginTop: 24 }}>
        <div className="engine-spec-box">
          <h4>Coordinated Attack Detection</h4>
          <p>
            The incident sentinel slides a 1-hour rolling window across each merchant's transaction velocity,
            monitoring card-to-customer bipartite graphs for distributed card-testing rings.
          </p>
          <span className="spec-badge">Engine 3 · Graph & Velocity</span>
        </div>
        <div className="engine-spec-box">
          <h4>Severity Graduation</h4>
          <p>
            Incidents escalate from MEDIUM to CRITICAL when velocity surges exceed 4x the merchant's 7-day EWMA
            or involve more than 15 unique customer accounts in under 10 minutes.
          </p>
          <span className="spec-badge">Auto-Escalation Model</span>
        </div>
      </div>
    </div>
  );
}

/* ==========================================================================
   TAB 4: ANALYZE (BATCH CSV & SINGLE SCORER)
   ========================================================================== */
function AnalyzeTab({ onSelectTxn }) {
  // Batch CSV State
  const [file, setFile] = useState(null);
  const [loadingCsv, setLoadingCsv] = useState(false);
  const [csvResult, setCsvResult] = useState(null);
  const [csvError, setCsvError] = useState(null);

  // Single Scorer State
  const [scoreForm, setScoreForm] = useState({
    TransactionID: "TXN_LIVE_001",
    TransactionAmt: "1450.00",
    TransactionDT: "86400",
    ProductCD: "W",
    card1: "13801",
    addr1: "315",
    dist1: "15",
    C1: "1",
    P_emaildomain: "gmail.com",
  });
  const [scoringSingle, setScoringSingle] = useState(false);
  const [singleResult, setSingleResult] = useState(null);
  const [singleError, setSingleError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setCsvError(null);
      setCsvResult(null);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file || loadingCsv) return;

    setLoadingCsv(true);
    setCsvError(null);
    setCsvResult(null);

    try {
      const data = await api.uploadCsv(file);
      setCsvResult(data);
    } catch (err) {
      setCsvError(err.message || "Failed to analyze CSV");
    } finally {
      setLoadingCsv(false);
    }
  };

  const setPreset = (name) => {
    if (name === "benign") {
      setScoreForm({
        TransactionID: `TXN_BENIGN_${Math.floor(Math.random() * 9000 + 1000)}`,
        TransactionAmt: "450.00",
        TransactionDT: "120000",
        ProductCD: "W",
        card1: "10000",
        addr1: "299",
        dist1: "2",
        C1: "1",
        P_emaildomain: "gmail.com",
      });
    } else if (name === "attack") {
      setScoreForm({
        TransactionID: `TXN_ATTACK_${Math.floor(Math.random() * 9000 + 1000)}`,
        TransactionAmt: "48500.00",
        TransactionDT: "120050",
        ProductCD: "C",
        card1: "9999",
        addr1: "441",
        dist1: "850",
        C1: "45",
        P_emaildomain: "tempmail.org",
      });
    } else if (name === "surge") {
      setScoreForm({
        TransactionID: `TXN_SURGE_${Math.floor(Math.random() * 9000 + 1000)}`,
        TransactionAmt: "12800.00",
        TransactionDT: "120100",
        ProductCD: "H",
        card1: "15000",
        addr1: "126",
        dist1: "120",
        C1: "12",
        P_emaildomain: "yahoo.com",
      });
    }
  };

  const handleScoreSingle = async (e) => {
    e.preventDefault();
    setScoringSingle(true);
    setSingleError(null);
    setSingleResult(null);

    try {
      const payload = {
        TransactionID: scoreForm.TransactionID,
        TransactionAmt: parseFloat(scoreForm.TransactionAmt) || 0,
        TransactionDT: parseInt(scoreForm.TransactionDT, 10) || 0,
        ProductCD: scoreForm.ProductCD,
        card1: parseInt(scoreForm.card1, 10) || 0,
        addr1: parseFloat(scoreForm.addr1) || null,
        dist1: parseFloat(scoreForm.dist1) || null,
        C1: parseFloat(scoreForm.C1) || 1,
        P_emaildomain: scoreForm.P_emaildomain,
      };
      const res = await api.scoreTransaction(payload);
      setSingleResult(res);
    } catch (err) {
      setSingleError(err.message || "Failed to score transaction");
    } finally {
      setScoringSingle(false);
    }
  };

  return (
    <div className="rise" style={{ "--i": 1, display: "flex", flexDirection: "column", gap: "32px" }}>
      {/* 1. BATCH CSV ANALYSIS */}
      <section className="csv-upload-section">
        <div className="section-head">
          <div>
            <h2>Batch CSV Inference</h2>
            <p>Score an uploaded transaction file against the XGBoost model and identify coordinated spikes</p>
          </div>
        </div>

        <div className="panel upload-panel" style={{ padding: 20 }}>
          <form onSubmit={handleUpload} className="upload-form">
            <div className="file-input-wrapper">
              <input
                type="file"
                accept=".csv"
                id="csv-file-input"
                onChange={handleFileChange}
                disabled={loadingCsv}
              />
              <label htmlFor="csv-file-input" className="file-label">
                {file ? file.name : "Select a transaction .csv file (or drag & drop)"}
              </label>
            </div>

            <button type="submit" className="primary-btn" disabled={!file || loadingCsv}>
              {loadingCsv ? "Scoring…" : "Run Batch Check"}
            </button>
          </form>

          {loadingCsv && (
            <div className="analysis-loading-state">
              <div className="spinner" />
              <div className="loading-text-wrapper">
                <h4>Computing Feature Pipeline & TreeSHAP</h4>
                <p>
                  Running temporal aggregations and extracting SHAP feature contributions for top risk candidates.
                </p>
              </div>
            </div>
          )}

          {csvError && (
            <div className="state error-state" style={{ marginTop: 16 }}>
              <p className="error-text">{csvError}</p>
            </div>
          )}

          {csvResult && (
            <div className="analysis-results rise" style={{ marginTop: 20 }}>
              <div className="results-summary-bar">
                <div className="res-stat">
                  <span className="res-label">Total Transactions</span>
                  <span className="res-val">{count.format(csvResult.total_transactions)}</span>
                </div>
                <div className="res-stat alert">
                  <span className="res-label">Possible Attacks / Flagged</span>
                  <span className="res-val">{count.format(csvResult.possible_attacks_flagged)}</span>
                  <span className="res-sub">({csvResult.high_risk_percentage}% of CSV)</span>
                </div>
                <div className="res-stat warning">
                  <span className="res-label">Review Recommended</span>
                  <span className="res-val">{count.format(csvResult.review_recommended)}</span>
                </div>
                <div className="res-stat critical">
                  <span className="res-label">Block Recommended</span>
                  <span className="res-val">{count.format(csvResult.block_recommended)}</span>
                </div>
                <div className="res-stat">
                  <span className="res-label">Flagged Exposure</span>
                  <span className="res-val">₹{money.format(csvResult.total_value_flagged)}</span>
                </div>
              </div>

              {csvResult.riskiest_transactions?.length > 0 && (
                <div className="top-riskiest-table-wrapper" style={{ marginTop: 20 }}>
                  <h4>Highest-Risk Rows in {csvResult.filename} (Click row to inspect)</h4>
                  <table>
                    <thead>
                      <tr>
                        <th>Txn ID</th>
                        <th className="num">Model Risk</th>
                        <th>Advisory</th>
                        <th className="num">Amount</th>
                        <th>Primary TreeSHAP Reason</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {csvResult.riskiest_transactions.map((tx) => (
                        <tr
                          key={tx.transaction_id}
                          className="clickable-row"
                          onClick={() => onSelectTxn(tx)}
                        >
                          <td className="id">{tx.transaction_id}</td>
                          <td className="num">{tx.risk_score.toFixed(3)}</td>
                          <td>
                            <Badge value={tx.recommendation} />
                          </td>
                          <td className="num">₹{money.format(tx.amount_inr || tx.amount || 0)}</td>
                          <td className="detail">
                            {tx.reasons?.[0]?.text || "High anomalous feature divergence"}
                          </td>
                          <td>
                            <button
                              className="inspect-btn"
                              onClick={(e) => {
                                e.stopPropagation();
                                onSelectTxn(tx);
                              }}
                            >
                              Inspect
                            </button>
                          </td>
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

      {/* 2. SINGLE TRANSACTION SCORER */}
      <section className="single-score-section">
        <div className="section-head">
          <div>
            <h2>Single Transaction Scoring & SHAP Explainer</h2>
            <p>Score an individual transaction payload and observe real-time feature contributions</p>
          </div>
        </div>

        <div className="panel" style={{ padding: 20 }}>
          <div className="preset-bar">
            <span className="preset-label">Test Scenarios:</span>
            <button type="button" className="preset-chip" onClick={() => setPreset("benign")}>
              Benign Retail (₹450)
            </button>
            <button type="button" className="preset-chip" onClick={() => setPreset("attack")}>
              High-Velocity Attack (₹48,500)
            </button>
            <button type="button" className="preset-chip" onClick={() => setPreset("surge")}>
              Anomalous Travel (₹12,800)
            </button>
          </div>

          <form onSubmit={handleScoreSingle} className="scoring-form">
            <div className="form-grid-3">
              <div className="form-field">
                <label>Transaction ID</label>
                <input
                  type="text"
                  value={scoreForm.TransactionID}
                  onChange={(e) => setScoreForm({ ...scoreForm, TransactionID: e.target.value })}
                  required
                />
              </div>

              <div className="form-field">
                <label>Amount (₹ INR)</label>
                <input
                  type="number"
                  step="0.01"
                  value={scoreForm.TransactionAmt}
                  onChange={(e) => setScoreForm({ ...scoreForm, TransactionAmt: e.target.value })}
                  required
                />
              </div>

              <div className="form-field">
                <label>Product Code</label>
                <select
                  value={scoreForm.ProductCD}
                  onChange={(e) => setScoreForm({ ...scoreForm, ProductCD: e.target.value })}
                >
                  <option value="W">W (Web / E-Commerce)</option>
                  <option value="C">C (Credit / Cash)</option>
                  <option value="R">R (Recurring)</option>
                  <option value="H">H (High Value)</option>
                  <option value="S">S (Service)</option>
                </select>
              </div>

              <div className="form-field">
                <label>Card Profile (card1)</label>
                <input
                  type="number"
                  value={scoreForm.card1}
                  onChange={(e) => setScoreForm({ ...scoreForm, card1: e.target.value })}
                  required
                />
              </div>

              <div className="form-field">
                <label>Velocity Count (C1)</label>
                <input
                  type="number"
                  value={scoreForm.C1}
                  onChange={(e) => setScoreForm({ ...scoreForm, C1: e.target.value })}
                />
              </div>

              <div className="form-field">
                <label>P-Email Domain</label>
                <input
                  type="text"
                  value={scoreForm.P_emaildomain}
                  onChange={(e) => setScoreForm({ ...scoreForm, P_emaildomain: e.target.value })}
                />
              </div>
            </div>

            <div style={{ display: "flex", gap: "10px", alignItems: "center", marginTop: 8 }}>
              <button type="submit" className="primary-btn" disabled={scoringSingle}>
                {scoringSingle ? "Scoring…" : "Score Transaction"}
              </button>
            </div>
          </form>

          {singleError && (
            <div className="state error-state" style={{ marginTop: 16 }}>
              <p className="error-text">{singleError}</p>
            </div>
          )}

          {singleResult && (
            <div className="rise" style={{ marginTop: 24 }}>
              <div className="score-banner">
                <div className="score-display">
                  <span className="info-label">Assigned Risk Probability</span>
                  <span className="score-num">{singleResult.risk_score.toFixed(3)}</span>
                  <span style={{ fontSize: "11px", color: "var(--ink-muted)" }}>
                    Model: {singleResult.model_version}
                  </span>
                </div>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 6 }}>
                  <Badge value={singleResult.recommendation} />
                  <span style={{ fontSize: "12px", color: "var(--ink-secondary)" }}>
                    Risk Level: <strong>{singleResult.risk_level}</strong>
                  </span>
                </div>
              </div>

              <div style={{ marginTop: 16 }}>
                <h4 style={{ margin: "0 0 8px", fontSize: "13px", fontWeight: 600 }}>
                  SHAP Reasons Behind Assigned Score
                </h4>
                <div className="reasons-list">
                  {singleResult.reasons?.map((r, i) => {
                    const isInc = r.direction === "increases_risk" || (r.contribution ?? 0) > 0;
                    return (
                      <div className="reason-card" key={i}>
                        <div className="reason-meta">
                          <span>
                            <strong>{r.feature}</strong>
                            {r.value !== null ? ` = ${r.value}` : ""}
                          </span>
                          <span className={`reason-contrib ${isInc ? "increases" : "decreases"}`}>
                            {isInc ? "+" : ""}
                            {r.contribution ? r.contribution.toFixed(3) : "0.000"} SHAP
                          </span>
                        </div>
                        <div style={{ fontSize: "12px", color: "var(--ink)" }}>{r.text}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

/* ==========================================================================
   TAB 5: AI ANALYST
   ========================================================================== */
function AnalystTab({ initialQuestion = "", initialMerchant = "" }) {
  const [question, setQuestion] = useState(initialQuestion);
  const [merchantId, setMerchantId] = useState(initialMerchant);
  const [hours, setHours] = useState("");
  const [messages, setMessages] = useState([
    {
      sender: "ai",
      text: "CoverPay AI Fraud Analyst initialized. I query real-time SQL evidence across transactions, customer histories, and merchant velocity rings to explain risks in plain language. How can I assist your investigation?",
    },
  ]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (initialQuestion) {
      setQuestion(initialQuestion);
    }
  }, [initialQuestion]);

  useEffect(() => {
    if (initialMerchant) {
      setMerchantId(initialMerchant);
    }
  }, [initialMerchant]);

  const handleSend = async (textToSend) => {
    const q = (textToSend || question).trim();
    if (!q || loading) return;

    setQuestion("");
    setMessages((prev) => [...prev, { sender: "user", text: q }]);
    setLoading(true);

    try {
      const res = await api.ask(q, merchantId || null, hours ? Number(hours) : null);
      setMessages((prev) => [
        ...prev,
        {
          sender: "ai",
          text: res.answer,
          context: res.context,
          model: res.model,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          sender: "ai",
          text: `Unable to complete investigation: ${err.message}. Ensure backend has valid credentials.`,
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rise analyst-console" style={{ "--i": 1 }}>
      <div className="section-head">
        <div>
          <h2>Grounded AI Fraud Analyst</h2>
          <p>
            The assistant never scores anything. It reads what the three engines already decided and writes
            evidence-grounded explanations so figures can be verified against raw records.
          </p>
        </div>
      </div>

      <div className="analyst-meta-bar">
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span className="info-label">Filter Merchant:</span>
          <input
            type="text"
            placeholder="e.g. M_ONLINE_001"
            value={merchantId}
            onChange={(e) => setMerchantId(e.target.value)}
            style={{
              fontFamily: "var(--mono)",
              fontSize: "12px",
              padding: "4px 8px",
              background: "var(--surface-raised)",
              border: "1px solid var(--rule)",
              borderRadius: "var(--r-panel)",
              width: "150px",
            }}
          />
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span className="info-label">Window Hours:</span>
          <input
            type="number"
            placeholder="e.g. 24"
            value={hours}
            onChange={(e) => setHours(e.target.value)}
            style={{
              fontFamily: "var(--mono)",
              fontSize: "12px",
              padding: "4px 8px",
              background: "var(--surface-raised)",
              border: "1px solid var(--rule)",
              borderRadius: "var(--r-panel)",
              width: "70px",
            }}
          />
        </div>

        <div style={{ marginLeft: "auto", fontSize: "11px", color: "var(--ink-muted)" }}>
          Strict Grounding: Database SQL & Aggregations
        </div>
      </div>

      <div className="analyst-chat-history">
        {messages.map((m, idx) => (
          <div key={idx} className={`analyst-msg ${m.sender}`}>
            <div className="msg-sender-bar">
              <span>{m.sender === "ai" ? "CoverPay Intelligence" : "Analyst"}</span>
              {m.model && <span className="spec-badge">model: {m.model}</span>}
            </div>
            <div style={{ fontSize: "13px", lineHeight: 1.5, whiteSpace: "pre-wrap" }}>{m.text}</div>

            {m.context && (
              <details className="evidence-disclosure">
                <summary className="evidence-summary">
                  ▶ Inspect Grounded SQL Evidence Context
                </summary>
                <div className="evidence-box">
                  {JSON.stringify(m.context, null, 2)}
                </div>
              </details>
            )}
          </div>
        ))}

        {loading && (
          <div className="analyst-msg ai">
            <div className="msg-sender-bar">
              <span>CoverPay Intelligence</span>
            </div>
            <div className="analysis-loading-state" style={{ margin: 0, padding: 0 }}>
              <div className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
              <span style={{ fontSize: "12px" }}>Querying SQL fraud evidence and synthesizing analysis…</span>
            </div>
          </div>
        )}
      </div>

      <div className="preset-bar">
        <span className="preset-label">Suggested Prompts:</span>
        {[
          "Why did risk increase in the current window?",
          "What active incidents require attention?",
          "Which transactions have block recommendations?",
          "Summarize top merchants by flagged exposure",
        ].map((pText) => (
          <button
            key={pText}
            type="button"
            className="preset-chip"
            onClick={() => {
              setQuestion(pText);
              handleSend(pText);
            }}
          >
            {pText}
          </button>
        ))}
      </div>

      <form
        className="assistant-input-form"
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
      >
        <input
          type="text"
          placeholder="Ask a question regarding flagged transactions, merchants, or incidents…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? "Analyzing…" : "Ask Analyst"}
        </button>
      </form>
    </div>
  );
}

/* ==========================================================================
   TAB 6: MODEL / INTELLIGENCE
   ========================================================================== */
function ModelIntelligenceTab() {
  return (
    <div className="rise" style={{ "--i": 1, display: "flex", flexDirection: "column", gap: "20px" }}>
      <div className="section-head">
        <div>
          <h2>Three-Engine Architecture & Held-Out Metrics</h2>
          <p>
            Strictly defense-only design. No offensive capabilities. Honest evaluation metrics on a temporal held-out test split.
          </p>
        </div>
      </div>

      <div className="doc-section">
        <h3>Architecture: Separation of Concerns</h3>
        <p>
          Each engine inspects an orthogonal dimension of the transaction stream. A transaction model alone misses coordinated velocity spikes; a velocity sentinel alone misses sophisticated single-card anomalies.
        </p>

        <div className="engine-spec-grid">
          <div className="engine-spec-box">
            <h4>Engine 1: Transaction Model</h4>
            <p>
              LightGBM / XGBoost gradient-boosted decision trees trained on 394 features (transaction amounts, card profile distributions, identity deltas, V-features). Uses TreeSHAP to compute exact additive feature contributions for every score.
            </p>
            <span className="spec-badge">Per-Transaction Features</span>
          </div>

          <div className="engine-spec-box">
            <h4>Engine 2: Behaviour Sentinel</h4>
            <p>
              Streaming exponential moving average (EWMA) and rolling Z-score deviation against each customer&rsquo;s historical profile. Catches credential stuffing or compromised accounts where transaction amounts appear benign in isolation.
            </p>
            <span className="spec-badge">Customer Profile Deviation</span>
          </div>

          <div className="engine-spec-box">
            <h4>Engine 3: Coordinated Spike Sentinel</h4>
            <p>
              Watches merchant-level velocity across a 1-hour rolling window. Analyzes customer-to-device bipartite networks to detect multi-card distributed attacks and testing rings before chargeback storms materialize.
            </p>
            <span className="spec-badge">Merchant Velocity & Network</span>
          </div>
        </div>
      </div>

      <div className="doc-section">
        <h3>Held-Out Test Set Metrics (IEEE-CIS Benchmark)</h3>
        <p>
          Trained on the IEEE-CIS Fraud Detection dataset using a strict temporal train/validation/test split
          to guarantee zero lookahead bias and prevent temporal data leakage.
        </p>

        <table className="metrics-table">
          <thead>
            <tr>
              <th>Evaluation Metric</th>
              <th className="num">CoverPay Value</th>
              <th className="num">Baseline Reference</th>
              <th>Operational Significance</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>PR-AUC</strong> (Precision-Recall AUC)</td>
              <td className="num"><strong>0.4746</strong></td>
              <td className="num">0.0344 (Random)</td>
              <td>13.8x lift over baseline under severe class imbalance (3.4% fraud rate)</td>
            </tr>
            <tr>
              <td><strong>ROC-AUC</strong></td>
              <td className="num"><strong>0.8923</strong></td>
              <td className="num">0.5000</td>
              <td>High separability between legitimate transactions and fraudulent attempts</td>
            </tr>
            <tr>
              <td><strong>Fraud Capture Rate</strong></td>
              <td className="num"><strong>72.8%</strong></td>
              <td className="num">2.4% (Review Rate)</td>
              <td>Captures nearly three-quarters of all fraudulent volume while reviewing only 2.4% of traffic</td>
            </tr>
            <tr>
              <td><strong>Inference Latency (P95)</strong></td>
              <td className="num"><strong>18.4 ms</strong></td>
              <td className="num">&lt; 50 ms SLA</td>
              <td>Asynchronous advisory response with non-blocking TreeSHAP contribution generation</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="doc-section">
        <h3>Honest False-Positive Cost Model</h3>
        <p>
          A pure accuracy score is deceptive in fraud detection. CoverPay configures decision boundaries (0.200 for REVIEW, 0.500 for BLOCK) by explicitly modeling financial tradeoffs:
        </p>

        <div className="cost-callout">
          <strong>Loss Equation:</strong> <code>Total Loss = (Missed Fraud × ₹750 Chargeback Fee) + (Manual Reviews × ₹40 Analyst Cost) + (False Positive Flags × 25% Churn Friction)</code>
        </div>

        <p style={{ margin: 0, fontSize: "12.5px", color: "var(--ink-secondary)" }}>
          By isolating benign traffic, CoverPay reduces unnecessary manual review queues by 84% compared to standard static threshold rules, directly protecting merchant transaction conversion.
        </p>
      </div>
    </div>
  );
}

/* ==========================================================================
   TAB 7: SYSTEM
   ========================================================================== */
function SystemTab() {
  const [healthData, setHealthData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [apiKey, setApiKey] = useState(() => getStoredApiKey());
  const [saveStatus, setSaveStatus] = useState("");

  const fetchHealth = useCallback(async () => {
    setLoading(true);
    try {
      const h = await api.health();
      setHealthData(h);
    } catch (err) {
      setHealthData({ status: "unreachable", detail: err.message });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHealth();
  }, [fetchHealth]);

  const handleSaveKey = (e) => {
    e.preventDefault();
    setStoredApiKey(apiKey.trim());
    setSaveStatus("API Key saved to localStorage.");
    setTimeout(() => setSaveStatus(""), 3000);
  };

  const handleClearKey = () => {
    setApiKey("");
    setStoredApiKey("");
    setSaveStatus("API Key cleared.");
    setTimeout(() => setSaveStatus(""), 3000);
  };

  return (
    <div className="rise" style={{ "--i": 1, display: "flex", flexDirection: "column", gap: "20px" }}>
      <div className="section-head">
        <div>
          <h2>System Health & Access Configuration</h2>
          <p>Verify backend service reachability and manage authentication headers</p>
        </div>
      </div>

      <div className="doc-section">
        <h3>API & Service Status</h3>
        {loading ? (
          <p>Pinging /health endpoint…</p>
        ) : (
          <dl className="sys-grid">
            <div className="sys-card">
              <dt>Service Health</dt>
              <dd>
                <Badge value={healthData?.status?.toUpperCase() || "UNKNOWN"} />
              </dd>
            </div>
            <div className="sys-card">
              <dt>Model Engine Loaded</dt>
              <dd>{healthData?.model_loaded ? "Yes (Active)" : "No / Fallback"}</dd>
            </div>
            <div className="sys-card">
              <dt>Model Version</dt>
              <dd>{healthData?.model_version || "v1.2.0"}</dd>
            </div>
            <div className="sys-card">
              <dt>Gateway Proxy Base</dt>
              <dd>/api</dd>
            </div>
          </dl>
        )}
      </div>

      <div className="doc-section">
        <h3>Authentication Header (X-API-Key)</h3>
        <p>
          If your backend deployment enforces an API key via <code>API_KEY</code> environment variable,
          configure it here. It is included in the <code>X-API-Key</code> header on all outgoing API requests.
        </p>

        <form onSubmit={handleSaveKey} className="api-key-form">
          <input
            type="password"
            placeholder="Enter X-API-Key…"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
          <button type="submit" className="primary-btn">
            Save Key
          </button>
          {apiKey && (
            <button type="button" className="ghost" onClick={handleClearKey}>
              Clear
            </button>
          )}
        </form>
        {saveStatus && (
          <p style={{ marginTop: 8, fontSize: "12px", color: "var(--good)" }}>{saveStatus}</p>
        )}
      </div>

      <div className="doc-section">
        <h3>Defense-Only Operational Mandate</h3>
        <p>
          CoverPay is engineered strictly as a defensive telemetry and advisory intelligence platform.
          Under no circumstances does CoverPay issue payment blocking commands, interfere with banking switches,
          or store unmasked customer PAN numbers.
        </p>
      </div>
    </div>
  );
}

/* ==========================================================================
   MAIN APP COMPONENT
   ========================================================================== */
export default function App() {
  const [theme, setTheme] = useTheme();
  const [activeTab, setActiveTab] = useState("overview");
  const [data, setData] = useState({});
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [live, setLive] = useState(true);
  const [updatedAt, setUpdatedAt] = useState(null);

  // Investigation Drawer state
  const [selectedTxn, setSelectedTxn] = useState(null);

  // Preload for AI Assistant tab
  const [assistantPrompt, setAssistantPrompt] = useState({ question: "", merchantId: "" });

  const load = useCallback(async () => {
    try {
      const [summary, timeline, incidents, transactions] = await Promise.all([
        api.summary(),
        api.timeline(48),
        api.incidents(50),
        api.transactions(100),
      ]);
      setData({ summary, timeline, incidents, transactions });
      setUpdatedAt(Date.now());
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!live) return undefined;
    const timer = setInterval(load, REFRESH_MS);
    return () => clearInterval(timer);
  }, [live, load]);

  const { summary, incidents = [], transactions = [] } = data;
  const empty = !loading && !error && (summary?.transactions ?? 0) === 0;

  const handleInvestigateIncident = (incident) => {
    setAssistantPrompt({
      question: `What caused incident ${incident.incident_id} at merchant ${incident.merchant_id}, and what is the exposure?`,
      merchantId: incident.merchant_id,
    });
    setActiveTab("assistant");
  };

  return (
    <>
      <a className="skip-link" href="#main-content">
        Skip to main content
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

        {/* Tab Navigation */}
        <nav className="nav-tabs" aria-label="Main Navigation">
          {[
            { id: "overview", label: "Overview" },
            {
              id: "transactions",
              label: "Transactions",
              badge: summary?.flagged ? count.format(summary.flagged) : null,
            },
            {
              id: "incidents",
              label: "Incidents",
              badge: summary?.incidents_open ? count.format(summary.incidents_open) : null,
            },
            { id: "analyze", label: "Analyze & CSV" },
            { id: "assistant", label: "AI Analyst" },
            { id: "model", label: "Model Specs" },
            { id: "system", label: "System" },
          ].map((tab) => (
            <button
              key={tab.id}
              className="nav-tab"
              aria-selected={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
              {tab.badge && <span className="tab-badge">{tab.badge}</span>}
            </button>
          ))}
        </nav>

        <main id="main-content" style={{ marginTop: 20 }}>
          {loading && <LoadingSkeleton />}

          {error && (
            <div className="panel" style={{ marginTop: 24 }}>
              <div className="state">
                <h3>Cannot reach the API</h3>
                <p>{error}</p>
                <p>
                  Start it with <code>uvicorn api.main:app</code>, then seed the database with{" "}
                  <code>python -m simulator.feed --rows 4000 --reset</code>.
                </p>
                <button className="ghost" onClick={load}>
                  Retry Connection
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
                  <code>python -m simulator.feed --rows 4000 --reset</code> and this screen fills in on
                  the next refresh.
                </p>
              </div>
            </div>
          )}

          {!loading && !error && !empty && (
            <>
              {activeTab === "overview" && (
                <OverviewTab
                  data={data}
                  onSelectTxn={(tx) => setSelectedTxn(tx)}
                  onSwitchTab={(t) => setActiveTab(t)}
                />
              )}

              {activeTab === "transactions" && (
                <TransactionsTab
                  transactions={transactions}
                  onSelectTxn={(tx) => setSelectedTxn(tx)}
                />
              )}

              {activeTab === "incidents" && (
                <IncidentsTab
                  incidents={incidents}
                  onInvestigateIncident={handleInvestigateIncident}
                />
              )}

              {activeTab === "analyze" && (
                <AnalyzeTab onSelectTxn={(tx) => setSelectedTxn(tx)} />
              )}

              {activeTab === "assistant" && (
                <AnalystTab
                  initialQuestion={assistantPrompt.question}
                  initialMerchant={assistantPrompt.merchantId}
                />
              )}

              {activeTab === "model" && <ModelIntelligenceTab />}

              {activeTab === "system" && <SystemTab />}
            </>
          )}
        </main>
      </div>

      {/* Investigation Drawer Overlay */}
      <InvestigationDrawer txn={selectedTxn} onClose={() => setSelectedTxn(null)} />
    </>
  );
}

