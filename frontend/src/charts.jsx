/* Charts, drawn as plain SVG.
 *
 * No charting library: two forms are needed and both are a path and some ticks,
 * so a dependency would cost more than it saves.
 *
 * Built to the data-viz rules: 2px lines, a single y-axis (never two scales),
 * recessive grid, a legend whenever two series are present, direct labels, a
 * crosshair + tooltip hover layer, and colours from the validated palette.
 */
import { useCallback, useMemo, useRef, useState } from "react";
import { count, elapsed, money } from "./api.js";

const W = 760;
const H = 220;
const PAD = { top: 14, right: 16, bottom: 26, left: 44 };

function niceTicks(max, steps = 4) {
  if (max <= 0) return [0];
  const raw = max / steps;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step =
    [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) || magnitude * 10;
  const out = [];
  for (let v = 0; v <= max + step * 0.001; v += step) out.push(v);
  return out;
}

const swatch = (colour) => ({
  width: 8,
  height: 8,
  borderRadius: 2,
  background: colour,
  display: "inline-block",
  flex: "none",
});

/**
 * Transaction volume against flagged volume over the stream's elapsed time.
 *
 * Both series are counts of transactions, so they legitimately share one axis.
 * Two different measures would need two charts, never a second y-scale.
 */
export function TimelineChart({ data }) {
  const svgRef = useRef(null);
  const [hover, setHover] = useState(null);

  const buckets = data?.buckets ?? [];
  const geometry = useMemo(() => {
    if (!buckets.length) return null;
    const max = Math.max(...buckets.map((b) => b.transactions), 1);
    const ticks = niceTicks(max);
    const top = ticks[ticks.length - 1];
    const innerW = W - PAD.left - PAD.right;
    const innerH = H - PAD.top - PAD.bottom;
    const x = (i) =>
      PAD.left + (buckets.length === 1 ? innerW / 2 : (i / (buckets.length - 1)) * innerW);
    const y = (v) => PAD.top + innerH - (v / top) * innerH;
    return { ticks, x, y };
  }, [buckets]);

  const onMove = useCallback(
    (event) => {
      if (!svgRef.current || buckets.length === 0) return;
      const box = svgRef.current.getBoundingClientRect();
      const ratio = (event.clientX - box.left) / box.width;
      const index = Math.round(ratio * (buckets.length - 1));
      setHover({
        index: Math.max(0, Math.min(buckets.length - 1, index)),
        x: event.clientX,
        y: event.clientY,
      });
    },
    [buckets.length]
  );

  if (!geometry) {
    return <p className="state">No transactions yet.</p>;
  }

  const line = (key) =>
    buckets
      .map((b, i) => `${i === 0 ? "M" : "L"}${geometry.x(i)},${geometry.y(b[key])}`)
      .join(" ");
  const area = `${line("flagged")} L${geometry.x(buckets.length - 1)},${geometry.y(
    0
  )} L${geometry.x(0)},${geometry.y(0)} Z`;
  const active = hover ? buckets[hover.index] : null;

  return (
    <>
      <div className="chart-wrap">
        <svg
          ref={svgRef}
          className="chart"
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Transaction volume across ${buckets.length} time buckets, with flagged transactions overlaid`}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        >
          {geometry.ticks.map((t) => (
            <g key={t}>
              <line
                className="gridline"
                x1={PAD.left}
                x2={W - PAD.right}
                y1={geometry.y(t)}
                y2={geometry.y(t)}
              />
              <text className="tick" x={PAD.left - 8} y={geometry.y(t) + 4} textAnchor="end">
                {count.format(t)}
              </text>
            </g>
          ))}

          <line
            className="baseline"
            x1={PAD.left}
            x2={W - PAD.right}
            y1={geometry.y(0)}
            y2={geometry.y(0)}
          />

          {/* Flagged sits under total: it is always the smaller series. */}
          <path d={area} fill="var(--series-2)" opacity="0.14" />
          <path d={line("transactions")} fill="none" stroke="var(--series-1)" strokeWidth="2" />
          <path d={line("flagged")} fill="none" stroke="var(--series-2)" strokeWidth="2" />

          {[0, Math.floor(buckets.length / 2), buckets.length - 1].map((i) => (
            <text key={i} className="tick" x={geometry.x(i)} y={H - 8} textAnchor="middle">
              +{elapsed(buckets[i].hours * 3600)}
            </text>
          ))}

          {active && (
            <g>
              <line
                className="baseline"
                x1={geometry.x(hover.index)}
                x2={geometry.x(hover.index)}
                y1={PAD.top}
                y2={geometry.y(0)}
              />
              <circle
                cx={geometry.x(hover.index)}
                cy={geometry.y(active.transactions)}
                r="4"
                fill="var(--series-1)"
                stroke="var(--surface)"
                strokeWidth="2"
              />
              <circle
                cx={geometry.x(hover.index)}
                cy={geometry.y(active.flagged)}
                r="4"
                fill="var(--series-2)"
                stroke="var(--surface)"
                strokeWidth="2"
              />
            </g>
          )}
        </svg>
      </div>

      <ul className="legend">
        <li>
          <i style={{ background: "var(--series-1)" }} />
          All transactions
        </li>
        <li>
          <i style={{ background: "var(--series-2)" }} />
          Flagged (REVIEW or BLOCK)
        </li>
      </ul>

      {active && (
        <div
          className="tooltip"
          style={{
            left: Math.min(hover.x + 14, window.innerWidth - 200),
            top: Math.min(hover.y + 14, window.innerHeight - 160),
          }}
        >
          <div className="head">+{elapsed(active.hours * 3600)} elapsed</div>
          <dl>
            <dt>
              <i style={swatch("var(--series-1)")} />
              Transactions
            </dt>
            <dd>{count.format(active.transactions)}</dd>
            <dt>
              <i style={swatch("var(--series-2)")} />
              Flagged
            </dt>
            <dd>{count.format(active.flagged)}</dd>
            <dt>Value</dt>
            <dd>{money.format(active.value)}</dd>
            <dt>Flagged value</dt>
            <dd>{money.format(active.flagged_value)}</dd>
          </dl>
        </div>
      )}
    </>
  );
}

const ENGINES = [
  { key: "model", name: "Model", colour: "var(--series-1)", note: "looks like fraud" },
  {
    key: "behaviour",
    name: "Behaviour",
    colour: "var(--series-2)",
    note: "unusual for this customer",
  },
  {
    key: "incident",
    name: "Incident",
    colour: "var(--series-3)",
    note: "part of a coordinated spike",
  },
];

/**
 * How much traffic each engine flags.
 *
 * The point the whole architecture rests on: each engine sees a different kind
 * of fraud, so these three counts are not interchangeable.
 */
export function EngineComparison({ flags, total }) {
  const max = Math.max(...ENGINES.map((e) => flags?.[e.key] ?? 0), 1);

  return (
    <dl className="engine-rows">
      {ENGINES.map((engine) => {
        const value = flags?.[engine.key] ?? 0;
        const share = total ? (value / total) * 100 : 0;
        return (
          <div className="engine-row" key={engine.key}>
            <dt className="name">
              {engine.name}
              <span className="visually-hidden"> ({engine.note})</span>
            </dt>
            <dd className="track" style={{ margin: 0 }}>
              <div
                className="fill"
                style={{ width: `${(value / max) * 100}%`, background: engine.colour }}
              />
            </dd>
            <dd className="value" style={{ margin: 0 }}>
              {count.format(value)}
              <span style={{ color: "var(--ink-muted)" }}> · {share.toFixed(1)}%</span>
            </dd>
          </div>
        );
      })}
    </dl>
  );
}
