"use client";

import { useState } from "react";
import { useCopy } from "@/components/Language";
import { formatAp } from "@/lib/format";
import type { Experiment } from "@/lib/types";

function format(value: number | null, digits = 3) {
  if (value === null || Number.isNaN(value)) return null;
  return value.toFixed(digits);
}

export function FractionStudy({ experiments }: { experiments: Experiment[] }) {
  const { t } = useCopy();
  const fractions = [1, 0.2, 0.05, 0.01];
  const [fraction, setFraction] = useState(1);
  const current = experiments.find((item) => item.model === "diffusion" && item.training_fraction === fraction);
  const history = current?.training;

  return (
    <div>
      <div className="fractions" role="group" aria-label={t.fractionLabel}>
        {fractions.map((value) => (
          <button key={value} type="button" aria-pressed={fraction === value} onClick={() => setFraction(value)}>
            {Math.round(value * 100)}%
          </button>
        ))}
      </div>
      {current ? <Evidence experiment={current} history={history ?? null} /> : <p className="awaiting">{t.awaiting}</p>}
    </div>
  );
}

function Evidence({ experiment, history }: { experiment: Experiment; history: { epoch: number; val_loss: number }[] | null }) {
  const { t } = useCopy();
  const pending = experiment.status === "pending";
  return (
    <div>
      <p className="note">
        {experiment.id} · {t.status(experiment.status)} · config {experiment.config}
      </p>
      {experiment.sample ? (
        <figure className="triptych">
          <img src={experiment.sample} alt={t.panelAlt(experiment.id)} />
          <figcaption>
            <span>RGB</span>
            <span>{t.generated}</span>
            <span>{t.realInfrared}</span>
          </figcaption>
        </figure>
      ) : null}
      <div className="metrics">
        <Metric label="SSIM" value={format(experiment.metrics.ssim)} pending={pending || experiment.metrics.ssim === null} />
        <Metric label="PSNR" value={format(experiment.metrics.psnr, 2)} pending={pending || experiment.metrics.psnr === null} />
        <Metric label="LPIPS" value={format(experiment.metrics.lpips)} pending={pending || experiment.metrics.lpips === null} />
        <Metric label="AP" value={experiment.metrics.ap50 === null ? null : formatAp(experiment.metrics.ap50)} pending={experiment.metrics.ap50 === null} />
      </div>
      <p className="note">{t.sameList}</p>
      {history && history.length > 0 ? <Curve rows={history} /> : <p className="awaiting">{t.noCurve}</p>}
    </div>
  );
}

const fractionOrder = [1, 0.2, 0.05, 0.01];

export function ScarcityChart({ experiments }: { experiments: Experiment[] }) {
  const { t } = useCopy();
  const rows = fractionOrder
    .map((fraction) => experiments.find((item) => item.model === "diffusion" && item.training_fraction === fraction))
    .filter((item): item is Experiment => Boolean(item));
  if (rows.length < 2 || rows.some((row) => row.metrics.psnr === null || row.metrics.ap50 === null)) {
    return null;
  }
  const panels = [
    { label: "PSNR", hint: t.hintCloser, values: rows.map((row) => row.metrics.psnr as number), max: 16, digits: 1 },
    { label: "SSIM", hint: t.hintCloser, values: rows.map((row) => row.metrics.ssim as number), max: 0.4, digits: 2 },
    { label: "LPIPS", hint: t.hintLower, values: rows.map((row) => row.metrics.lpips as number), max: 1.2, digits: 2 },
    { label: "AP", hint: t.hintBetter, values: rows.map((row) => row.metrics.ap50 as number), max: 0.02, digits: 3 },
  ];
  return (
    <figure className="trajectories">
      <div>
        {panels.map((panel) => (
          <Panel key={panel.label} {...panel} ticks={rows.map((row) => `${Math.round(row.training_fraction * 100)}`)} />
        ))}
      </div>
      <figcaption className="note">{t.chartCaption}</figcaption>
    </figure>
  );
}

function Panel({ label, hint, values, max, digits, ticks }: { label: string; hint: string; values: number[]; max: number; digits: number; ticks: string[] }) {
  const { t } = useCopy();
  const pad = { left: 32, right: 8, top: 16, bottom: 22 };
  const width = 180;
  const height = 120;
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const points = values
    .map((value, index) => {
      const x = pad.left + (index / Math.max(values.length - 1, 1)) * plotWidth;
      const y = pad.top + (1 - Math.min(value, max) / max) * plotHeight;
      return { x, y };
    });
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={t.chartAria(label, hint, ticks.join(", "))}>
      <text x={pad.left} y="12">{label}</text>
      <line x1={pad.left} y1={pad.top} x2={pad.left} y2={pad.top + plotHeight} />
      <line x1={pad.left} y1={pad.top + plotHeight} x2={pad.left + plotWidth} y2={pad.top + plotHeight} />
      <text x="2" y={pad.top + 4}>{max >= 10 ? max.toFixed(0) : max.toFixed(Math.max(digits, 1))}</text>
      <polyline fill="none" points={points.map((point) => `${point.x},${point.y}`).join(" ")} />
      {points.map((point, index) => (
        <g key={ticks[index]}>
          <circle cx={point.x} cy={point.y} r="2.4" />
          <text x={point.x} y={height - 6} textAnchor="middle">{ticks[index]}</text>
        </g>
      ))}
    </svg>
  );
}

function Metric({ label, value, pending }: { label: string; value: string | null; pending: boolean }) {
  return (
    <div className="metric">
      <span>{label}</span>
      {pending || value === null ? <b className="awaiting">—</b> : <b>{value}</b>}
    </div>
  );
}

function Curve({ rows }: { rows: { epoch: number; val_loss: number }[] }) {
  const { t } = useCopy();
  const values = rows.map((row) => row.val_loss);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pad = { left: 16, right: 12, top: 36, bottom: 32 };
  const width = 340;
  const height = 200;
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const points = values
    .map((value, index) => {
      const x = pad.left + (index / Math.max(values.length - 1, 1)) * plotWidth;
      const y = pad.top + (1 - (value - min) / span) * plotHeight;
      return `${x},${y}`;
    })
    .join(" ");
  return (
    <svg className="curve" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={t.curveLabel}>
      <text x={pad.left} y="18">{t.valLoss}</text>
      <line x1={pad.left} y1={pad.top} x2={pad.left} y2={pad.top + plotHeight} />
      <line x1={pad.left} y1={pad.top + plotHeight} x2={pad.left + plotWidth} y2={pad.top + plotHeight} />
      <polyline fill="none" points={points} />
      <text x={pad.left + plotWidth / 2} y={height - 8} textAnchor="middle">{t.epoch}</text>
    </svg>
  );
}

function modelLabel(experiment: Experiment, t: ReturnType<typeof useCopy>["t"]) {
  return t.modelName(experiment.model === "baseline", Math.round(experiment.training_fraction * 100));
}

function dominantPrefix(rows: Experiment["highest_error"]) {
  const counts = new Map<string, number>();
  for (const row of rows) {
    const prefix = row.sample_id.slice(0, 3);
    counts.set(prefix, (counts.get(prefix) ?? 0) + 1);
  }
  return [...counts.entries()].sort((left, right) => right[1] - left[1])[0];
}

export function FailureList({ experiments }: { experiments: Experiment[] }) {
  const { t } = useCopy();
  const ranked = experiments.filter((experiment) => experiment.highest_error.length > 0);
  if (ranked.length === 0) {
    return <p className="awaiting">{t.reviewMissing}</p>;
  }
  return (
    <div>
      <div className="failure-grid">
        {Object.keys(t.scenes).map((sampleId) => (
          <figure key={sampleId} className="failure-pair">
            <img src={`/samples/failures/${sampleId}`} alt={t.pairAlt(sampleId)} />
            <figcaption>
              <span>{sampleId.replace(".jpg", "")} · {t.visibleShort}</span>
              <span>{t.realInfrared}</span>
            </figcaption>
            <p>{t.scenes[sampleId]}</p>
          </figure>
        ))}
      </div>
      <div className="rank-list">
        {ranked.map((experiment) => {
          const [prefix, count] = dominantPrefix(experiment.highest_error);
          const worst = experiment.highest_error[0];
          return (
            <div key={experiment.id}>
              <p className="note">
                {t.rank(modelLabel(experiment, t), worst.sample_id.replace(".jpg", ""), worst.mae?.toFixed(3) ?? "—", count, experiment.highest_error.length, prefix)}
              </p>
              <ol>
                {experiment.highest_error.map((row) => (
                  <li key={row.sample_id}>
                    <span>{row.sample_id.replace(".jpg", "")}</span>
                    <span>{row.mae?.toFixed(3)}</span>
                  </li>
                ))}
              </ol>
            </div>
          );
        })}
      </div>
    </div>
  );
}
