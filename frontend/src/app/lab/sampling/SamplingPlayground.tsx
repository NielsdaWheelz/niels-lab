'use client'

import { useId, useMemo, useState } from 'react'
import styles from './sampling.module.css'

/* ------------------------------------------------------------------ *
 * A fixed toy next-token distribution for the prompt
 *   "the bug is never where you ___"
 * Logits are hand-authored constants (not a real model) so the math is
 * fully inspectable. Sorted by logit, descending.
 * ------------------------------------------------------------------ */
type Token = { text: string; logit: number }

const PROMPT = 'the bug is never where you'

const TOKENS: readonly Token[] = [
  { text: 'left', logit: 6.2 },
  { text: 'look', logit: 4.0 },
  { text: 'think', logit: 3.7 },
  { text: 'expect', logit: 3.3 },
  { text: 'debug', logit: 2.6 },
  { text: 'grep', logit: 2.2 },
  { text: 'sleep', logit: 1.8 },
  { text: 'blame', logit: 1.5 },
  { text: 'refactor', logit: 1.2 },
  { text: 'panic', logit: 0.9 },
  { text: 'git', logit: 0.6 },
  { text: 'coffee', logit: 0.3 },
  { text: 'cry', logit: 0.0 },
  { text: 'ship', logit: -0.4 },
]

const K_MAX = TOKENS.length

/* ------------------------------------------------------------------ *
 * Math — pure. Computed during render from slider state, never in an
 * effect. Math.random() lives ONLY inside event handlers.
 * ------------------------------------------------------------------ */
type CutReason = null | 'k' | 'p' | 'both'

type Bin = {
  text: string
  index: number
  /** softmax(logits / T) before any masking */
  base: number
  /** renormalized probability after top-k + top-p (0 if cut) */
  prob: number
  kept: boolean
  cutBy: CutReason
}

/** Softmax with temperature: p_i = exp(logit_i / T) / Σ exp(logit_j / T). */
function softmaxWithTemperature(tokens: readonly Token[], t: number): number[] {
  const scaled = tokens.map((tok) => tok.logit / t)
  const max = Math.max(...scaled)
  const exps = scaled.map((s) => Math.exp(s - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}

function computeDistribution(
  tokens: readonly Token[],
  temperature: number,
  topK: number,
  topP: number,
): Bin[] {
  const base = softmaxWithTemperature(tokens, temperature)

  // Rank indices by probability, descending. (Order matches the token order
  // because logits are pre-sorted and softmax is monotonic — but rank
  // explicitly so the masks are correct for any inputs.)
  const ranked = base
    .map((prob, index) => ({ index, prob }))
    .sort((a, b) => b.prob - a.prob)

  // top-k: keep the k highest-probability tokens.
  const keepByK = new Set(ranked.slice(0, topK).map((r) => r.index))

  // top-p (nucleus): smallest set of top tokens whose cumulative probability
  // reaches p. The top token is always kept (added before the threshold check),
  // which handles the p < prob(top) edge case correctly.
  const keepByP = new Set<number>()
  let cumulative = 0
  for (const r of ranked) {
    keepByP.add(r.index)
    cumulative += r.prob
    if (cumulative >= topP) break
  }

  const kept = tokens.map((_, i) => keepByK.has(i) && keepByP.has(i))
  const keptSum = base.reduce((s, prob, i) => (kept[i] ? s + prob : s), 0)

  return tokens.map((tok, i) => {
    let cutBy: CutReason = null
    if (!kept[i]) {
      const outK = !keepByK.has(i)
      const outP = !keepByP.has(i)
      cutBy = outK && outP ? 'both' : outK ? 'k' : 'p'
    }
    return {
      text: tok.text,
      index: i,
      base: base[i],
      prob: kept[i] ? base[i] / keptSum : 0,
      kept: kept[i],
      cutBy,
    }
  })
}

/** Draw one index from the renormalized distribution. rand ∈ [0, 1). */
function sampleIndex(dist: Bin[], rand: number): number {
  let cumulative = 0
  for (const bin of dist) {
    cumulative += bin.prob
    if (rand < cumulative) return bin.index
  }
  // Floating-point fallback: return the last surviving token.
  for (let i = dist.length - 1; i >= 0; i--) {
    if (dist[i].kept) return dist[i].index
  }
  return 0
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`

/* ------------------------------------------------------------------ *
 * SVG chart geometry (user units; the SVG scales to its container).
 * ------------------------------------------------------------------ */
const VIEW_W = 700
const PAD_TOP = 10
const ROW_H = 32
const BAR_H = 14
const LABEL_X = 112
const BAR_X = 126
const BAR_MAX = 452
const PCT_X = BAR_X + BAR_MAX + 14
const VIEW_H = PAD_TOP * 2 + ROW_H * K_MAX

const rowCenter = (i: number) => PAD_TOP + i * ROW_H + ROW_H / 2

/* ------------------------------------------------------------------ */

export function SamplingPlayground() {
  const [temperature, setTemperature] = useState(1.0)
  const [topK, setTopK] = useState(K_MAX)
  const [topP, setTopP] = useState(1.0)
  const [draws, setDraws] = useState<number[]>([])
  const [lastDraw, setLastDraw] = useState<number | null>(null)

  const uid = useId()
  const hatchId = `${uid}-hatch`

  const dist = useMemo(
    () => computeDistribution(TOKENS, temperature, topK, topP),
    [temperature, topK, topP],
  )

  const maxProb = useMemo(
    () => Math.max(...dist.map((b) => (b.kept ? b.prob : 0)), 1e-6),
    [dist],
  )

  const survivors = dist.filter((b) => b.kept).length
  const totalDraws = draws.length

  const counts = useMemo(() => {
    const c = new Array<number>(TOKENS.length).fill(0)
    for (const d of draws) c[d] += 1
    return c
  }, [draws])

  function drawOnce() {
    const index = sampleIndex(dist, Math.random())
    setDraws((prev) => [...prev, index])
    setLastDraw(index)
  }

  function drawMany() {
    const batch: number[] = []
    for (let i = 0; i < 20; i++) batch.push(sampleIndex(dist, Math.random()))
    setDraws((prev) => [...prev, ...batch])
    setLastDraw(batch[batch.length - 1])
  }

  function reset() {
    setDraws([])
    setLastDraw(null)
  }

  const controls = [
    {
      id: `${uid}-temp`,
      label: 'temperature',
      value: temperature.toFixed(2),
      min: 0.1,
      max: 2.0,
      step: 0.05,
      current: temperature,
      onChange: (v: number) => setTemperature(v),
      hint:
        temperature < 0.85
          ? 'sharpening — the peak grabs more mass'
          : temperature > 1.15
            ? 'flattening — the tail wakes up'
            : 'the raw softmax, untouched',
    },
    {
      id: `${uid}-topk`,
      label: 'top-k',
      value: topK === K_MAX ? `${topK} · off` : String(topK),
      min: 1,
      max: K_MAX,
      step: 1,
      current: topK,
      onChange: (v: number) => setTopK(v),
      hint:
        topK === K_MAX
          ? 'keeping every candidate'
          : `only the ${topK} likeliest survive`,
    },
    {
      id: `${uid}-topp`,
      label: 'top-p',
      value: topP >= 1 ? '1.00 · off' : topP.toFixed(2),
      min: 0.05,
      max: 1.0,
      step: 0.05,
      current: topP,
      onChange: (v: number) => setTopP(v),
      hint:
        topP >= 1
          ? 'keeping the whole distribution'
          : `smallest set covering ${Math.round(topP * 100)}% of the mass`,
    },
  ]

  return (
    <div className={styles.playground}>
      <div className={styles.controls}>
        {controls.map((c) => (
          <div key={c.id} className={styles.control}>
            <div className={styles.controlHead}>
              <label htmlFor={c.id} className={styles.controlName}>
                {c.label}
              </label>
              <output htmlFor={c.id} className={styles.controlValue}>
                {c.value}
              </output>
            </div>
            <input
              id={c.id}
              className={styles.slider}
              type="range"
              min={c.min}
              max={c.max}
              step={c.step}
              value={c.current}
              onChange={(e) => c.onChange(Number(e.target.value))}
            />
            <p className={styles.hint}>{c.hint}</p>
          </div>
        ))}
      </div>

      <figure className={styles.chartWrap}>
        <figcaption className={styles.chartCaption}>
          <span className={styles.promptLabel}>next-token distribution</span>
          <span className={styles.survivorTag}>
            {survivors} of {K_MAX} live
          </span>
        </figcaption>

        <svg
          className={styles.chart}
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          role="img"
          aria-label={`Bar chart of next-token probabilities after temperature ${temperature.toFixed(
            2,
          )}, top-k ${topK}, and top-p ${topP.toFixed(
            2,
          )}. ${survivors} of ${K_MAX} tokens survive the cut.`}
          preserveAspectRatio="xMinYMin meet"
        >
          <defs>
            <pattern
              id={hatchId}
              width="7"
              height="7"
              patternUnits="userSpaceOnUse"
              patternTransform="rotate(45)"
            >
              <line
                className={styles.hatchLine}
                x1="0"
                y1="0"
                x2="0"
                y2="7"
                strokeWidth="2.5"
              />
            </pattern>
          </defs>

          {dist.map((bin, i) => {
            const cy = rowCenter(i)
            const barY = cy - BAR_H / 2
            const solidW = (bin.prob / maxProb) * BAR_MAX
            const ghostW = (bin.base / maxProb) * BAR_MAX
            const isDrawn = lastDraw === bin.index
            const empFreq = totalDraws > 0 ? counts[bin.index] / totalDraws : 0
            const empX = BAR_X + Math.min(empFreq / maxProb, 1) * BAR_MAX

            return (
              <g key={bin.index}>
                {/* track */}
                <rect
                  className={styles.track}
                  x={BAR_X}
                  y={barY}
                  width={BAR_MAX}
                  height={BAR_H}
                  rx={3}
                />

                {/* token label */}
                <text
                  className={`${styles.tokenLabel} ${
                    bin.kept ? '' : styles.tokenLabelCut
                  }`}
                  x={LABEL_X}
                  y={cy}
                  textAnchor="end"
                  dominantBaseline="central"
                >
                  {bin.text}
                </text>

                {bin.kept ? (
                  <rect
                    className={`${styles.bar} ${isDrawn ? styles.barDrawn : ''}`}
                    x={BAR_X}
                    y={barY}
                    height={BAR_H}
                    rx={3}
                    style={{ width: `${solidW}px` }}
                  />
                ) : (
                  <rect
                    className={styles.barCut}
                    x={BAR_X}
                    y={barY}
                    height={BAR_H}
                    rx={3}
                    fill={`url(#${hatchId})`}
                    style={{ width: `${ghostW}px` }}
                  />
                )}

                {/* empirical-frequency tick */}
                {totalDraws > 0 && (
                  <line
                    className={styles.empTick}
                    x1={empX}
                    x2={empX}
                    y1={cy - BAR_H / 2 - 3}
                    y2={cy + BAR_H / 2 + 3}
                  >
                    <title>
                      {`observed ${counts[bin.index]}/${totalDraws} = ${pct(
                        empFreq,
                      )}`}
                    </title>
                  </line>
                )}

                {/* value / cut reason */}
                {bin.kept ? (
                  <text
                    className={styles.valueLabel}
                    x={PCT_X}
                    y={cy}
                    textAnchor="start"
                    dominantBaseline="central"
                  >
                    {pct(bin.prob)}
                  </text>
                ) : (
                  <text
                    className={styles.cutLabel}
                    x={PCT_X}
                    y={cy}
                    textAnchor="start"
                    dominantBaseline="central"
                  >
                    cut · {bin.cutBy === 'both' ? 'k+p' : bin.cutBy}
                  </text>
                )}
              </g>
            )
          })}
        </svg>

        {totalDraws > 0 && (
          <p className={styles.empLegend}>
            <span className={styles.empSwatch} aria-hidden="true" />
            observed frequency over {totalDraws}{' '}
            {totalDraws === 1 ? 'draw' : 'draws'} — watch it settle onto the
            bars
          </p>
        )}
      </figure>

      <div className={styles.actions}>
        <button
          type="button"
          className={`${styles.btn} ${styles.btnPrimary}`}
          onClick={drawOnce}
        >
          sample <span aria-hidden="true">→</span>
        </button>
        <button type="button" className={styles.btn} onClick={drawMany}>
          sample ×20
        </button>
        <button
          type="button"
          className={styles.btn}
          onClick={reset}
          disabled={totalDraws === 0}
        >
          reset
        </button>
      </div>

      <div className={styles.strip} aria-live="polite">
        <span className={styles.stripPrompt}>{PROMPT}</span>
        {draws.length === 0 ? (
          <span className={styles.stripEmpty}>___ (nothing drawn yet)</span>
        ) : (
          draws.map((d, i) => (
            <span
              key={`${d}-${i}`}
              className={`${styles.chip} ${
                i === draws.length - 1 ? styles.chipRecent : ''
              }`}
            >
              {TOKENS[d].text}
            </span>
          ))
        )}
      </div>

      {/* Accessible table equivalent of the chart. */}
      <table className={styles.srOnly}>
        <caption>
          Next-token probabilities for “{PROMPT} ___” after temperature, top-k,
          and top-p.
        </caption>
        <thead>
          <tr>
            <th scope="col">token</th>
            <th scope="col">probability</th>
            <th scope="col">status</th>
          </tr>
        </thead>
        <tbody>
          {dist.map((bin) => (
            <tr key={bin.index}>
              <td>{bin.text}</td>
              <td>{bin.kept ? pct(bin.prob) : '0.0%'}</td>
              <td>{bin.kept ? 'kept' : `cut by top-${bin.cutBy}`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
