import Link from 'next/link'
import { PageTitle } from '@/app/components/PageTitle'
import { createPageMetadata } from '@/app/site'
import { SamplingPlayground } from './SamplingPlayground'
import styles from './sampling.module.css'

export const metadata = createPageMetadata({
  title: 'How sampling works',
  description:
    'An interactive token-sampling playground: temperature, top-k, and top-p applied to a language model’s next-token distribution.',
  path: '/lab/sampling',
})

export default function Page() {
  return (
    <article className="canon leaf">
      <PageTitle>how sampling works</PageTitle>

      <div className="prose">
        <p>
          A language model doesn’t pick a word — it produces a{' '}
          <em>distribution</em> over every word, then samples from it. The knobs
          you set at inference time (temperature, top-k, top-p) reshape that
          distribution before the sample is drawn. Below is a hand-authored toy
          distribution for one prompt. Drag the sliders and watch the
          probabilities move.
        </p>
      </div>

      <SamplingPlayground />

      <aside className={styles.note}>
        <p>
          toy logits, real math — the softmax and the cuts are the ones
          production models run.
        </p>
      </aside>

      <section className="prose">
        <h2>what you’re looking at</h2>
        <p>
          The final layer of a transformer emits one raw score — a{' '}
          <strong>logit</strong> — for every token in its vocabulary. Logits are
          unbounded real numbers; on their own they aren’t probabilities. To
          turn them into probabilities that sum to 1, you run them through{' '}
          <strong>softmax</strong>:
        </p>
        <pre>
          <code>{`p_i = exp(logit_i) / Σ_j exp(logit_j)`}</code>
        </pre>
        <p>
          Exponentiating makes every value positive and widens the gaps: a logit
          that’s a couple points higher than its neighbors ends up owning a
          lopsided share of the probability mass. That’s the tall bar you see
          for <code>left</code>.
        </p>

        <h2>temperature: dividing before you exponentiate</h2>
        <p>
          Temperature <code>T</code> is a single number you divide the logits by{' '}
          <em>before</em> the softmax:
        </p>
        <pre>
          <code>{`p_i = exp(logit_i / T) / Σ_j exp(logit_j / T)`}</code>
        </pre>
        <p>
          That’s the entire mechanism — no randomness lives here, just a
          rescale. But it does two opposite things depending on which side of 1
          you’re on:
        </p>
        <ul>
          <li>
            <strong>T &lt; 1</strong> divides by something small, stretching the
            logit gaps <em>wider</em>. Softmax then concentrates even harder on
            the leader. The distribution gets peaky and confident.
          </li>
          <li>
            <strong>T &gt; 1</strong> shrinks the gaps, so softmax spreads mass
            toward the tail. Low-probability tokens gain share and the output
            gets more surprising.
          </li>
          <li>
            <strong>T = 1</strong> is the raw distribution, untouched.
          </li>
        </ul>
        <p>
          Drag temperature to <code>0.1</code> and the model becomes nearly
          greedy — the leading token takes almost all of the mass. Push it to{' '}
          <code>2.0</code> and the tail carries a large share.
        </p>

        <aside className={styles.note}>
          <p>
            temperature only rescales — the <em>ordering</em> of tokens never
            changes, at any value of T.
          </p>
        </aside>

        <h2>top-k and top-p: cutting the tail</h2>
        <p>
          Even a well-shaped distribution has a long tail of low-probability
          tokens. Any single one is unlikely, but collectively they carry enough
          mass that, sampled often enough, an incoherent token will eventually
          be drawn. Truncation sampling removes that tail before the draw. Two
          ways to set the cutoff:
        </p>
        <ul>
          <li>
            <strong>Top-k</strong> keeps a <em>fixed count</em> — the k highest
            tokens — and zeroes the rest. The count ignores the shape of the
            distribution: k=5 keeps too few tokens when the model is uncertain
            across 40 good options, and too many when it is certain of one.
          </li>
          <li>
            <strong>Top-p</strong> (nucleus sampling) keeps a{' '}
            <em>fixed probability mass</em> instead: sort tokens high to low,
            walk down accumulating probability, and stop the moment the running
            total crosses p. When the model is confident, that’s one or two
            tokens; when it’s unsure, the nucleus widens to include many. The
            cutoff adapts to the shape of the distribution — which is exactly
            why it was proposed (Holtzman et al., 2019) as a fix for top-k’s
            rigidity.
          </li>
        </ul>
        <p>
          After either cut, the survivors are <strong>renormalized</strong> so
          their probabilities sum back to 1 — that’s why the kept bars grow when
          you tighten a knob: they’re dividing up the mass the cut removed. In
          the playground, cut tokens don’t vanish; they go ghosted and hatched,
          tagged with which knob did the cutting, so you can see the mass that
          got thrown away. Set both, and a token is cut if <em>either</em> rule
          rejects it.
        </p>

        <h2>myths worth retiring</h2>
        <ul>
          <li>
            <strong>“Temperature 0 is deterministic.”</strong> Mostly. You can’t
            literally divide by zero, so implementations special-case{' '}
            <code>T = 0</code> to mean “skip sampling, take the argmax” (greedy
            decoding). That’s deterministic in principle — but the same prompt
            can still vary run to run because of floating-point
            non-associativity, GPU kernel scheduling, and batching. Determinism
            is a property of the whole stack, not just the temperature field.
          </li>
          <li>
            <strong>“Higher temperature = more creative.”</strong> It’s more{' '}
            <em>random</em>, which isn’t the same thing. Past a point you’re not
            adding creativity; you’re sampling from the incoherent tail.
          </li>
          <li>
            <strong>
              “Top-p and top-k are alternatives you pick between.”
            </strong>{' '}
            They stack. Many production configs apply a large top-k as a hard
            cap and let top-p do the adaptive trimming underneath it.
          </li>
        </ul>

        <p>
          Sampling settings don’t change what the model <em>knows</em> — the
          logits are fixed the instant the forward pass ends. They change which
          part of the distribution the sample can come from.
        </p>
      </section>

      <hr />
      <p className={styles.back}>
        <Link href="/lab">back to the lab</Link>
      </p>
    </article>
  )
}
