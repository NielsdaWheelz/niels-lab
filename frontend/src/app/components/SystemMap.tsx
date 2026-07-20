export function SystemMap() {
  return (
    <aside className="system-map" aria-label="Niels's product engineering loop">
      <div className="system-map-header">
        <span>system sketch / 001</span>
        <span>rev. 07</span>
      </div>

      <div
        className="system-map-stage"
        role="img"
        aria-label="Ambiguous problems become useful products through explicit contracts, deterministic cores, observable AI, and human feedback."
      >
        <svg
          className="system-map-lines"
          viewBox="0 0 520 420"
          aria-hidden="true"
          focusable="false"
        >
          <path
            className="map-path map-path-a"
            d="M94 82 C170 86 174 172 246 198"
          />
          <path
            className="map-path map-path-b"
            d="M426 91 C353 94 347 168 274 198"
          />
          <path
            className="map-path map-path-c"
            d="M248 225 C196 255 160 299 126 346"
          />
          <path
            className="map-path map-path-d"
            d="M276 225 C335 254 371 300 408 347"
          />
          <path
            className="map-path map-path-loop"
            d="M129 348 C222 392 319 392 407 348"
          />
          <circle className="map-pulse map-pulse-a" cx="94" cy="82" r="5" />
          <circle className="map-pulse map-pulse-b" cx="426" cy="91" r="5" />
          <circle className="map-pulse map-pulse-c" cx="126" cy="346" r="5" />
          <circle className="map-pulse map-pulse-d" cx="408" cy="347" r="5" />
        </svg>

        <div className="system-node system-node-problem">
          <span>01</span>
          <strong>fuzzy problem</strong>
          <small>find the real constraint</small>
        </div>
        <div className="system-node system-node-signal">
          <span>02</span>
          <strong>human signal</strong>
          <small>make steering possible</small>
        </div>
        <div className="system-core">
          <span>product loop</span>
          <strong>ship → inspect → learn</strong>
        </div>
        <div className="system-node system-node-determinism">
          <span>03</span>
          <strong>deterministic core</strong>
          <small>contracts before magic</small>
        </div>
        <div className="system-node system-node-observable">
          <span>04</span>
          <strong>observable AI</strong>
          <small>traces, tests, evidence</small>
        </div>
      </div>

      <p className="system-map-note">make the ambitious thing legible ↗</p>
    </aside>
  )
}
