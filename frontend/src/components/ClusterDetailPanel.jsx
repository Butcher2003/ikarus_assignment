import Plot from "react-plotly.js";

import { ASSET_BASE_URL } from "../lib/api";

const IMAGE_PLACEHOLDER = "https://via.placeholder.com/320x240?text=No+Image";

function toAbsoluteUrl(path) {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  return `${ASSET_BASE_URL}${path}`;
}

function WordCloud({ terms }) {
  if (!terms?.length) {
    return <p className="text-sm text-slate-500">No standout keywords for this cluster yet.</p>;
  }
  const maxCount = Math.max(...terms.map((term) => term.count));
  return (
    <div className="flex flex-wrap gap-2">
      {terms.map((term) => {
        const weight = term.count / maxCount;
        const fontSizeRem = 0.75 + weight * 1.75;
        return (
          <span
            key={term.term}
            className="rounded-lg bg-slate-800/80 px-2 py-1 text-slate-200 transition hover:bg-primary/40"
            style={{ fontSize: `${fontSizeRem}rem` }}
            title={`Frequency: ${term.count}`}
          >
            {term.term}
          </span>
        );
      })}
    </div>
  );
}

function ClusterDetailPanel({ clusterId, detail, loading, error, onClose }) {
  if (clusterId === null) {
    return null;
  }

  const scatterProducts = detail?.products?.filter(
    (product) => Array.isArray(product.coordinates) && product.coordinates.length >= 3
  );

  const plotData = scatterProducts?.length
    ? [
        {
          x: scatterProducts.map((product) => product.coordinates[0]),
          y: scatterProducts.map((product) => product.coordinates[1]),
          z: scatterProducts.map((product) => product.coordinates[2]),
          text: scatterProducts.map((product) => product.title),
          type: "scatter3d",
          mode: "markers",
          marker: {
            size: 4,
            color: "#8B5CF6",
            opacity: 0.85,
          },
        },
      ]
    : null;

  const plotLayout = {
    margin: { l: 0, r: 0, t: 24, b: 0 },
    scene: {
      xaxis: { title: "Dim 1", showbackground: false, zeroline: false, color: "#94a3b8" },
      yaxis: { title: "Dim 2", showbackground: false, zeroline: false, color: "#94a3b8" },
      zaxis: { title: "Dim 3", showbackground: false, zeroline: false, color: "#94a3b8" },
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#94a3b8" },
  };

  const formatPrice = (price) => {
    if (typeof price === "number" && !Number.isNaN(price)) {
      return `$${price.toFixed(2)}`;
    }
    return "N/A";
  };

  return (
    <div className="mt-6 rounded-xl border border-slate-800 bg-slate-900/70 p-5 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold text-white">Cluster {clusterId}</h3>
          {detail ? (
            <p className="text-sm text-slate-400">{detail.total_products} products grouped by semantic similarity.</p>
          ) : null}
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded-full border border-slate-700 px-4 py-1 text-xs font-semibold uppercase tracking-wide text-slate-300 transition hover:border-primary hover:text-primary-light"
        >
          Clear selection
        </button>
      </div>

      {loading ? (
        <p className="mt-4 text-sm text-slate-400">Loading cluster insights…</p>
      ) : error ? (
        <p className="mt-4 text-sm text-rose-400">{error}</p>
      ) : detail ? (
        <div className="mt-4 flex flex-col gap-6">
          <div className="grid gap-4 lg:grid-cols-2">
            <section className="rounded-lg border border-slate-800/80 bg-slate-950/60 p-4">
              <h4 className="text-sm font-semibold uppercase tracking-wide text-slate-400">Keyword spotlight</h4>
              <div className="mt-3">
                <WordCloud terms={detail.top_terms} />
              </div>
            </section>
            <section className="rounded-lg border border-slate-800/80 bg-slate-950/60 p-4">
              <h4 className="text-sm font-semibold uppercase tracking-wide text-slate-400">3D embedding view</h4>
              <div className="mt-3 overflow-hidden rounded-lg bg-slate-950/80">
                {plotData ? (
                  <Plot data={plotData} layout={plotLayout} style={{ width: "100%", height: "320px" }} config={{ displayModeBar: false }} />
                ) : (
                  <p className="p-4 text-sm text-slate-500">No projection data available for this cluster.</p>
                )}
              </div>
              {detail?.centroid ? (
                <p className="mt-2 text-xs text-slate-500">
                  Centroid: {detail.centroid.map((value) => value.toFixed(2)).join(", ")}
                </p>
              ) : null}
            </section>
          </div>

          <section>
            <h4 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
              Sample products ({detail.products.length} shown)
            </h4>
            <ul className="mt-3 grid gap-3 md:grid-cols-2 lg:grid-cols-3">
              {detail.products.map((product) => {
                const imageSrc = toAbsoluteUrl(product.image_url) || IMAGE_PLACEHOLDER;
                return (
                  <li key={product.uniq_id} className="flex flex-col overflow-hidden rounded-xl border border-slate-800 bg-slate-950/60">
                    <img src={imageSrc} alt={product.title} className="h-40 w-full object-cover" loading="lazy" />
                    <div className="flex flex-1 flex-col gap-2 p-4">
                      <div>
                        <h5 className="text-sm font-semibold text-slate-100">{product.title}</h5>
                        {product.brand ? (
                          <p className="text-xs uppercase tracking-wide text-primary-light">{product.brand}</p>
                        ) : null}
                      </div>
                      {product.description ? (
                        <p className="text-xs text-slate-400">{product.description}</p>
                      ) : (
                        <p className="text-xs text-slate-500">No description captured.</p>
                      )}
                      <div className="flex items-center justify-between text-xs text-slate-400">
                        <span className="font-semibold text-slate-200">{formatPrice(product.price)}</span>
                        {product.coordinates ? (
                          <span title={product.coordinates.join(", ")}>
                            ({product.coordinates.map((value) => value.toFixed(2)).join(", ")})
                          </span>
                        ) : (
                          <span>—</span>
                        )}
                      </div>
                      <div className="flex flex-wrap gap-1 text-[11px] text-slate-400">
                        {product.categories.slice(0, 4).map((category) => (
                          <span key={`${product.uniq_id}-${category}`} className="rounded-full bg-slate-800/80 px-2 py-1">
                            {category}
                          </span>
                        ))}
                      </div>
                    </div>
                  </li>
                );
              })}
            </ul>
          </section>
        </div>
      ) : (
        <p className="mt-4 text-sm text-slate-500">No insights available for this cluster.</p>
      )}
    </div>
  );
}

export default ClusterDetailPanel;
