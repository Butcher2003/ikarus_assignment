import { useEffect, useMemo, useState } from "react";

import SelectableProductCard from "../components/SelectableProductCard";
import { ASSET_BASE_URL, fetchRecommendations, generateDescription } from "../lib/api";

const PLACEHOLDER = "https://via.placeholder.com/320x240?text=No+Image";

function toAbsoluteUrl(path) {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  return `${ASSET_BASE_URL}${path}`;
}

function DescriptionPage() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(6);
  const [recommendations, setRecommendations] = useState([]);
  const [recommendationsLoading, setRecommendationsLoading] = useState(false);
  const [recommendationsError, setRecommendationsError] = useState(null);

  const [selectedIds, setSelectedIds] = useState([]);

  const [descriptionResult, setDescriptionResult] = useState(null);
  const [descriptionLoading, setDescriptionLoading] = useState(false);
  const [descriptionError, setDescriptionError] = useState(null);

  const queryIsValid = query.trim().length > 0;

  const selectedCount = selectedIds.length;

  useEffect(() => {
    if (!recommendations.length) {
      return;
    }
    setSelectedIds((prev) => {
      if (prev.length) {
        return prev;
      }
      return recommendations.slice(0, Math.min(3, recommendations.length)).map((item) => item.uniq_id);
    });
  }, [recommendations]);

  const handleSearch = async (event) => {
    event.preventDefault();
    if (!queryIsValid) {
      setRecommendationsError("Enter a prompt to look up matching products.");
      return;
    }
    try {
      setRecommendationsLoading(true);
      setRecommendationsError(null);
      setDescriptionResult(null);
      const results = await fetchRecommendations(query.trim(), topK);
      setRecommendations(results);
      setSelectedIds(results.slice(0, Math.min(3, results.length)).map((item) => item.uniq_id));
    } catch (err) {
      console.error(err);
      setRecommendationsError("Unable to fetch recommendations right now.");
      setRecommendations([]);
      setSelectedIds([]);
    } finally {
      setRecommendationsLoading(false);
    }
  };

  const toggleSelection = (id) => {
    setSelectedIds((prev) => {
      if (prev.includes(id)) {
        return prev.filter((item) => item !== id);
      }
      return [...prev, id];
    });
  };

  const clearSelection = () => setSelectedIds([]);
  const selectAll = () => setSelectedIds(recommendations.map((item) => item.uniq_id));

  const handleGenerate = async () => {
    if (!queryIsValid) {
      setDescriptionError("Please enter the lifestyle or theme you want to describe.");
      return;
    }
    try {
      setDescriptionLoading(true);
      setDescriptionError(null);
      const response = await generateDescription({
        query: query.trim(),
        productIds: selectedIds,
        topK,
      });
      setDescriptionResult(response);
      if (Array.isArray(response.used_product_ids)) {
        setSelectedIds(response.used_product_ids);
      }
    } catch (err) {
      console.error(err);
      setDescriptionError("Could not generate a description. Try refining your selection.");
      setDescriptionResult(null);
    } finally {
      setDescriptionLoading(false);
    }
  };

  const descriptionText = useMemo(() => descriptionResult?.description ?? "", [descriptionResult]);

  return (
    <section className="flex flex-col gap-6">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold text-white">Generate Product Story</h1>
        <p className="text-sm text-slate-400">
          Describe the mood you want to set, explore recommended pieces, and pick the exact products to weave into a
          marketing blurb.
        </p>
      </header>

      <form onSubmit={handleSearch} className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-lg">
        <div className="flex flex-col gap-4 md:flex-row md:items-end">
          <div className="flex-1">
            <label htmlFor="description-query" className="text-sm font-medium text-slate-300">
              Inspiration prompt
            </label>
            <textarea
              id="description-query"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="e.g. A welcoming foyer with natural textures and smart storage"
              rows={2}
              className="mt-2 w-full resize-none rounded-xl border border-slate-800 bg-slate-950/80 p-3 text-sm text-slate-100 focus:border-primary-light focus:outline-none focus:ring-2 focus:ring-primary/40"
            />
          </div>
          <div className="w-full md:w-40">
            <label htmlFor="topk" className="text-sm font-medium text-slate-300">
              Matches to fetch
            </label>
            <input
              id="topk"
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
              className="mt-2 w-full rounded-xl border border-slate-800 bg-slate-950/80 p-3 text-sm text-slate-100 focus:border-primary-light focus:outline-none focus:ring-2 focus:ring-primary/40"
            />
          </div>
          <button
            type="submit"
            disabled={recommendationsLoading}
            className="h-12 rounded-full bg-primary px-6 text-sm font-semibold text-white transition hover:bg-primary-light disabled:opacity-50"
          >
            {recommendationsLoading ? "Searching…" : "Find matches"}
          </button>
        </div>
        {recommendationsError ? <p className="mt-3 text-sm text-rose-400">{recommendationsError}</p> : null}
      </form>

      {recommendations.length ? (
        <section className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-base font-semibold text-white">Select products</h2>
              <p className="text-xs text-slate-400">{selectedCount} selected</p>
            </div>
            <div className="flex gap-2 text-xs">
              <button
                type="button"
                onClick={selectAll}
                className="rounded-full border border-slate-700 px-3 py-1 font-semibold uppercase tracking-wide text-slate-300 transition hover:border-primary hover:text-primary-light"
              >
                Select all
              </button>
              <button
                type="button"
                onClick={clearSelection}
                className="rounded-full border border-slate-700 px-3 py-1 font-semibold uppercase tracking-wide text-slate-300 transition hover:border-primary hover:text-primary-light"
              >
                Clear
              </button>
            </div>
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {recommendations.map((product) => (
              <SelectableProductCard
                key={product.uniq_id}
                product={product}
                selected={selectedIds.includes(product.uniq_id)}
                onToggle={toggleSelection}
              />
            ))}
          </div>
        </section>
      ) : null}

      {recommendations.length ? (
        <section className="flex flex-col gap-3 rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-lg">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-base font-semibold text-white">Compose description</h2>
            <button
              type="button"
              onClick={handleGenerate}
              disabled={descriptionLoading}
              className="rounded-full bg-primary px-6 py-2 text-sm font-semibold text-white transition hover:bg-primary-light disabled:opacity-50"
            >
              {descriptionLoading ? "Writing…" : "Generate copy"}
            </button>
          </div>
          {descriptionError ? <p className="text-sm text-rose-400">{descriptionError}</p> : null}
          {descriptionText ? (
            <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-4 text-sm text-slate-200">
              {descriptionText.split("\n").map((line, index) => (
                <p key={index} className="mb-2 last:mb-0">
                  {line}
                </p>
              ))}
            </div>
          ) : null}
        </section>
      ) : null}

      {descriptionResult?.products?.length ? (
        <section className="space-y-3 rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-lg">
          <h2 className="text-base font-semibold text-white">Featured items in this story</h2>
          <ul className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {descriptionResult.products.map((product) => {
              const imageSrc = toAbsoluteUrl(product.image_url) || PLACEHOLDER;
              return (
                <li key={product.uniq_id} className="flex flex-col overflow-hidden rounded-xl border border-slate-800 bg-slate-950/60">
                <img
                  src={imageSrc}
                  alt={product.title}
                  className="h-40 w-full object-cover"
                  loading="lazy"
                />
                <div className="flex flex-1 flex-col gap-2 p-4 text-sm">
                  <div>
                    <h3 className="font-semibold text-slate-100">{product.title}</h3>
                    {product.brand ? (
                      <p className="text-xs uppercase tracking-wide text-primary-light">{product.brand}</p>
                    ) : null}
                  </div>
                  {product.description ? (
                    <p className="text-xs text-slate-400">{product.description}</p>
                  ) : (
                    <p className="text-xs text-slate-500">No short description captured.</p>
                  )}
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span className="font-semibold text-slate-200">
                      {typeof product.price === "number" ? `$${product.price.toFixed(2)}` : "N/A"}
                    </span>
                    <span>{product.categories?.[0] || ""}</span>
                  </div>
                </div>
              </li>
              );
            })}
          </ul>
        </section>
      ) : null}
    </section>
  );
}

export default DescriptionPage;
