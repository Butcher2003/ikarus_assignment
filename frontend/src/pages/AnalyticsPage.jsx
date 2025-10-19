import { useEffect, useRef, useState } from "react";

import ClusterDetailPanel from "../components/ClusterDetailPanel";
import { fetchAnalytics, fetchClusterDetail } from "../lib/api";

function StatCard({ title, value, description }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5 shadow-lg">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</p>
      <p className="mt-2 text-2xl font-semibold text-white">{value}</p>
      {description ? <p className="mt-1 text-sm text-slate-400">{description}</p> : null}
    </div>
  );
}

function AnalyticsPage() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedClusterId, setSelectedClusterId] = useState(null);
  const [clusterDetail, setClusterDetail] = useState(null);
  const [clusterError, setClusterError] = useState(null);
  const [clusterLoading, setClusterLoading] = useState(false);
  const detailRef = useRef(null);

  useEffect(() => {
    async function load() {
      try {
        const response = await fetchAnalytics();
        setData(response);
      } catch (err) {
        console.error(err);
        setError("Analytics service is unavailable right now.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  useEffect(() => {
    if (selectedClusterId === null) {
      return undefined;
    }

    let cancelled = false;

    async function loadCluster() {
      try {
        setClusterLoading(true);
        setClusterError(null);
        const detail = await fetchClusterDetail(selectedClusterId, 36);
        if (!cancelled) {
          setClusterDetail(detail);
        }
      } catch (err) {
        console.error(err);
        if (!cancelled) {
          setClusterError("Unable to load this cluster right now.");
          setClusterDetail(null);
        }
      } finally {
        if (!cancelled) {
          setClusterLoading(false);
        }
      }
    }

    setClusterDetail(null);
    loadCluster();

    return () => {
      cancelled = true;
    };
  }, [selectedClusterId]);

  useEffect(() => {
    if (selectedClusterId === null) {
      return;
    }
    if (clusterLoading) {
      return;
    }
    if (detailRef.current) {
      detailRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [selectedClusterId, clusterLoading]);

  if (loading) {
    return <p className="text-sm text-slate-400">Loading analytics…</p>;
  }

  if (error) {
    return <p className="text-sm text-rose-400">{error}</p>;
  }

  if (!data) {
    return null;
  }

  const currency = (amount) => (amount ? `$${amount.toFixed(2)}` : "N/A");

  const handleClusterSelect = (clusterId) => {
    if (selectedClusterId === clusterId) {
      setSelectedClusterId(null);
      setClusterDetail(null);
      setClusterError(null);
      setClusterLoading(false);
      return;
    }
    setSelectedClusterId(clusterId);
  };

  return (
    <section className="flex flex-col gap-6">
      <div className="grid gap-4 md:grid-cols-3">
        <StatCard title="Total Products" value={data.total_products} description="Items indexed in the vector store" />
        <StatCard
          title="Products With Images"
          value={data.images.with}
          description={`${data.images.without} items missing visuals`}
        />
        <StatCard
          title="Tracked Prices"
          value={data.price.count}
          description={`Avg ${currency(data.price.mean)} · Median ${currency(data.price.median)}`}
        />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">Top Categories</h2>
          <ul className="mt-3 space-y-2 text-sm">
            {data.categories.map((cat) => (
              <li key={cat.name} className="flex items-center justify-between rounded-lg bg-slate-950/60 px-3 py-2">
                <span>{cat.name}</span>
                <span className="text-slate-400">{cat.count}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">Top Brands</h2>
          <ul className="mt-3 space-y-2 text-sm">
            {data.brands.map((brand) => (
              <li key={brand.name} className="flex items-center justify-between rounded-lg bg-slate-950/60 px-3 py-2">
                <span>{brand.name}</span>
                <span className="text-slate-400">{brand.count}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">Embedding Clusters</h2>
        {data.clusters.length ? (
          <ul className="mt-3 grid gap-3 md:grid-cols-3">
            {data.clusters.map((cluster) => {
              const isActive = selectedClusterId === cluster.cluster;
              return (
                <li
                  key={cluster.cluster}
                  onClick={() => handleClusterSelect(cluster.cluster)}
                  className={`cursor-pointer rounded-lg border px-4 py-3 text-sm transition ${
                    isActive
                      ? "border-primary bg-primary/15 text-primary-light"
                      : "border-slate-800 bg-slate-950/60 text-slate-300 hover:border-primary/40 hover:text-primary-light"
                  }`}
                >
                  <p className="text-xs uppercase tracking-wide">Cluster {cluster.cluster}</p>
                  <p className="text-lg font-semibold text-white">{cluster.count}</p>
                  {isActive ? (
                    <p className="mt-1 text-[11px] text-slate-400">Tap again to collapse insights.</p>
                  ) : null}
                </li>
              );
            })}
          </ul>
        ) : (
          <p className="mt-2 text-sm text-slate-500">Run the clustering pipeline to unlock cluster analytics.</p>
        )}
      </div>

      <div ref={detailRef}>
        <ClusterDetailPanel
          clusterId={selectedClusterId}
          detail={clusterDetail}
          loading={clusterLoading}
          error={clusterError}
          onClose={() => {
            setSelectedClusterId(null);
            setClusterDetail(null);
            setClusterError(null);
          }}
        />
      </div>
    </section>
  );
}

export default AnalyticsPage;
