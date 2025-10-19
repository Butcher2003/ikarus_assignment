import { ASSET_BASE_URL } from "../lib/api";

function toAbsoluteUrl(path) {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  return `${ASSET_BASE_URL}${path}`;
}

function ProductCard({ product }) {
  const {
    title,
    brand,
    generated_description: generatedDescription,
    categories,
    image_url: imageUrl,
    price,
    similarity,
  } = product;
  const placeholder = "https://via.placeholder.com/320x240?text=No+Image";
  const imageSrc = toAbsoluteUrl(imageUrl) || placeholder;

  return (
    <article className="flex h-full flex-col overflow-hidden rounded-xl border border-slate-800 bg-slate-900/70">
      <img
        src={imageSrc}
        alt={title}
        className="h-48 w-full object-cover"
        loading="lazy"
      />
      <div className="flex flex-1 flex-col gap-3 p-4">
        <header>
          <h3 className="text-base font-semibold text-white">{title}</h3>
          {brand ? <p className="text-xs uppercase tracking-wide text-primary-light">{brand}</p> : null}
        </header>
        <p className="flex-1 text-sm text-slate-300">{generatedDescription}</p>
        <dl className="grid grid-cols-2 gap-2 text-xs text-slate-400">
          <div>
            <dt className="uppercase tracking-wide text-slate-500">Confidence</dt>
            <dd>{(similarity * 100).toFixed(1)}%</dd>
          </div>
          <div>
            <dt className="uppercase tracking-wide text-slate-500">Price</dt>
            <dd>{price ? `$${price.toFixed(2)}` : "N/A"}</dd>
          </div>
        </dl>
        <div className="flex flex-wrap gap-2 text-[11px] text-slate-400">
          {categories?.slice(0, 3).map((category) => (
            <span key={category} className="rounded-full bg-slate-800 px-3 py-1">
              {category}
            </span>
          ))}
        </div>
      </div>
    </article>
  );
}

export default ProductCard;
