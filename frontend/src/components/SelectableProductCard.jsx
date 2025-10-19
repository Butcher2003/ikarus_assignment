import { ASSET_BASE_URL } from "../lib/api";

const PLACEHOLDER = "https://via.placeholder.com/320x240?text=No+Image";

function toAbsoluteUrl(path) {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  return `${ASSET_BASE_URL}${path}`;
}

function SelectableProductCard({ product, selected, onToggle }) {
  const {
    uniq_id: id,
    title,
    brand,
    generated_description: generatedDescription,
    image_url: imageUrl,
    categories,
    price,
  } = product;

  const displayPrice = typeof price === "number" && !Number.isNaN(price) ? `$${price.toFixed(2)}` : "N/A";
  const imageSrc = toAbsoluteUrl(imageUrl) || PLACEHOLDER;

  return (
    <label
      htmlFor={`select-${id}`}
      className={`group relative flex h-full cursor-pointer flex-col overflow-hidden rounded-xl border transition ${
        selected ? "border-primary bg-primary/10" : "border-slate-800 bg-slate-900/60 hover:border-primary/60"
      }`}
    >
      <input
        id={`select-${id}`}
        type="checkbox"
        checked={selected}
        onChange={() => onToggle(id)}
        className="absolute right-3 top-3 h-4 w-4 accent-primary"
      />
      <img src={imageSrc} alt={title} className="h-40 w-full object-cover" loading="lazy" />
      <div className="flex flex-1 flex-col gap-2 p-4">
        <div>
          <h3 className="text-sm font-semibold text-white">{title}</h3>
          {brand ? <p className="text-xs uppercase tracking-wide text-primary-light">{brand}</p> : null}
        </div>
        {generatedDescription ? (
          <p className="text-xs text-slate-300">{generatedDescription}</p>
        ) : (
          <p className="text-xs text-slate-500">Select to include this product in the generated description.</p>
        )}
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span className="font-medium text-slate-200">{displayPrice}</span>
          <span>{categories?.[0] || ""}</span>
        </div>
      </div>
    </label>
  );
}

export default SelectableProductCard;
