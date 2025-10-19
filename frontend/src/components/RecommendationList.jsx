import ProductCard from "./ProductCard";

function RecommendationList({ items }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {items.map((item) => (
        <ProductCard key={item.uniq_id} product={item} />
      ))}
    </div>
  );
}

export default RecommendationList;
