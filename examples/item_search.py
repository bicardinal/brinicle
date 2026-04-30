import brinicle

engine = brinicle.ItemSearchEngine("examples/indexes/item_search", dim=96)

engine.init("build")

engine.ingest(
    external_id="p1",
    title="Apple iPhone 15 Pro Max 256GB Natural Titanium",
    category="Electronics",
    subcategory="Smartphones",
    attributes={"brand": "Apple", "storage": "256GB", "color": "Natural Titanium"},
)

engine.ingest(
    external_id="p2",
    title="Samsung Galaxy S24 Ultra 512GB Black",
    category="Electronics",
    subcategory="Smartphones",
    attributes={"brand": "Samsung", "storage": "512GB", "color": "Black"},
)

engine.finalize()

print(engine.search("iphone 15 pro max", k=2))
print(engine.search_with_distance("iphone 15", k=2))