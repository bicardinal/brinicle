import brinicle

engine = brinicle.AutocompleteEngine("examples/indexes/autocomplete_search", dim=48)

engine.init("build")

engine.ingest("iphone 15 pro max", "iphone 15 pro max")
engine.ingest("iphone 15 case", "iphone 15 case")
engine.ingest("samsung galaxy s24 ultra", "samsung galaxy s24 ultra")

engine.finalize()

print(engine.search("iphone 15", k=3))
print(engine.search_with_distance("iphone 15", k=3))