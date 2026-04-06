from brinicle import product_feed
import brinicle
import json
import os
import numpy
import xxhash
import time
from examples.clean import html_clean
from examples.create_gt import build_bruteforce_gt

tokenizer, pad_id, unk_id = product_feed.load_tokenizer("brinicle/pe_tokenizer_4k.json")

dim = 96
K = 200
EFS = 512
EFC = 400
M = 32
GT_K = 200

def recognize_english(slug):
    if not slug:
        return False
    ords = [ord(x) <= 256 for x in slug]
    return sum(ords) / len(slug) >= 0.9

def return_elements():
	if not os.path.exists("examples/elements.json"):
		big_table = json.load(open("examples/_big_table.json"))
		digikala = json.load(open("examples/digikala_products.json"))

		rep = set()
		for x in digikala:
			category = x["category"]
			for y in x["items"]:
				y["category"] = category
				title = y["title"]
				slug = y["english_title"]
				y["real_title"] = title
				if title != slug:
					y["title"] = f"{title} || {slug}"
				pid = xxhash.xxh32(y["href"]).hexdigest()
				if pid not in rep:
					y["pid"] = pid
					elements.append(y)
					rep.add(pid)
		for x in big_table:
			title = x['title']
			slug = x['urlpath']
			x["real_title"] = title
			if recognize_english(slug):
				if title != slug:
					x["title"] = f"{title} || {slug}"
			if x["pid"] not in rep:
				elements.append(x)
				rep.add(x["pid"])
		del digikala
		del big_table
		print(len(elements))
		json.dump(elements, open("examples/elements.json", "w"))
		exit()
	else:
		elements = json.load(open("examples/elements.json"))
	return elements

elements = return_elements()



def create_X():
	X = []
	ids = []
	all_categories = set()
	pid_to_row = {}
	for i, x in enumerate(elements):
		category = html_clean(x["category"].lower())
		all_categories.add(category)
		vec = product_feed.encode_product_vector(
			tokenizer=tokenizer,
			pad_id=pad_id,
			unk_id=unk_id,
			title=html_clean(x["title"]),
			attributes={},
			category=category,
			# brand=html_clean(x["brand"].lower()),
			brand=None,
			max_dim=dim,
		)
		X.append(vec)
		ids.append(x["pid"])
		pid_to_row[x["pid"]] = i

	return X, ids, pid_to_row

X, ids, pid_to_row = create_X()
X = numpy.array(X, dtype=numpy.float32)

def create_Q():
	Q = []
	for x in queries:
		vec = product_feed.encode_query_vector(
			tokenizer=tokenizer,
			pad_id=pad_id,
			unk_id=unk_id,
			query=html_clean(x["query"]),
			max_dim=dim,
			category=x["category"],
		)
		Q.append(vec)
	return Q

def compute_recalls(pred_ids, gt_ids, K):
	nq = gt_ids.shape[0]
	K = min(K, pred_ids.shape[1], gt_ids.shape[1])
	hits = 0
	for i in range(nq):
		a = pred_ids[i, :K]
		b = gt_ids[i, :K]
		hits += len(set(a.tolist()) & set(b.tolist()))
	return {f"recall@{K}": hits / (nq * K)}


def compute_pool_hit(pred_ids, gt_ids, K, pool_k=100):
	nq = gt_ids.shape[0]
	K = min(K, pred_ids.shape[1])
	pool_k = min(pool_k, gt_ids.shape[1])
	hits = 0
	for i in range(nq):
		a = set(pred_ids[i, :K].tolist())
		b = set(gt_ids[i, :pool_k].tolist())
		if a & b:
			hits += 1
	return {f"hit@{K}_in_top{pool_k}": hits / nq}


def compute_top1_metrics(pred_ids, gt_ids, K):
	nq = gt_ids.shape[0]
	K = min(K, pred_ids.shape[1])

	top1_in_topk = 0
	top1_exact_match = 0
	mrr_top1 = 0.0

	for i in range(nq):
		pred = pred_ids[i, :K].tolist()
		gt1 = int(gt_ids[i, 0])

		if pred and pred[0] == gt1:
			top1_exact_match += 1

		try:
			rank = pred.index(gt1) + 1
			top1_in_topk += 1
			mrr_top1 += 1.0 / rank
		except ValueError:
			pass

	return {
		f"top1_in_top{K}": top1_in_topk / nq,
		"top1_exact_match": top1_exact_match / nq,
		f"mrr_top1@{K}": mrr_top1 / nq,
	}


def compute_survival_metrics(pred_ids, gt_ids, pred_k, gt_prefixes=(1, 10, 20, 50, 100)):
	nq = gt_ids.shape[0]
	pred_k = min(pred_k, pred_ids.shape[1])

	out = {}
	valid_prefixes = []
	for L in gt_prefixes:
		if L >= 1:
			valid_prefixes.append(min(L, gt_ids.shape[1]))

	# remove duplicates while preserving order
	seen = set()
	valid_prefixes = [x for x in valid_prefixes if not (x in seen or seen.add(x))]

	for L in valid_prefixes:
		total_hits = 0
		any_hits = 0

		for i in range(nq):
			a = set(pred_ids[i, :pred_k].tolist())
			b = set(gt_ids[i, :L].tolist())
			inter = len(a & b)
			total_hits += inter
			if inter > 0:
				any_hits += 1

		out[f"survival@{pred_k}_of_gt{L}"] = total_hits / (nq * L)
		out[f"any_gt{L}_in_top{pred_k}"] = any_hits / nq

	return out


def compute_query_recall_distribution(pred_ids, gt_ids, K):
	K = min(K, pred_ids.shape[1], gt_ids.shape[1])
	nq = gt_ids.shape[0]
	per_query = numpy.empty(nq, dtype=numpy.float32)

	for i in range(nq):
		a = set(pred_ids[i, :K].tolist())
		b = set(gt_ids[i, :K].tolist())
		per_query[i] = len(a & b) / K
		if per_query[i] == 0:
			print(i)
	return {
		f"query_recall@{K}_mean": float(numpy.mean(per_query)),
		f"query_recall@{K}_p50": float(numpy.percentile(per_query, 50)),
		f"query_recall@{K}_p95": float(numpy.percentile(per_query, 95)),
		f"query_recall@{K}_min": float(numpy.min(per_query)),
	}


def brinicle_build(X, M, EFC, EFS, index_name="examples/product_index"):
	client = brinicle.VectorEngine(index_name, X.shape[1], 0.1, M=M, ef_construction=EFC, ef_search=EFS, seed=123, dist_func="lexical")
	t0 = time.perf_counter()
	client.init(mode="build")
	for b in range(X.shape[0]):
		client.ingest(
			external_id=str(b),
			vec=X[b],
		)
		print(b, end='\r')
	client.finalize()
	build_latency = time.perf_counter() - t0
	print(f"[build] Done in {build_latency:.3f}s.")
	return build_latency


def brinicle_search(build_latency, Q, GT, GT_K, K, EFS, index_name="examples/product_index", second_stage=False):
	client = brinicle.VectorEngine(index_name, Q.shape[1], 0.1, dist_func="lexical")

	# Queries
	nq_total = Q.shape[0]

	preds = numpy.empty((nq_total, K), dtype=numpy.int64)

	if nq_total == 0:
		raise ValueError("No queries available.")

	# warm up
	client.search(Q[0], k=K, efs=EFS)

	print(f"[search] Running {nq_total} queries @ top-{K}...")
	t1 = time.perf_counter()
	query_latencies = []

	short_return_count = 0
	returned_counts = []
	unique_returned_counts = []

	for i, v in enumerate(Q, start=0):
		q0 = time.perf_counter()
		raw_labels = client.search(v, k=K, efs=EFS)  # must return list/iterable of IDs
		q1 = time.perf_counter()

		query_latencies.append(q1 - q0)

		labels = [int(x) for x in raw_labels]
		returned_counts.append(len(labels))
		unique_returned_counts.append(len(set(labels)))

		if len(labels) < K:
			short_return_count += 1
			labels = list(labels) + [-1] * (K - len(labels))

		preds[i, :] = numpy.asarray(labels[:K], dtype=numpy.int64)
		print(i, end="\r")

	t2 = time.perf_counter()

	search_wall = t2 - t1
	query_latencies = numpy.array(query_latencies)
	avg_latency = numpy.mean(query_latencies)
	p50_latency = numpy.percentile(query_latencies, 50)
	p95_latency = numpy.percentile(query_latencies, 95)
	p99_latency = numpy.percentile(query_latencies, 99)
	total_q_time = numpy.sum(query_latencies)
	qps = nq_total / total_q_time if total_q_time > 0 else float("inf")
	if second_stage:
		results = {
			"queries": int(nq_total),
			"params": {"M": M, "ef_construction": EFC, "ef_search": EFS, "seed": 123},
			"build_latency": float(build_latency),
			"search_avg_latency": float(avg_latency),
			"search_p50_latency": float(p50_latency),
			"search_p95_latency": float(p95_latency),
			"search_p99_latency": float(p99_latency),
			"qps": float(qps),
			"search_wall_time": float(search_wall)
		}
		return results, preds


	# Existing overlap metrics
	recalls = compute_recalls(preds, GT, K)
	loose_recall = compute_pool_hit(preds, GT, K=K, pool_k=GT_K)

	# New important metrics
	top1_metrics = compute_top1_metrics(preds, GT, K=K)
	survival_metrics = compute_survival_metrics(
		preds,
		GT,
		pred_k=K,
		gt_prefixes=(1, 10, 20, 50, 100),
	)
	query_recall_distribution = compute_query_recall_distribution(preds, GT, K=K)

	results = {
		"queries": int(nq_total),
		"params": {"M": M, "ef_construction": EFC, "ef_search": EFS, "seed": 123},
		"build_latency": float(build_latency),
		"search_avg_latency": float(avg_latency),
		"search_p50_latency": float(p50_latency),
		"search_p95_latency": float(p95_latency),
		"search_p99_latency": float(p99_latency),
		"qps": float(qps),
		"search_wall_time": float(search_wall),

		# sanity / integrity checks
		"returned_less_than_k": int(short_return_count),
		"avg_returned_k": float(numpy.mean(returned_counts)),
		"avg_unique_returned_k": float(numpy.mean(unique_returned_counts)),

		# metrics
		"loose_recall": loose_recall,
		"top1_metrics": top1_metrics,
		"survival_metrics": survival_metrics,
		"query_recall_distribution": query_recall_distribution,
	}

	results.update(recalls)

	return results, preds


if __name__ == "__main__":
	queries = [{"query": q.split(" : ")[0], "category": q.split(" : ")[2], "brand": q.split(" : ")[1]} for q in open("examples/queries.txt").read().split("\n")]



	Q = numpy.array(create_Q(), dtype=numpy.float32)

	path=f"examples/gt_{GT_K}.npy"
	if os.path.exists(path):
		GT = numpy.load(path)
	else:
		GT = build_bruteforce_gt(X, Q, gt_k=GT_K)
		numpy.save(path.replace(".npy",""), GT)

	# build_latency = brinicle_build(X, M, EFC, EFS)
	results, preds = brinicle_search(46.116647538001416, Q, GT, GT_K, K, EFS)

	print(json.dumps(results, indent=4))
