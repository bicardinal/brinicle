from brinicle import product_feed
import brinicle
import json
import numpy
from examples.clean import html_clean

from examples.candidate_select_eval import brinicle_build, brinicle_search, X, tokenizer, pad_id, unk_id, pid_to_row
from examples.debug import inspect_misses


def compute_single_positive_metrics(pred_ids, gt_ids, ks=(1, 5, 10, 20, 50, 100, 200)):
	nq = len(gt_ids)
	out = {}

	max_pred_k = pred_ids.shape[1]
	ks = [k for k in ks if k <= max_pred_k]

	for k in ks:
		hits = 0
		rr = 0.0

		for i in range(nq):
			pred = pred_ids[i, :k].tolist()
			gt = int(gt_ids[i])

			if gt in pred:
				hits += 1
				rr += 1.0 / (pred.index(gt) + 1)

		out[f"hit@{k}"] = hits / nq
		out[f"mrr@{k}"] = rr / nq

	# exact rank-1 accuracy
	out["top1_accuracy"] = float((pred_ids[:, 0] == gt_ids).mean())

	# rank stats over full returned list
	found_ranks = []
	not_found = 0
	full_k = pred_ids.shape[1]

	for i in range(nq):
		pred = pred_ids[i, :full_k].tolist()
		gt = int(gt_ids[i])

		if gt in pred:
			found_ranks.append(pred.index(gt) + 1)
		else:
			not_found += 1

	out["not_found_rate"] = not_found / nq
	if found_ranks:
		out["mean_found_rank"] = float(numpy.mean(found_ranks))
		out["p50_found_rank"] = float(numpy.percentile(found_ranks, 50))
		out["p95_found_rank"] = float(numpy.percentile(found_ranks, 95))
	else:
		out["mean_found_rank"] = None
		out["p50_found_rank"] = None
		out["p95_found_rank"] = None

	return out


if __name__ == "__main__":
	synthetic = json.load(open("examples/evals/_final_pairs.json"))
	elements = json.load(open("examples/elements.json"))

	flat_queries = []
	for pid, item in synthetic.items():
		for q in item["queries"]:
			q = q.strip()
			if not q:
				continue
			flat_queries.append({
				"query": q,
				"pid": pid,
				"category": item.get("category", None),   # optional
				"brand": item.get("brand", None),         # optional
			})

	Q = []
	GT_exact = []
	dim = 96
	K = 200
	EFS = 512
	EFC = 1024
	M = 48
	GT_K = 200

	for item in flat_queries:
		pid = item["pid"]
		if pid not in pid_to_row:
			continue

		vec = product_feed.encode_query_vector(
			tokenizer=tokenizer,
			pad_id=pad_id,
			unk_id=unk_id,
			query=html_clean(item["query"]),
			max_dim=dim,
			category=item["category"],
			brand=None,
		)

		Q.append(vec)
		GT_exact.append(pid_to_row[pid])

	Q = numpy.array(Q, dtype=numpy.float32)
	GT_exact = numpy.array(GT_exact, dtype=numpy.int64)


	index_name = "examples/semantic_eval_index"
	build_latency = brinicle_build(X, M, EFC, EFS, index_name=index_name)
	results, preds = brinicle_search(build_latency, Q, GT_exact, GT_K, K, EFS, index_name=index_name, second_stage=True)
	results["semantic_metrics"] = compute_single_positive_metrics(preds, GT_exact)
	print(json.dumps(results, indent=4))

	inspect_misses(flat_queries, preds, GT_exact, elements, X, Q, top_n=200)
