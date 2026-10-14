import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from campaign_train_v2.base_model import DecisionHead
from campaign_train_v2.modeling import AnalyticPolicy
from campaign_train_v2.runtime import Context, save_checkpoint, load_checkpoint
from .targets import group_targets
from .data import build_partitions, assert_disjoint
from .objective import loss, configure_head, configure_training, update
from .train import score
from .configuration import load_config, prepare_inputs


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3, bias=False)
        self.head = DecisionHead(3, 2)

    def forward(self, hidden):
        return self.head(self.backbone(hidden), [0, 1], 2)


class TinyAdaptedBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.frozen = nn.Linear(3, 3, bias=False).requires_grad_(False)
        self.backbone.lora_A = nn.Linear(3, 3, bias=False)
        self.backbone.visual = nn.Module()
        self.backbone.visual.merger = nn.Linear(3, 3, bias=False)
        self.head = DecisionHead(3, 2)

    def forward(self, hidden):
        features = self.backbone.frozen(hidden) + self.backbone.lora_A(hidden)
        features = self.backbone.visual.merger(features)
        return self.head(features, [0, 1], 2)


class CoraTests(unittest.TestCase):
    def test_full_sft_delta_scope_and_recomputed_gradient_equivalence(self):
        torch.manual_seed(51)
        model = AnalyticPolicy(TinyAdaptedBase(), actor_mode="sft_adapters")
        before = copy.deepcopy(model.state_dict())
        params = configure_training(model, "sft_adapters")
        reference = copy.deepcopy(model)
        optimizer = torch.optim.SGD(params, lr=.05)
        reference_optimizer = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=.05)
        hidden = torch.randn(1, 3, 3)
        batch = [{"record_index": i, "qid": "q", "modality": "text",
                  "gold_answer": "a" if i < 15 else "b"} for i in range(20)]
        cfg = {"bins": 1, "min_bin_count": 20, "grad_clip": 100.}
        flags = []
        def forward(_):
            flags.append(torch.is_grad_enabled())
            return model(hidden), ["a", "b"]
        ctx = Context("cpu")
        update(model, optimizer, batch, forward, ctx, cfg)
        logits = [reference(hidden) for _ in batch]
        rows = [{"key": i, "confidence": float(z.detach().softmax(-1).max()),
                 "correct": int(["a", "b"][int(z.detach().argmax())] == batch[i]["gold_answer"])}
                for i, z in enumerate(logits)]
        targets, _ = group_targets(rows, bins=1, min_count=20)
        torch.stack([loss(z, targets[i]) for i, z in enumerate(logits)]).mean().backward()
        reference_optimizer.step()
        self.assertEqual(flags, [False] * 20 + [True] * 20)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], rtol=1e-5, atol=1e-7)
        self.assertTrue(torch.equal(model.base.backbone.frozen.weight, before["base.backbone.frozen.weight"]))
        for prefix in ("base.backbone.lora_A", "base.backbone.visual.merger", "base.head"):
            self.assertTrue(any(not torch.equal(v, before[n]) for n, v in model.state_dict().items()
                                if n.startswith(prefix)), prefix)
        self.assertIsNone(model.base.backbone.frozen.weight.grad)
        ctx.close()

    def test_validation_forward_reads_dev_not_same_index_training_record(self):
        from .train import predict
        def record(name, answer):
            return {"group_id": name, "request": {"state": name, "questions": {
                "q": {"type": "choice", "instructions": "choose", "criteria": {"a": "A", "b": "B"}}}},
                "targets": {"q": {"probabilities": {"a": float(answer == "a"), "b": float(answer == "b")}}}}
        records = {"train": [record("train", "a")], "validation": [record("dev", "b")]}
        compiler = SimpleNamespace(compile=lambda r, **kwargs:
            SimpleNamespace(questions=[SimpleNamespace(qid="q", keys=["a", "b"], origin=r["group_id"])]))
        seen = []
        def model(q):
            seen.append(q.origin)
            return torch.tensor([1., 0.])
        row = predict(model, compiler, records, {"split": "validation", "record_index": 0, "qid": "q"})
        self.assertEqual(seen, ["dev"])
        self.assertEqual(row["group_id"], "dev")
        self.assertEqual(row["target"], {"a": 0., "b": 1.})

    def test_config_rejects_test_as_training_and_obsolete_dev_role(self):
        c, _ = load_config("configs/cora_train_v2/soft_sft_2b.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(dict(c, data=c["final_eval_data"])))
            with self.assertRaisesRegex(ValueError, "paths must differ"):
                load_config(path)
            path.write_text(json.dumps(dict(c, dev_data=c["data"])))
            with self.assertRaisesRegex(ValueError, "Obsolete development-set"):
                load_config(path)
            path.write_text(json.dumps(dict(c, internal_validation_fraction=0.9)))
            with self.assertRaisesRegex(ValueError, "small held-out share"):
                load_config(path)

    def test_distributed_bins_and_gradients_match_one_global_group(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=(directory,), nprocs=2, join=True)
            torch.manual_seed(10)
            model = AnalyticPolicy(TinyBase(), actor_mode="head")
            optimizer = torch.optim.AdamW(configure_head(model), lr=.02, weight_decay=0)
            hidden = torch.randn(1, 3, 3)
            batch = [{"record_index": i, "qid": "q", "modality": "text",
                      "gold_answer": "a" if i < 15 else "b"} for i in range(20)]
            cfg = {"bins": 1, "min_bin_count": 20, "grad_clip": 1.}
            ctx = Context("cpu")
            expected = update(model, optimizer, batch, lambda _: (model(hidden), ["a", "b"]), ctx, cfg)
            for rank in range(2):
                actual = torch.load(Path(directory) / f"rank-{rank}.pt", weights_only=True)
                self.assertEqual(actual["stats"]["supported_questions"], 20)
                self.assertAlmostEqual(actual["stats"]["loss"], expected["loss"], places=6)
                for key, value in model.base.head.state_dict().items():
                    torch.testing.assert_close(actual["head"][key], value, rtol=1e-5, atol=1e-7)
            ctx.close()

    def test_training_selection_and_native_evaluation_end_to_end(self):
        from .train import train
        from .evaluate import run as evaluate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial"
            initial.mkdir()
            torch.manual_seed(7)
            init_model = AnalyticPolicy(TinyBase(), actor_mode="head")
            torch.save({"trainable": init_model.base.state_dict(), "progress": {"step": 0}},
                       initial / "checkpoint.pt")
            records = []
            for i in range(20):
                records.append({"group_id": str(i), "request": {"state": f"item {i}",
                    "questions": {"q": {"type": "choice", "instructions": "choose",
                                       "criteria": {"a": "A", "b": "B"}}}},
                    "targets": {"q": {"probabilities": {"a": float(i % 3 != 0), "b": float(i % 3 == 0)}}}})
            items = [{"record_index": i, "qid": "q", "group_id": str(i),
                      "gold_answer": "a" if i % 3 else "b", "modality": "text",
                      "split": "train" if i < 16 else "validation"} for i in range(20)]
            split = {"items": items, "audit": {"questions": 20}}
            c = {"output": str(root / "trained"), "world_size": 1, "seed": 42,
                 "head_lr": .02, "max_steps": 2, "eval_every": 1, "group_questions": 8,
                 "bins": 1, "min_bin_count": 1, "grad_clip": 1.}
            base = {"init_checkpoint": str(initial), "media_root": ".", "objective": "cora_supervised_bins"}
            features = torch.randn(20, 1, 3, 3)
            def forward(model, compiler, records, item):
                return model(features[item["record_index"]]), ["a", "b"]
            with patch("campaign_train_v2.runtime.Context", side_effect=lambda _: Context("cpu")), \
                 patch("campaign_train_v2.modeling.build_model", side_effect=lambda _: AnalyticPolicy(TinyBase(), actor_mode="head")), \
                 patch("campaign_train_v2.modeling.build_compiler", return_value=None), \
                 patch("cora_train_v2.train.forward_question", side_effect=forward):
                train(c, {"fixture": True}, records, split, base)
                proof = json.loads((root / "trained/finished.json").read_text())
                self.assertTrue(proof["gradient_training"])
                self.assertFalse(proof["test_used_for_selection"])
                self.assertTrue((root / "trained/step-00002/checkpoint.pt").is_file())
                data = root / "test.jsonl"
                data.write_text("".join(json.dumps(r) + "\n" for r in records))
                source = {"final_eval_data": str(data), "expected_eval_counts": {"text": 20}}
                # Reporting protocol: only the fixed-budget final checkpoint is accepted.
                with self.assertRaisesRegex(ValueError, "fixed-budget final checkpoint"):
                    evaluate(c, source, base, root / "trained/step-00001", root / "evaluation")
                evaluate(c, source, base, root / "trained/step-00002", root / "evaluation")
                report = json.loads((root / "evaluation/comparison_metrics.json").read_text())
                self.assertFalse(report["posthoc_mapping"])
                self.assertEqual(report["metrics"]["text"]["questions"], 20)

    def test_mixed_modality_group_is_rejected(self):
        torch.manual_seed(10)
        model = AnalyticPolicy(TinyBase(), actor_mode="head")
        optimizer = torch.optim.AdamW(configure_head(model), lr=.02, weight_decay=0)
        hidden = torch.randn(1, 3, 3)
        batch = [{"record_index": i, "qid": "q", "modality": "text" if i < 10 else "image",
                  "gold_answer": "a"} for i in range(20)]
        ctx = Context("cpu")
        cfg = {"bins": 1, "min_bin_count": 1, "grad_clip": 1.}
        with self.assertRaisesRegex(ValueError, "mixes modalities"):
            update(model, optimizer, batch, lambda _: (model(hidden), ["a", "b"]), ctx, cfg)
        ctx.close()

    def test_modality_pure_groups_follow_question_share(self):
        import random
        pools = {"text": list(range(100)), "image": list(range(600)), "video": list(range(300))}
        chosen = []
        for step in range(1, 201):
            rng = random.Random(42 + step)
            names = sorted(pools)
            chosen.append(rng.choices(names, weights=[len(pools[m]) for m in names])[0])
        counts = {m: chosen.count(m) for m in pools}
        self.assertEqual(sum(counts.values()), 200)
        # Every modality is scheduled, and the ordering is proportional to pool size.
        self.assertTrue(all(counts[m] > 0 for m in pools))
        self.assertGreater(counts["image"], counts["video"])
        self.assertGreater(counts["video"], counts["text"])

    def test_config_requires_modality_confidence_grouping(self):
        c, _ = load_config("configs/cora_train_v2/soft_sft_2b.json")
        self.assertEqual(c["grouping"], "modality_confidence")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(dict(c, grouping="confidence")))
            with self.assertRaisesRegex(ValueError, "bins by confidence within one modality"):
                load_config(path)
            path.write_text(json.dumps(dict(c, target_statistic="mean_confidence")))
            with self.assertRaisesRegex(ValueError, "within-modality bin accuracy"):
                load_config(path)

    def test_twenty_cases_fifteen_hits_share_target_including_errors(self):
        rows = [{"key": i, "confidence": .81 + i * .003, "correct": int(i < 15)}
                for i in range(20)]
        targets, fitted = group_targets(rows)
        self.assertEqual(list(targets.values()), [.75] * 20)
        self.assertEqual(fitted["groups"][8]["count"], 20)

    def test_groups_and_targets_are_recomputed_from_current_outputs(self):
        first = [{"key": i, "confidence": .85, "correct": int(i < 15)} for i in range(20)]
        second = [{"key": i, "confidence": .95, "correct": int(i < 10)} for i in range(20)]
        self.assertEqual(group_targets(first)[0][0], .75)
        targets, fitted = group_targets(second)
        self.assertEqual(targets[0], .5)
        self.assertEqual(fitted["groups"][8]["count"], 0)
        self.assertEqual(fitted["groups"][9]["count"], 20)

    def test_sparse_bins_are_masked_not_assigned_identity_training_labels(self):
        rows = [{"key": i, "confidence": .85, "correct": 1} for i in range(19)]
        self.assertTrue(all(v is None for v in group_targets(rows)[0].values()))

    def test_boundaries_and_duplicates(self):
        rows = [{"key": i, "confidence": v, "correct": 1}
                for i, v in enumerate([0., .1, .8, .9, 1.])]
        _, fitted = group_targets(rows, min_count=1)
        self.assertEqual([g["count"] for g in fitted["groups"]], [1,1,0,0,0,0,0,0,1,2])
        with self.assertRaises(ValueError):
            group_targets(rows + rows[:1])

    def test_gradient_follows_own_confidence_not_gold_option(self):
        logits = torch.tensor([2., 0.], requires_grad=True)
        value = loss(logits, .75)
        value.backward()
        self.assertGreater(float(logits.grad[0]), 0)
        self.assertLess(float(logits.grad[1]), 0)

    def test_tied_probabilities_use_the_selected_answer_gradient(self):
        logits = torch.zeros(2, requires_grad=True)
        loss(logits, .75).backward()
        torch.testing.assert_close(logits.grad, torch.tensor([-.125, .125]))

    def test_head_updates_backbone_frozen_and_checkpoint_roundtrip(self):
        torch.manual_seed(10)
        model = AnalyticPolicy(TinyBase(), actor_mode="head")
        parameters = configure_head(model)
        hidden = torch.randn(1, 3, 3)
        batch = [{"record_index": i, "qid": "q", "modality": "text",
                  "gold_answer": "a" if i < 15 else "b"} for i in range(20)]
        frozen = copy.deepcopy(model.base.backbone.state_dict())
        old = copy.deepcopy(model.base.head.state_dict())
        optimizer = torch.optim.AdamW(parameters, lr=.02, weight_decay=0)
        ctx = Context("cpu")
        cfg = {"bins": 1, "min_bin_count": 1, "grad_clip": 1.}
        stats = update(model, optimizer, batch, lambda _: (model(hidden), ["a", "b"]), ctx, cfg)
        self.assertEqual(stats["supported_questions"], 20)
        self.assertTrue(any(not torch.equal(old[k], v) for k, v in model.base.head.state_dict().items()))
        for key, value in model.base.backbone.state_dict().items():
            self.assertTrue(torch.equal(value, frozen[key]))
        self.assertTrue(all(p.grad is None for p in model.base.backbone.parameters()))
        expected = model(hidden).detach()
        with tempfile.TemporaryDirectory() as d:
            progress = {"step": 1, "best_step": 1, "best_brier": .2}
            save_checkpoint(Path(d) / "step-00001", model, {"objective": "cora_supervised_bins"},
                            progress, optimizer, [])
            other = AnalyticPolicy(TinyBase(), actor_mode="head")
            self.assertEqual(load_checkpoint(other, Path(d) / "step-00001"), progress)
            self.assertTrue(torch.equal(other(hidden).detach(), expected))
            state = torch.load(Path(d) / "step-00001/training_state.pt", weights_only=False)
            resumed = torch.optim.AdamW(configure_head(other), lr=.02, weight_decay=0)
            resumed.load_state_dict(state["optimizer"])
            update(model, optimizer, batch, lambda _: (model(hidden), ["a", "b"]), ctx, cfg)
            update(other, resumed, batch, lambda _: (other(hidden), ["a", "b"]), ctx, cfg)
            self.assertTrue(torch.equal(model(hidden).detach(), other(hidden).detach()))
        ctx.close()

    def test_group_loss_matches_manual_detached_target(self):
        logits = torch.tensor([2., 0.], requires_grad=True)
        before = logits.softmax(-1).max()
        target = .75
        expected = (before - target).square()
        self.assertTrue(torch.equal(loss(logits, target), expected))

    def test_dataset_roles_and_leakage(self):
        def record(name):
            return {"group_id": name, "request": {"state": name, "questions": {
                "q": {"type": "choice", "instructions": "choose", "criteria": {"a": "A", "b": "B"}}}},
                "targets": {"q": {"probabilities": {"a": 1., "b": 0.}}}}
        training = [record(f"g{i}") for i in range(200)] + [record("g0")]
        testing = [record("test")]
        split = build_partitions(training, testing, 0.1)
        roles = {r["record_index"]: r["split"] for r in split["items"]}
        self.assertEqual(set(roles.values()), {"train", "validation"})
        self.assertNotIn("test", [r["group_id"] for r in split["items"]])
        self.assertLess(split["audit"]["validation_questions"], split["audit"]["train_questions"])
        # Same source group always lands on the same side (records 0 and 200 share g0).
        self.assertEqual(roles[0], roles[200])
        # The hash split is deterministic across calls.
        again = build_partitions(training, testing, 0.1)
        self.assertEqual(split["items"], again["items"])
        with self.assertRaisesRegex(ValueError, "request leakage"):
            renamed = copy.deepcopy(testing)
            renamed[0]["group_id"] = "fresh-group"
            renamed[0]["request"] = training[1]["request"]
            build_partitions(training, renamed, 0.1)
        with self.assertRaisesRegex(ValueError, "source-group leakage"):
            build_partitions(training, [record("g5")], 0.1)
        with self.assertRaisesRegex(ValueError, "small held-out share"):
            build_partitions(training, testing, 0.5)

    def test_real_data_split_and_config(self):
        c, source = load_config("configs/cora_train_v2/soft_sft_2b.json")
        if not Path(c["data"]).is_file():
            self.skipTest("full training mixture is not distributed with this release")
        records, split, model = prepare_inputs(c, source, thorough=False)
        self.assertEqual(split["audit"]["train_questions"], 159080)
        self.assertEqual(split["audit"]["validation_questions"], 1528)
        self.assertEqual(split["audit"]["question_counts"]["validation"],
                         {"text": 164, "image": 894, "video": 470})
        self.assertEqual(split["audit"]["test_questions"], 3124)
        self.assertEqual(set(records), {"train", "validation"})
        # Both roles index into the same training file; the split lives in the items.
        self.assertIs(records["train"], records["validation"])
        self.assertEqual(model["actor_mode"], "sft_adapters")
        self.assertTrue(model["gradient_checkpointing"])
        self.assertEqual(c["data"], source["data"])
        self.assertIn("trimodal_full_train.jsonl", c["data"])
        self.assertGreaterEqual(split["audit"]["train_questions"], c["group_questions"])

    def test_final_metrics_follow_changed_answers(self):
        before = [{"probabilities": {"a": .8, "b": .2}, "target": {"a": 1., "b": 0.}}]
        after = [{"probabilities": {"a": .4, "b": .6}, "target": {"a": 1., "b": 0.}}]
        self.assertEqual(score(before)["acc"], 1.)
        self.assertEqual(score(after)["acc"], 0.)
        self.assertAlmostEqual(score(after)["binary"]["correctness_brier"], .36)


def distributed_worker(rank, directory):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + str(Path(directory) / "rendezvous"),
                            rank=rank, world_size=2)
    context = Context("cpu")
    context.rank, context.world_size = rank, 2
    torch.manual_seed(10)
    model = AnalyticPolicy(TinyBase(), actor_mode="head")
    optimizer = torch.optim.AdamW(configure_head(model), lr=.02, weight_decay=0)
    hidden = torch.randn(1, 3, 3)
    batch = [{"record_index": i, "qid": "q", "modality": "text",
              "gold_answer": "a" if i < 15 else "b"} for i in range(20)]
    cfg = {"bins": 1, "min_bin_count": 20, "grad_clip": 1.}
    stats = update(model, optimizer, batch, lambda _: (model(hidden), ["a", "b"]), context, cfg)
    torch.save({"head": model.base.head.state_dict(), "stats": stats}, Path(directory) / f"rank-{rank}.pt")
    context.close()


if __name__ == "__main__":
    unittest.main()
